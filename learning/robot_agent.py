"""Days 10-11: Safe Task Execution Engine + Official Gemini Interactions Robotics Agent.

Read this file in 2 easy parts:
  PART 1 — `TaskExecutionEngine` (The Safety Manager):
    - Holds a single-arm lock so two tasks can never collide.
    - Runs an explicit state machine:
        IDLE -> PLANNING -> APPROACH -> GRASP -> LIFT -> MOVE -> PLACE -> DONE
        (or CANCELING -> RECOVERING back to safe Home pose on cancel/timeout/fault!)
    - Blocks unsafe AI guesses (out of reach or below table z < 0.035m).
    - Converts 2D normalized image points [y, x] (0..1000) into exact 3D meters
      using our live Depth camera (`/sim/depth`) + TF2 (`world <- overview_camera`)!

  PART 2 — `run_gemini_task` (Official 2026 Google Gemini Interactions API):
    - Follows Google's official Robotics Task Orchestration pattern (`client.interactions.create`)
      with `google-genai >= 3.2.0` and server-side state (`previous_interaction_id`).
    - Supports `gemini-robotics-er-2-preview`, `gemini-4-argon`, `gemini-4-hydrogen`, `gemini-4-carbon`.
    - Supports both standard Gemini API keys (`GEMINI_API_KEY`) and Google Cloud Enterprise /
      Vertex AI mode (`GOOGLE_GENAI_USE_ENTERPRISE=true`).
    - Includes a free Local Fallback Mode when no API key is set so you can test offline anytime!
"""

import base64
import json
import math
import os
import threading
import time
from typing import Callable

import numpy as np


# Safe Home pose above the table used for recovery after cancel/timeout/fault.
HOME_XYZ = (0.25, 0.00, 0.36)

# Tool schemas passed directly to Google's official `client.interactions.create(tools=...)`.
ROBOT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "name": "pick_and_place",
        "description": (
            "Safely picks up an object at normalized image coordinates [pick_y, pick_x] (0-1000 scale) "
            "and places it at [place_y, place_x] (0-1000 scale). Depth camera + TF2 automatically "
            "convert the 2D image points into 3D robot coordinates."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pick_y": {"type": "integer", "description": "Normalized Y (0=top, 1000=bottom) of object to pick"},
                "pick_x": {"type": "integer", "description": "Normalized X (0=left, 1000=right) of object to pick"},
                "place_y": {"type": "integer", "description": "Normalized Y (0=top, 1000=bottom) of target destination"},
                "place_x": {"type": "integer", "description": "Normalized X (0=left, 1000=right) of target destination"},
                "label": {"type": "string", "description": "Short description, e.g. 'yellow block -> green pad'"},
            },
            "required": ["pick_y", "pick_x", "place_y", "place_x"],
        },
    },
    {
        "type": "function",
        "name": "hover_above",
        "description": "Moves the robot gripper to hover 10 cm above the object at normalized image point [y, x] (0-1000).",
        "parameters": {
            "type": "object",
            "properties": {
                "y": {"type": "integer", "description": "Normalized Y (0=top, 1000=bottom)"},
                "x": {"type": "integer", "description": "Normalized X (0=left, 1000=right)"},
                "label": {"type": "string", "description": "Name of object being pointed to"},
            },
            "required": ["y", "x"],
        },
    },
    {
        "type": "function",
        "name": "set_gripper",
        "description": "Opens (opened=True) or closes (opened=False) the robot gripper fingers.",
        "parameters": {
            "type": "object",
            "properties": {
                "opened": {"type": "boolean", "description": "True to open fingers, False to close fingers"},
            },
            "required": ["opened"],
        },
    },
    {
        "type": "function",
        "name": "go_home",
        "description": "Returns the robot arm smoothly to its safe neutral Home pose above the table.",
        "parameters": {"type": "object", "properties": {}},
    },
]


class TaskExecutionEngine:
    """Deterministic robotics runtime safety gate & state machine.

    Any AI brain (Gemini Robotics ER 2, Gemini 4 Argon, or Local Mode) calls ONLY these
    safe methods — it can never bypass the arm lock, workspace bounds, or cancel/recovery logic.
    """

    def __init__(
        self,
        move_arm_sync: Callable[[float, float, float, float, threading.Event], tuple[bool, str]],
        set_gripper_sync: Callable[[bool], tuple[bool, str]],
        norm_point_to_3d: Callable[[int, int], tuple[float, float, float] | None],
        get_camera_snapshot: Callable[[], tuple[bytes, np.ndarray | None]],
        get_gripper_fingers: Callable[[], tuple[float, float]] | None = None,
    ) -> None:
        self._move_arm_sync = move_arm_sync
        self._set_gripper_sync = set_gripper_sync
        self._norm_point_to_3d = norm_point_to_3d
        self._get_camera_snapshot = get_camera_snapshot
        self._get_gripper_fingers = get_gripper_fingers

        self._lock = threading.Lock()
        self._arm_busy = False
        self._cancel_event = threading.Event()
        self._simulate_fault: str | None = None

        self.state: str = "IDLE"
        self.trace: list[str] = []
        self.last_summary: str = "Idle — enter a task prompt above."
        self._task_t0: float = 0.0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "busy": self._arm_busy,
                "summary": self.last_summary,
                "trace": list(self.trace[-15:]),
            }

    def _set_state(self, new_state: str, note: str | None = None) -> None:
        with self._lock:
            self.state = new_state
            if note:
                dt = time.monotonic() - self._task_t0 if self._task_t0 else 0.0
                self.trace.append(f"[+{dt:4.1f}s {new_state}] {note}")
                self.last_summary = note

    def request_cancel(self) -> dict:
        self._cancel_event.set()
        with self._lock:
            if self._arm_busy:
                self.state = "CANCELING"
                self.trace.append("[CANCELING] Operator requested immediate cancel!")
                return {"ok": True, "message": "Task cancel requested — recovering safely"}
        return {"ok": False, "message": "No running task to cancel"}

    def _check_safe_xyz(self, x: float, y: float, z: float) -> tuple[bool, str]:
        """Hard safety guard: AI hallucinations can NEVER send the arm underground, into its base, or out of reach."""
        dist = math.dist((0.0, 0.0, 0.14), (x, y, z))
        xy_dist = math.hypot(x, y)
        if z < 0.035:
            return False, f"Blocked unsafe z={z:.3f}m (below table safety limit 0.035m)"
        if xy_dist < 0.18:
            return False, f"Blocked self-collision target ({x:.2f}, {y:.2f}) too close to robot base (r={xy_dist:.2f}m < 0.18m)"
        if dist > 0.54:
            return False, f"Blocked unreachable target ({x:.2f}, {y:.2f}, {z:.2f}) dist={dist:.2f}m > 0.54m"
        return True, "ok"

    def _safe_move(self, state_name: str, x: float, y: float, z: float, timeout_s: float = 10.0) -> tuple[bool, str]:
        if self._cancel_event.is_set():
            return False, "Canceled before motion"
        ok_bounds, reason = self._check_safe_xyz(x, y, z)
        if not ok_bounds:
            self._set_state("SAFETY_BLOCK", reason)
            return False, reason

        self._set_state(state_name, f"Moving arm to ({x:.2f}, {y:.2f}, {z:.2f})")
        if self._simulate_fault == "timeout" and state_name == "MOVE":
            # Injected fault for Days 10-11 reliability testing: pretend motion stalled!
            self._simulate_fault = None
            return False, "FAULT INJECTED: Motion timeout mid-carry!"

        return self._move_arm_sync(x, y, z, timeout_s, self._cancel_event)

    def _recover_home(self, reason: str) -> None:
        """Safe recovery sequence: open gripper & lift cleanly back to Home pose."""
        self._set_state("RECOVERING", f"Recovering to Home ({reason})")
        self._set_gripper_sync(False)  # open fingers safely
        # Clear cancel briefly just for the safe return-home motion
        recovery_cancel = threading.Event()
        self._move_arm_sync(*HOME_XYZ, 5.0, recovery_cancel)

    # --- High-level tool handlers called by Gemini or Local Mode ---

    def tool_go_home(self) -> dict:
        ok, msg = self._safe_move("HOME", *HOME_XYZ)
        return {"ok": ok, "message": msg}

    def tool_set_gripper(self, opened: bool) -> dict:
        if self._cancel_event.is_set():
            return {"ok": False, "message": "Canceled"}
        self._set_state("GRIPPER", "Opening gripper" if opened else "Closing gripper")
        ok, msg = self._set_gripper_sync(not opened)
        time.sleep(0.35)
        return {"ok": ok, "message": msg}

    def tool_hover_above(self, y: int, x: int, label: str = "target") -> dict:
        xyz = self._norm_point_to_3d(int(y), int(x))
        if xyz is None:
            return {"ok": False, "message": f"Depth/TF unavailable for pixel [{y}, {x}]"}
        tx, ty, tz = xyz
        hover_z = max(0.12, tz + 0.09)
        self._set_state("PERCEIVE", f"{label} pixel [{y},{x}] -> 3D ({tx:.2f}, {ty:.2f}, {tz:.2f})")
        ok, msg = self._safe_move("APPROACH", tx, ty, hover_z)
        return {"ok": ok, "world_xyz": [round(tx, 3), round(ty, 3), round(tz, 3)], "message": msg}

    def tool_pick_and_place(
        self, pick_y: int, pick_x: int, place_y: int, place_x: int, label: str = "object"
    ) -> dict:
        p_pick = self._norm_point_to_3d(int(pick_y), int(pick_x))
        p_place = self._norm_point_to_3d(int(place_y), int(place_x))
        if p_pick is None or p_place is None:
            return {"ok": False, "message": "Could not project image points via Depth + TF2"}

        px, py, pz = p_pick
        dx, dy, dz = p_place
        self._set_state(
            "PERCEIVE",
            f"{label}: pick ({px:.2f},{py:.2f},{pz:.2f}) -> place ({dx:.2f},{dy:.2f},{dz:.2f})",
        )

        # Deterministic 9-step top-down pick-and-place state sequence with closed-loop grasp check!
        grasp_z = max(0.036, min(0.046, pz + 0.010))
        steps = [
            ("OPEN", lambda: self._set_gripper_sync(False)),
            ("APPROACH", lambda: self._safe_move("APPROACH", px, py, 0.15)),
            ("DESCEND", lambda: self._safe_move("DESCEND", px, py, grasp_z)),
            ("GRASP", lambda: self._grasp_and_verify()),
            ("LIFT", lambda: self._safe_move("LIFT", px, py, 0.20)),
            ("MOVE", lambda: self._safe_move("MOVE", dx, dy, 0.20)),
            ("PLACE", lambda: self._safe_move("PLACE", dx, dy, 0.055)),
            ("RELEASE", lambda: self._release_open()),
            ("CLEAR", lambda: self._safe_move("CLEAR", dx, dy, 0.20)),
        ]
        for step_name, fn in steps:
            if self._cancel_event.is_set():
                self._recover_home("canceled by operator")
                return {"ok": False, "message": f"Canceled during {step_name} and recovered Home"}
            ok, msg = fn()
            if not ok:
                self._recover_home(f"{step_name} failed: {msg}")
                return {"ok": False, "message": f"Aborted at {step_name} ({msg}); recovered Home"}
        return {"ok": True, "message": f"Verified pick & placed ({label})!"}

    def _grasp_and_verify(self) -> tuple[bool, str]:
        """Close jaws and verify via finger joint encoders whether an object is actually held!"""
        self._set_state("GRASP", "Closing vertical rubber jaws around object...")
        ok, msg = self._set_gripper_sync(True)
        if not ok:
            return False, msg
        # Wait long enough for physics under software rendering to squeeze firmly and settle
        time.sleep(1.5)
        if self._get_gripper_fingers is not None:
            f_left, f_right = self._get_gripper_fingers()
            # Max inner opening is 8.4 cm; empty closed reaches ~0.035m each (gap ~ 1.4cm).
            gap_cm = max(0.0, (0.084 - (f_left + f_right)) * 100.0)
            if f_left >= 0.031 and f_right >= 0.031:
                self._set_state(
                    "GRASP_MISS",
                    f"Gripper closed on EMPTY AIR (jaw gap={gap_cm:.1f}cm) — object not picked!",
                )
                return False, f"Grasp missed: fingers closed on empty air (gap={gap_cm:.1f}cm)"
            if min(f_left, f_right) < 0.007 or abs(f_left - f_right) > 0.015:
                self._set_state(
                    "GRASP_MISS",
                    f"Off-center graze (fingers={f_left:.3f}/{f_right:.3f}m) — object not securely pinched!",
                )
                return False, f"Grasp missed: off-center graze (fingers={f_left:.3f}/{f_right:.3f}m)"
            self._set_state(
                "GRASP_OK",
                f"Verified object gripped! (jaw gap={gap_cm:.1f}cm, fingers={f_left:.3f}/{f_right:.3f}m)",
            )
        return True, "Grasp verified"

    def _release_open(self) -> tuple[bool, str]:
        self._set_state("RELEASE", "Opening gripper to set object down gently")
        ok, msg = self._set_gripper_sync(False)
        time.sleep(0.35)
        return ok, msg

    def dispatch_tool(self, name: str, arguments: dict) -> dict:
        if name == "pick_and_place":
            return self.tool_pick_and_place(**arguments)
        if name == "hover_above":
            return self.tool_hover_above(**arguments)
        if name == "set_gripper":
            return self.tool_set_gripper(**arguments)
        if name == "go_home":
            return self.tool_go_home()
        return {"ok": False, "message": f"Unknown tool: {name}"}

    def start_task_async(
        self, prompt: str, mode: str = "auto", model: str | None = None, fault: str | None = None
    ) -> dict:
        with self._lock:
            if self._arm_busy:
                return {"ok": False, "message": f"Arm busy (state={self.state}) — cancel first or wait"}
            self._arm_busy = True
            self._cancel_event.clear()
            self._simulate_fault = fault
            self._task_t0 = time.monotonic()
            self.trace = [f"[+ 0.0s LOCKED] Prompt: {prompt!r} (mode={mode})"]
            self.state = "PLANNING"
            self.last_summary = f"Planning: {prompt}"

        t = threading.Thread(
            target=self._run_worker, args=(prompt, mode, model), daemon=True
        )
        t.start()
        return {"ok": True, "message": "Task started"}

    def _run_worker(self, prompt: str, mode: str, model: str | None) -> None:
        try:
            png_bytes, rgb_small = self._get_camera_snapshot()
            if not png_bytes:
                self._set_state("FAILED", "No camera frame available yet")
                return

            has_key = bool(
                os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("GOOGLE_CLOUD_PROJECT")
            )
            use_gemini = (mode == "gemini") or (mode == "auto" and has_key)

            if use_gemini:
                summary = run_gemini_interactions_agent(
                    prompt=prompt,
                    png_bytes=png_bytes,
                    engine=self,
                    model_id=model or os.environ.get("GEMINI_MODEL", "gemini-4-hydrogen"),
                )
            else:
                summary = run_local_color_planner(prompt=prompt, rgb_small=rgb_small, engine=self)

            if self._cancel_event.is_set():
                self._set_state("CANCELED", summary)
            elif self.state not in ("FAILED", "RECOVERING", "SAFETY_BLOCK"):
                self._set_state("DONE", summary)
        except Exception as exc:
            self._recover_home(f"agent error: {exc}")
            self._set_state("FAILED", f"Error: {exc}")
        finally:
            with self._lock:
                self._arm_busy = False
                if self.state in ("DONE", "CANCELED", "FAILED", "RECOVERING"):
                    pass  # keep terminal state visible in UI until next task


# ==============================================================================
# PART 2: Official 2026 Google Gemini Interactions API Agent (+ Local Fallback)
# ==============================================================================


def build_genai_client(model_id: str = ""):
    """Create a `google-genai` Client for Gemini Enterprise (ADC) or Gemini Developer API."""
    from google import genai
    from google.genai import types

    use_enterprise = os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE", "").lower() in ("1", "true", "yes")
    project = (os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip() or None
    location = (os.environ.get("GOOGLE_CLOUD_LOCATION") or "global").strip()
    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip() or None

    # Fast-fail on 429 (1 attempt + 20s timeout) so we immediately fall back instead of hanging 30s!
    retry_cfg = types.HttpRetryOptions(attempts=1)

    # Enterprise Agent Platform (Vertex AI) with Application Default Credentials (ADC)
    if use_enterprise and project:
        return genai.Client(
            enterprise=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(timeout=20_000, retry_options=retry_cfg),
        )

    # Preview models like `gemini-robotics-er-2-preview` require `v1beta` on Gemini Developer API
    api_ver = "v1beta" if "preview" in model_id else None
    http_opts = types.HttpOptions(api_version=api_ver, timeout=20_000, retry_options=retry_cfg)
    if api_key:
        return genai.Client(api_key=api_key, http_options=http_opts)
    return genai.Client(http_options=http_opts)


def run_gemini_interactions_agent(
    prompt: str, png_bytes: bytes, engine: TaskExecutionEngine, model_id: str
) -> str:
    """Official Google Robotics Task Orchestration loop using `client.interactions.create`."""
    img_b64 = base64.b64encode(png_bytes).decode("utf-8")

    system_prompt = (
        "You control a real 3-DOF robotic arm looking at a tabletop camera image. "
        "Coordinates in tool arguments are normalized [y, x] on a 0-1000 scale "
        "(y=0 is top edge, y=1000 is bottom edge; x=0 is left edge, x=1000 is right edge). "
        "Available tabletop objects include a red cube, yellow block, purple cup, and green drop pad. "
        "Locate the objects visually in the image and call the provided robot tools to fulfill the "
        f"user's instruction step-by-step.\n\nUser instruction: {prompt}"
    )

    # Automatic fallback chain if a model hits 429 Resource Exhausted!
    fallback_order = [model_id] + [
        m for m in ("gemini-3.5-flash-lite", "gemini-4-hydrogen", "gemini-4-argon") if m != model_id
    ]
    client = None
    active_model = model_id
    interaction = None
    last_err: Exception | None = None

    for candidate in fallback_order:
        t0 = time.monotonic()
        engine._set_state("PLANNING", f"Calling Gemini ({candidate})...")
        try:
            client = build_genai_client(model_id=candidate)
            interaction = client.interactions.create(
                model=candidate,
                input=[
                    {
                        "type": "user_input",
                        "content": [
                            {"type": "image", "data": img_b64, "mime_type": "image/png"},
                            {"type": "text", "text": system_prompt},
                        ],
                    }
                ],
                tools=ROBOT_TOOLS_SCHEMA,
                generation_config={"thinking_level": "low"},
            )
            active_model = candidate
            engine._set_state("PLANNED", f"{candidate} replied in {time.monotonic() - t0:.2f}s")
            break
        except Exception as exc:
            last_err = exc
            if "429" in str(exc) or "too_many_requests" in str(exc) or "Resource exhausted" in str(exc):
                engine._set_state("RETRY", f"429 rate-limit on {candidate} ({time.monotonic() - t0:.1f}s) -> trying fallback...")
                continue
            raise

    if interaction is None or client is None:
        raise RuntimeError(f"All Gemini models rate-limited (429): {last_err}")

    max_turns = 8
    for _ in range(max_turns):
        if engine._cancel_event.is_set():
            return "Canceled by operator."

        tool_calls = [step for step in interaction.steps if step.type == "function_call"]
        if not tool_calls:
            return interaction.output_text or "Task finished."

        function_results = []
        for step in tool_calls:
            args = dict(step.arguments or {})
            engine._set_state("TOOL_CALL", f"{active_model} called {step.name}({args})")
            result = engine.dispatch_tool(step.name, args)
            function_results.append(
                {
                    "type": "function_result",
                    "name": step.name,
                    "call_id": step.id,
                    "result": [{"type": "text", "text": json.dumps(result)}],
                }
            )

        t_reply = time.monotonic()
        try:
            interaction = client.interactions.create(
                model=active_model,
                previous_interaction_id=interaction.id,
                tools=ROBOT_TOOLS_SCHEMA,
                input=function_results,
            )
            engine._set_state("PLANNED", f"{active_model} follow-up in {time.monotonic() - t_reply:.2f}s")
        except Exception as exc:
            if "429" in str(exc):
                return f"Robot finished tool call cleanly (final summary skipped due to 429 on {active_model})."
            raise

    return interaction.output_text or "Reached max tool steps."


def _find_color_norm_yx(rgb: np.ndarray, target_name: str) -> tuple[int, int] | None:
    """Offline fallback detector (no API key needed) returning normalized [y, x] in 0..1000."""
    h, w, _ = rgb.shape
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    masks = {
        "red": (r > 120) & (r > g * 1.8) & (r > b * 1.8),
        "yellow": (r > 150) & (g > 130) & (b < 80),
        "purple": (r > 100) & (b > 130) & (g < 90),
        "green": (g > 120) & (g > r * 1.5) & (g > b * 1.3),
    }
    mask = masks.get(target_name)
    if mask is None:
        return None
    ys, xs = np.where(mask)
    if len(xs) < 5:
        return None
    return int(round(float(ys.mean()) / h * 1000.0)), int(round(float(xs.mean()) / w * 1000.0))


def run_local_color_planner(
    prompt: str, rgb_small: np.ndarray | None, engine: TaskExecutionEngine
) -> str:
    """Free offline planner so you can test the exact same Task Engine without an API key."""
    if rgb_small is None:
        return "No RGB frame received yet."

    text = prompt.lower()
    if "home" in text:
        res = engine.dispatch_tool("go_home", {})
        return res["message"]

    # Match spoken object names in order of appearance in the prompt
    aliases = [
        ("red", ("red", "cube")),
        ("yellow", ("yellow", "block")),
        ("purple", ("purple", "cup", "mug")),
        ("green", ("green", "pad", "mat", "drop")),
    ]
    found: list[tuple[int, str]] = []
    for color, words in aliases:
        positions = [text.find(w) for w in words if text.find(w) != -1]
        if positions:
            found.append((min(positions), color))
    found.sort()
    ordered = [color for _, color in found]

    if not ordered:
        return "Local Mode understood 'red cube', 'yellow block', 'purple cup', 'green pad', or 'home'. (Set GEMINI_API_KEY for full zero-shot prompts!)"

    if len(ordered) >= 2 and any(w in text for w in ("pick", "lift", "put", "place", "move", "drop", "keep")):
        src, dst = ordered[0], ordered[1]
        p1 = _find_color_norm_yx(rgb_small, src)
        p2 = _find_color_norm_yx(rgb_small, dst)
        if p1 is None or p2 is None:
            return f"Could not see {src} or {dst} in camera."
        res = engine.dispatch_tool(
            "pick_and_place",
            {"pick_y": p1[0], "pick_x": p1[1], "place_y": p2[0], "place_x": p2[1], "label": f"{src} -> {dst}"},
        )
        return res["message"]

    # Single object mentioned -> hover above it
    p = _find_color_norm_yx(rgb_small, ordered[0])
    if p is None:
        return f"Could not see {ordered[0]} in camera."
    res = engine.dispatch_tool("hover_above", {"y": p[0], "x": p[1], "label": ordered[0]})
    return f"Hovered over {ordered[0]} ({res['message']})"
