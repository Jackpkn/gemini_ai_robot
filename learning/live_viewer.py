"""Day 7 (Node 3): Live Browser Viewer & Control Panel (http://localhost:8000).

Run inside Docker alongside `mujoco_ros_sim.py` and `arm_ik_controller.py`:
  python3 /workspace/learning/live_viewer.py

Then open http://localhost:8000 in Chrome/Safari on your Mac!
- Watches `/sim/camera` (BEST_EFFORT CompressedImage) and streams live video to Chrome.
- Watches `/joint_states` and displays live joint angles in real time.
- Lets you click buttons (or run terminal CLI scripts!) to move the real MuJoCo arm
  via ROS 2 `/arm/move` Action and `/gripper/set_closed` Service.
"""

import asyncio
from contextlib import asynccontextmanager
import os
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from geometry_msgs.msg import PointStamped
import numpy as np
from pydantic import BaseModel
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState
from std_srvs.srv import SetBool, Trigger
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener
import uvicorn

from robot_agent import TaskExecutionEngine
from robot_interfaces.action import MoveArm


class ViewerBridgeNode(Node):
    """ROS 2 node bridged into FastAPI: subscribes to camera/joints and sends commands."""

    def __init__(self) -> None:
        super().__init__("live_viewer")
        self.lock = threading.Lock()
        self.latest_frame: bytes = b""
        self.latest_fmt: str = "jpeg"
        self.latest_rgb_small: np.ndarray | None = None
        self.latest_depth_m: np.ndarray | None = None
        self.camera_k: tuple[float, float, float, float] | None = None
        self.camera_frame: str = "overview_camera"
        self.frames_received: int = 0
        self.joints: dict[str, float] = {}
        self.detected_cube: dict[str, float] | None = None
        self.status_text: str = "Ready — type a prompt or click a button!"
        self.active_goal = None

        self.cb_group = ReentrantCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.cam_sub = self.create_subscription(
            CompressedImage, "/sim/camera", self.on_camera, sensor_qos, callback_group=self.cb_group
        )
        self.rgb_sub = self.create_subscription(
            Image, "/sim/rgb", self.on_rgb, sensor_qos, callback_group=self.cb_group
        )
        self.depth_sub = self.create_subscription(
            Image, "/sim/depth", self.on_depth, sensor_qos, callback_group=self.cb_group
        )
        self.info_sub = self.create_subscription(
            CameraInfo, "/sim/camera_info", self.on_camera_info, latched_qos, callback_group=self.cb_group
        )
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self.on_joints, sensor_qos, callback_group=self.cb_group
        )
        self.cube_sub = self.create_subscription(
            PointStamped, "/detected_cube", self.on_detected_cube, 10, callback_group=self.cb_group
        )
        self.arm_client = ActionClient(self, MoveArm, "/arm/move", callback_group=self.cb_group)
        self.gripper_client = self.create_client(
            SetBool, "/gripper/set_closed", callback_group=self.cb_group
        )
        self.rand_cube_client = self.create_client(
            Trigger, "/sim/randomize_cube", callback_group=self.cb_group
        )
        self.task_engine = TaskExecutionEngine(
            move_arm_sync=self.move_arm_sync,
            set_gripper_sync=self.set_gripper_sync,
            norm_point_to_3d=self.norm_point_to_3d,
            get_camera_snapshot=self.get_camera_snapshot,
            get_gripper_fingers=self.get_gripper_fingers,
        )

    def get_gripper_fingers(self) -> tuple[float, float]:
        with self.lock:
            return (
                float(self.joints.get("left_finger_slide", 0.0)),
                float(self.joints.get("right_finger_slide", 0.0)),
            )

    def on_camera(self, msg: CompressedImage) -> None:
        with self.lock:
            self.latest_frame = bytes(msg.data)
            self.latest_fmt = msg.format or "jpeg"
            self.frames_received += 1

    def on_rgb(self, msg: Image) -> None:
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
        with self.lock:
            self.latest_rgb_small = arr

    def on_depth(self, msg: Image) -> None:
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
        with self.lock:
            self.latest_depth_m = arr
            self.camera_frame = msg.header.frame_id or "overview_camera"

    def on_camera_info(self, msg: CameraInfo) -> None:
        with self.lock:
            self.camera_k = (float(msg.k[0]), float(msg.k[4]), float(msg.k[2]), float(msg.k[5]))

    def get_camera_snapshot(self) -> tuple[bytes, np.ndarray | None]:
        with self.lock:
            rgb_copy = self.latest_rgb_small.copy() if self.latest_rgb_small is not None else None
            return self.latest_frame, rgb_copy

    def norm_point_to_3d(self, y_norm: int, x_norm: int) -> tuple[float, float, float] | None:
        """Convert normalized [y, x] (0..1000) from Gemini/Local Mode -> 3D world (x, y, z) meters."""
        with self.lock:
            depth = self.latest_depth_m
            cam_k = self.camera_k
            cam_frame = self.camera_frame
        if depth is None or cam_k is None:
            return None
        h, w = depth.shape
        v = int(np.clip(round(y_norm / 1000.0 * (h - 1)), 0, h - 1))
        u = int(np.clip(round(x_norm / 1000.0 * (w - 1)), 0, w - 1))
        # Take median of a tiny 3x3 patch around (u, v) for noise immunity
        patch = depth[max(0, v - 1) : min(h, v + 2), max(0, u - 1) : min(w, u + 2)]
        d = float(np.median(patch))
        if d <= 0.05 or d > 5.0:
            return None

        # Depth camera sees the FRONT surface of a ~4.5cm object; step +2.2cm along the camera ray
        # to project into the object's physical 3D center (standard RGB-D grasp center correction!).
        d_center = d + 0.022
        fx, fy, cx, cy = cam_k
        pt_cam = PointStamped()
        pt_cam.header.frame_id = cam_frame
        pt_cam.point.x = (u - cx) / fx * d_center
        pt_cam.point.y = -(v - cy) / fy * d_center
        pt_cam.point.z = -d_center
        try:
            tf = self.tf_buffer.lookup_transform("world", cam_frame, Time(), timeout=Duration(seconds=0.2))
            pt_world = do_transform_point(pt_cam, tf)
            return float(pt_world.point.x), float(pt_world.point.y), float(pt_world.point.z)
        except TransformException:
            return None

    def move_arm_sync(
        self, x: float, y: float, z: float, timeout_s: float, cancel_event: threading.Event
    ) -> tuple[bool, str]:
        """Blocking wrapper around ROS 2 `/arm/move` Action with live cancel + timeout support."""
        if not self.arm_client.wait_for_server(timeout_sec=1.5):
            return False, "/arm/move server not running"

        goal = MoveArm.Goal(target_x=float(x), target_y=float(y), target_z=float(z))
        goal_done = threading.Event()
        holder: dict = {"handle": None, "result": None}

        def on_goal(fut) -> None:
            holder["handle"] = fut.result()
            goal_done.set()

        self.arm_client.send_goal_async(goal).add_done_callback(on_goal)
        if not goal_done.wait(timeout=2.0):
            return False, "Timed out waiting for goal acceptance"
        handle = holder["handle"]
        if handle is None or not handle.accepted:
            return False, f"Goal ({x:.2f}, {y:.2f}, {z:.2f}) rejected by IK controller"

        with self.lock:
            self.active_goal = handle

        res_done = threading.Event()
        handle.get_result_async().add_done_callback(lambda rf: (holder.__setitem__("result", rf.result()), res_done.set()))

        deadline = time.monotonic() + timeout_s
        while not res_done.is_set():
            if cancel_event.is_set():
                handle.cancel_goal_async()
                res_done.wait(timeout=1.0)
                return False, "Canceled mid-motion"
            if time.monotonic() > deadline:
                handle.cancel_goal_async()
                res_done.wait(timeout=1.0)
                return False, f"Motion timed out after {timeout_s:.1f}s"
            res_done.wait(timeout=0.05)

        wrapped = holder["result"]
        if wrapped is None:
            return False, "No result received"
        return bool(wrapped.result.success), str(wrapped.result.message)

    def set_gripper_sync(self, closed: bool) -> tuple[bool, str]:
        if not self.gripper_client.wait_for_service(timeout_sec=1.5):
            return False, "/gripper/set_closed service not running"
        done = threading.Event()
        holder: dict = {"resp": None}
        fut = self.gripper_client.call_async(SetBool.Request(data=closed))
        fut.add_done_callback(lambda f: (holder.__setitem__("resp", f.result()), done.set()))
        if not done.wait(timeout=2.0) or holder["resp"] is None:
            return False, "Gripper service timed out"
        return bool(holder["resp"].success), str(holder["resp"].message)

    def on_joints(self, msg: JointState) -> None:
        with self.lock:
            self.joints = {name: round(float(val), 3) for name, val in zip(msg.name, msg.position)}

    def on_detected_cube(self, msg: PointStamped) -> None:
        with self.lock:
            self.detected_cube = {
                "x": round(float(msg.point.x), 3),
                "y": round(float(msg.point.y), 3),
                "z": round(float(msg.point.z), 3),
            }

    def set_status(self, text: str) -> None:
        with self.lock:
            self.status_text = text

    def send_move(self, x: float, y: float, z: float) -> dict:
        if not self.arm_client.wait_for_server(timeout_sec=1.0):
            return {"ok": False, "message": "/arm/move server not running"}

        goal = MoveArm.Goal(target_x=x, target_y=y, target_z=z)
        self.set_status(f"Moving to ({x:.2f}, {y:.2f}, {z:.2f})...")

        def on_feedback(fb_msg) -> None:
            fb = fb_msg.feedback
            self.set_status(f"Moving... {fb.progress_pct:.0f}% ({fb.distance_remaining_m:.3f}m left)")

        send_fut = self.arm_client.send_goal_async(goal, feedback_callback=on_feedback)

        def on_goal_resp(fut) -> None:
            handle = fut.result()
            if handle is None or not handle.accepted:
                self.set_status("Goal REJECTED (out of reach or sim not ready)")
                return
            with self.lock:
                self.active_goal = handle
            res_fut = handle.get_result_async()
            res_fut.add_done_callback(
                lambda rf: self.set_status(
                    f"Done: {rf.result().result.message} (err={rf.result().result.final_distance_m:.3f}m)"
                )
            )

        send_fut.add_done_callback(on_goal_resp)
        return {"ok": True, "message": f"Sent goal ({x}, {y}, {z})"}

    def cancel_move(self) -> dict:
        task_cancel = self.task_engine.request_cancel()
        with self.lock:
            handle = self.active_goal
        if handle is not None:
            handle.cancel_goal_async()
            self.set_status("Cancel requested!")
            return {"ok": True, "message": "Cancel sent"}
        return task_cancel

    def set_gripper(self, closed: bool) -> dict:
        if not self.gripper_client.wait_for_service(timeout_sec=1.0):
            return {"ok": False, "message": "/gripper/set_closed service not running"}
        req = SetBool.Request(data=closed)
        fut = self.gripper_client.call_async(req)
        fut.add_done_callback(lambda f: self.set_status(f"Gripper: {f.result().message}"))
        return {"ok": True, "message": "Closed" if closed else "Opened"}

    def randomize_cube(self) -> dict:
        if not self.rand_cube_client.wait_for_service(timeout_sec=1.0):
            return {"ok": False, "message": "/sim/randomize_cube service not running"}
        fut = self.rand_cube_client.call_async(Trigger.Request())
        fut.add_done_callback(lambda f: self.set_status(f"Table: {f.result().message}"))
        return {"ok": True, "message": "Randomizing cube"}

    def move_to_detected_cube(self) -> dict:
        with self.lock:
            cube = dict(self.detected_cube) if self.detected_cube else None
        if cube is None:
            return {"ok": False, "message": "Camera has not detected the cube yet"}
        # Hover 8 cm above whatever (x, y, z) the camera detector measured!
        return self.send_move(cube["x"], cube["y"], max(0.11, cube["z"] + 0.08))


class MoveBody(BaseModel):
    x: float
    y: float
    z: float


class GripperBody(BaseModel):
    closed: bool


class TaskBody(BaseModel):
    prompt: str
    mode: str = "auto"
    model: str | None = None
    fault: str | None = None


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Teleop-Lab Live MuJoCo + ROS 2 + Gemini Agent</title>
  <style>
    :root { color-scheme: dark; }
    body { font-family: system-ui, sans-serif; background: #111318; color: #eef2f7; margin: 0; padding: 1.25rem; }
    .wrap { max-width: 1020px; margin: 0 auto; display: grid; grid-template-columns: 1.25fr 1fr; gap: 1.25rem; }
    @media (max-width: 820px) { .wrap { grid-template-columns: 1fr; } }
    .card { background: #1b1f27; border: 1px solid #2e3440; border-radius: 12px; padding: 1rem; }
    h1 { font-size: 1.15rem; margin: 0 0 0.75rem; }
    h2 { font-size: 0.85rem; margin: 0.9rem 0 0.5rem; color: #9fb0c7; text-transform: uppercase; letter-spacing: 0.04em; }
    h2:first-child { margin-top: 0; }
    img { width: 100%; border-radius: 8px; background: #000; display: block; aspect-ratio: 4 / 3; object-fit: contain; }
    .btns { display: grid; grid-template-columns: 1fr 1fr; gap: 0.45rem; margin-bottom: 0.75rem; }
    button, input, select { padding: 0.55rem 0.7rem; border-radius: 8px; border: 1px solid #3a4458; background: #262e3d; color: #eef2f7; font-size: 0.88rem; }
    button { cursor: pointer; }
    button:hover { background: #313d52; }
    button.primary { background: #1e3a5f; border-color: #60a5fa; }
    button.danger { background: #4c1d24; border-color: #ef4444; }
    button.warn { background: #422c14; border-color: #f59e0b; }
    .row { display: flex; gap: 0.45rem; margin-bottom: 0.5rem; }
    .row input { flex: 1; background: #111318; }
    .chips { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 0.65rem; }
    .chip { font-size: 0.76rem; padding: 0.3rem 0.55rem; background: #151922; border: 1px solid #2e3440; border-radius: 999px; }
    .badge { display: inline-block; padding: 0.2rem 0.55rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; background: #262e3d; color: #93c5fd; }
    .badge.busy { background: #1e3a5f; color: #60a5fa; }
    .badge.err { background: #4c1d24; color: #fca5a5; }
    pre { background: #111318; padding: 0.65rem; border-radius: 8px; font-size: 0.8rem; overflow-x: auto; margin: 0; white-space: pre-wrap; }
    .status { padding: 0.55rem 0.7rem; border-radius: 8px; background: #111318; border-left: 3px solid #3b82f6; margin-bottom: 0.75rem; font-size: 0.86rem; }
  </style>
</head>
<body>
  <main class="wrap">
    <section class="card">
      <h1>Live MuJoCo Camera (<code>/sim/camera</code>)</h1>
      <img id="cam" alt="Live MuJoCo Robot Arm Camera">

      <h2>Zero-Shot AI Agent (Gemini Interactions API + Safety Engine)</h2>
      <div class="row">
        <input id="prompt" type="text" value="Pick up the yellow block and put it on the green pad" placeholder="Type a task in English...">
      </div>
      <div class="row">
        <select id="modeSelect" style="flex:1;">
          <option value="auto">Auto (Gemini 4 Hydrogen via ADC/Key, else Local)</option>
          <option value="gemini:gemini-4-hydrogen">Gemini 4 Hydrogen (fast frontier)</option>
          <option value="gemini:gemini-4-argon">Gemini 4 Argon (stronger reasoning)</option>
          <option value="gemini:gemini-4-carbon">Gemini 4 Carbon (flagship)</option>
          <option value="gemini:gemini-3.5-flash-lite">Gemini 3.5 Flash-Lite (ultra-fast)</option>
          <option value="local">Free Local Mode (offline)</option>
        </select>
        <button class="primary" onclick="runTask()">Run Task</button>
        <button class="warn" onclick="runTask('timeout')">Test Fault</button>
      </div>
      <div class="chips">
        <button class="chip" onclick="setPrompt('Pick up the yellow block and put it on the green pad')">Yellow block → Green pad</button>
        <button class="chip" onclick="setPrompt('Pick up the purple cup and put it on the green pad')">Purple cup → Green pad</button>
        <button class="chip" onclick="setPrompt('Hover over the red cube')">Hover Red cube</button>
        <button class="chip" onclick="setPrompt('Go home')">Go Home</button>
      </div>
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.4rem;">
        <span style="font-size:0.82rem;color:#9fb0c7;">State Machine: <span id="taskState" class="badge">IDLE</span></span>
        <span id="keyBadge" style="font-size:0.78rem;color:#9fb0c7;"></span>
      </div>
      <pre id="trace">No task executed yet.</pre>
    </section>

    <section class="card">
      <h2>Manual ROS 2 Controls</h2>
      <div class="status" id="status" role="status" aria-live="polite">Connecting...</div>
      <div class="btns">
        <button class="primary" onclick="autoFind()">Auto-Find Red Cube</button>
        <button onclick="randomizeCube()">Randomize Red Cube</button>
        <button onclick="move(0.00, 0.35, 0.18)">Over Green Pad</button>
        <button onclick="move(0.25, 0.00, 0.36)">Home Pose</button>
        <button onclick="gripper(true)">Close Gripper</button>
        <button onclick="gripper(false)">Open Gripper</button>
        <button class="danger" style="grid-column: span 2;" onclick="cancelMove()">EMERGENCY CANCEL / RECOVER</button>
      </div>
      <h2>Camera Vision (/detected_cube)</h2>
      <pre id="vision">waiting for camera detector...</pre>
      <h2>Live /joint_states (rad / m)</h2>
      <pre id="joints">waiting...</pre>
    </section>
  </main>
  <script>
    function setPrompt(txt) { document.getElementById('prompt').value = txt; }
    async function runTask(fault = null) {
      const prompt = document.getElementById('prompt').value.trim();
      if (!prompt) return;
      const sel = document.getElementById('modeSelect').value;
      let mode = sel, model = null;
      if (sel.startsWith('gemini:')) { mode = 'gemini'; model = sel.slice(7); }
      const res = await fetch('/api/task', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ prompt, mode, model, fault })
      });
      const out = await res.json();
      if (!out.ok) document.getElementById('status').textContent = out.message;
    }
    async function move(x, y, z) {
      await fetch('/api/move', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({x, y, z}) });
    }
    async function autoFind() { await fetch('/api/auto_find', { method: 'POST' }); }
    async function randomizeCube() { await fetch('/api/randomize_cube', { method: 'POST' }); }
    async function cancelMove() { await fetch('/api/cancel', { method: 'POST' }); }
    async function gripper(closed) {
      await fetch('/api/gripper', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({closed}) });
    }
    async function poll() {
      try {
        const res = await fetch('/api/state');
        const data = await res.json();
        document.getElementById('status').textContent = data.status + ` (frames: ${data.frames_received})`;
        document.getElementById('vision').textContent = data.detected_cube
          ? JSON.stringify(data.detected_cube)
          : 'no red cube detected yet';
        document.getElementById('joints').textContent = JSON.stringify(data.joints, null, 2);
        const badge = document.getElementById('taskState');
        badge.textContent = data.task.state;
        badge.className = 'badge' + (data.task.busy ? ' busy' : '') + (['FAILED','CANCELED','RECOVERING','SAFETY_BLOCK'].includes(data.task.state) ? ' err' : '');
        document.getElementById('keyBadge').textContent = data.gemini_configured ? 'Gemini Enterprise ADC / Key: Ready' : 'Mode: Free Local (offline)';
        document.getElementById('trace').textContent = data.task.trace.length ? data.task.trace.join('\\n') : data.task.summary;
      } catch (e) {}
    }
    const cam = document.getElementById('cam');
    async function refreshFrame() {
      try {
        const res = await fetch('/frame', { cache: 'no-store' });
        if (res.ok) {
          const blob = await res.blob();
          const url = URL.createObjectURL(blob);
          const oldUrl = cam.dataset.url;
          cam.src = url;
          cam.dataset.url = url;
          if (oldUrl) URL.revokeObjectURL(oldUrl);
        }
      } catch (e) {}
      setTimeout(refreshFrame, 260);
    }
    setInterval(poll, 350);
    poll();
    refreshFrame();
  </script>
</body>
</html>
"""


def build_app(bridge: ViewerBridgeNode) -> FastAPI:
    shutdown_event = asyncio.Event()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        bridge.get_logger().info("Live Viewer ready -> open http://localhost:8000 in your browser!")
        yield
        shutdown_event.set()

    app = FastAPI(lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML_PAGE

    @app.get("/api/state")
    def state() -> dict:
        with bridge.lock:
            return {
                "frames_received": bridge.frames_received,
                "joints": bridge.joints,
                "detected_cube": bridge.detected_cube,
                "status": bridge.status_text,
                "task": bridge.task_engine.snapshot(),
                "gemini_configured": bool(
                    os.environ.get("GEMINI_API_KEY")
                    or os.environ.get("GOOGLE_API_KEY")
                    or os.environ.get("GOOGLE_CLOUD_PROJECT")
                ),
            }

    @app.get("/frame")
    def single_frame() -> Response:
        with bridge.lock:
            data = bridge.latest_frame
            fmt = bridge.latest_fmt
        if not data:
            return Response(status_code=503, content=b"No camera frame yet")
        return Response(content=data, media_type=f"image/{fmt}")

    @app.get("/stream")
    async def mjpeg_stream(request: Request) -> StreamingResponse:
        async def generate():
            last_sent = -1
            while rclpy.ok() and not shutdown_event.is_set():
                if await request.is_disconnected():
                    break
                with bridge.lock:
                    frame_id = bridge.frames_received
                    data = bridge.latest_frame
                    fmt = bridge.latest_fmt
                if data and frame_id != last_sent:
                    last_sent = frame_id
                    yield (
                        b"--frame\r\n"
                        + f"Content-Type: image/{fmt}\r\n".encode()
                        + f"Content-Length: {len(data)}\r\n\r\n".encode()
                        + data
                        + b"\r\n"
                    )
                await asyncio.sleep(0.05)

        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/move")
    def api_move(body: MoveBody) -> dict:
        return bridge.send_move(body.x, body.y, body.z)

    @app.post("/api/cancel")
    def api_cancel() -> dict:
        return bridge.cancel_move()

    @app.post("/api/gripper")
    def api_gripper(body: GripperBody) -> dict:
        return bridge.set_gripper(body.closed)

    @app.post("/api/randomize_cube")
    def api_randomize_cube() -> dict:
        return bridge.randomize_cube()

    @app.post("/api/auto_find")
    def api_auto_find() -> dict:
        return bridge.move_to_detected_cube()

    @app.post("/api/task")
    def api_task(body: TaskBody) -> dict:
        return bridge.task_engine.start_task_async(
            prompt=body.prompt, mode=body.mode, model=body.model, fault=body.fault
        )

    return app


def main() -> None:
    rclpy.init()
    bridge = ViewerBridgeNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(bridge)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    app = build_app(bridge)
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    finally:
        executor.shutdown()
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
