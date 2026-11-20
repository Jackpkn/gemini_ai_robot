"""Day 6: Your First MuJoCo 3D Robot Arm (runs directly on Mac OR headless inside Docker).

Watch it in 3D on your Mac:
  .venv/bin/mjpython learning/mujoco_arm_demo.py --viewer

Run headless (prints joint angles + 3D end-effector position):
  .venv/bin/python learning/mujoco_arm_demo.py

Verify offscreen camera rendering inside Docker:
  docker compose -f docker/docker-compose.yml run --rm teleop \
    python3 /workspace/learning/mujoco_arm_demo.py --render-check
"""

import argparse
from pathlib import Path
import time

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).with_name("simple_arm.xml")

# Sequence of (label, [base_yaw, shoulder_pitch, elbow_pitch, wrist_pitch, left_finger, right_finger]) poses
WAYPOINTS = [
    ("Home (neutral)",      [0.0,  0.0,  0.0,  0.0,   0.0,   0.0]),
    ("Reach over red cube", [0.0,  0.55, 0.55, -1.10, 0.0,   0.0]),
    ("Close gripper",       [0.0,  0.55, 0.55, -1.10, 0.036, 0.036]),
    ("Swing to green pad",  [1.57, 0.35, 0.45, -0.80, 0.036, 0.036]),
    ("Open gripper",        [1.57, 0.35, 0.45, -0.80, 0.0,   0.0]),
]


def print_state(model: mujoco.MjModel, data: mujoco.MjData, label: str) -> None:
    ee_pos = data.site("ee_tip").xpos
    cube_pos = data.body("target_cube").xpos
    # Joint indices 0..2 of our robot are right after the cube's 7 freejoint values (`qpos[7:10]`),
    # so looking them up by name is always cleaner and bug-free!
    arm_q = [float(data.joint(name).qpos[0]) for name in ("base_yaw", "shoulder_pitch", "elbow_pitch")]
    print(
        f"[{label:<20}] joints={[round(v, 2) for v in arm_q]} rad | "
        f"ee_tip=({ee_pos[0]:+.2f}, {ee_pos[1]:+.2f}, {ee_pos[2]:+.2f})m | "
        f"cube=({cube_pos[0]:+.2f}, {cube_pos[1]:+.2f}, {cube_pos[2]:+.2f})m"
    )


def run_headless(model: mujoco.MjModel, data: mujoco.MjData, render_check: bool) -> None:
    for label, ctrl in WAYPOINTS:
        data.ctrl[:] = ctrl
        # Step 1.0 second of simulated physics (500 steps * 0.002s = 1.0s)
        for _ in range(500):
            mujoco.mj_step(model, data)
        print_state(model, data, label)

    if render_check:
        renderer = mujoco.Renderer(model, height=240, width=320)
        renderer.update_scene(data, camera="overview")
        rgb = renderer.render()
        renderer.close()
        print(
            f"[Offscreen Render OK] shape={rgb.shape}, dtype={rgb.dtype}, "
            f"mean_pixel={float(np.mean(rgb)):.1f}"
        )


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    import mujoco.viewer

    print("Opening 3D MuJoCo viewer! Drag with mouse to orbit; close the window to quit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        idx = 0
        while viewer.is_running():
            label, ctrl = WAYPOINTS[idx % len(WAYPOINTS)]
            data.ctrl[:] = ctrl
            for _ in range(500):
                if not viewer.is_running():
                    break
                step_start = time.monotonic()
                mujoco.mj_step(model, data)
                viewer.sync()
                remaining = model.opt.timestep - (time.monotonic() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
            print_state(model, data, label)
            idx += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="Open interactive 3D window (use mjpython on Mac)")
    parser.add_argument("--render-check", action="store_true", help="Test offscreen RGB camera rendering")
    args = parser.parse_args()

    # Load static XML blueprint (`MjModel`) and live simulation state (`MjData`).
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    if args.viewer:
        run_viewer(model, data)
    else:
        run_headless(model, data, render_check=args.render_check)


if __name__ == "__main__":
    main()
