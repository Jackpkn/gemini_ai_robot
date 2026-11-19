# Teleop-Lab

A hands-on robotics middleware and kinematics learning repo built step-by-step for Python engineers.

## Project Structure

- `docker/` — Linux Ubuntu 24.04 + ROS 2 Jazzy (`arm64`) container setup (`Dockerfile`, `docker-compose.yml`, `entrypoint.sh`).
- `ros2_ws/src/teleop_basics/` — First ROS 2 Python Pub/Sub package:
  - `telemetry_payload.py` — Pure Python JSON heartbeat + monotonic latency + per-node drop tracker.
  - `heartbeat_publisher.py` — ROS 2 node publishing heartbeats to `/teleop/heartbeat`.
  - `heartbeat_subscriber.py` — ROS 2 node receiving heartbeats and measuring latency in ms.
  - `launch/phase1_pubsub.launch.py` — Starts both nodes together.
- `learning/` — Small, focused robotics math & concept scripts (`fk.py`, etc.).

## Quick Commands

Run your kinematics script on Mac:
```bash
.venv/bin/python learning/fk.py
```

Build and run the ROS 2 heartbeat nodes inside Docker:
```bash
docker compose -f docker/docker-compose.yml run --rm teleop bash -lc \
  "cd /workspace/ros2_ws && colcon build --symlink-install && source install/setup.bash && ros2 launch teleop_basics phase1_pubsub.launch.py"
```
