#!/usr/bin/env bash
# Exit immediately if a command fails. Note we intentionally do NOT enable `set -u`
# because ROS 2's setup.bash scripts sometimes check optional unset environment variables.
set -e

# Activate the base ROS 2 Jazzy installation (adds `ros2`, `rclpy`, etc. to PATH/PYTHONPATH).
source /opt/ros/jazzy/setup.bash

# If our custom workspace (`ros2_ws`) has already been compiled with `colcon build`,
# source its overlay too so our custom packages (like `teleop_basics`) are immediately runnable.
if [ -f /workspace/ros2_ws/install/setup.bash ]; then
  source /workspace/ros2_ws/install/setup.bash
fi

# Replace this shell process with whatever command the user/Compose passed (for example:
# `ros2 launch ...`, `pytest ...`, or `bash`). Keeping PID 1 clean ensures Ctrl+C works properly.
exec "$@"
