"""ROS 2 Launch file for Phase 1: starts both publisher and subscriber together.

Why use a Launch file?
----------------------
In robotics, a system consists of many separate nodes. Instead of opening 5 terminals
and running `ros2 run` manually in each, a Python Launch file starts a configured
group of nodes in one command (`ros2 launch teleop_basics phase1_pubsub.launch.py`).
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Return the launch plan ROS 2 executes when `ros2 launch` is called."""
    publisher_node = Node(
        package="teleop_basics",
        executable="heartbeat_publisher",
        name="heartbeat_publisher",
        parameters=[
            {
                "node_id": "arm_sim",
                "rate_hz": 2.0,
            }
        ],
        output="screen",
    )

    subscriber_node = Node(
        package="teleop_basics",
        executable="heartbeat_subscriber",
        name="heartbeat_subscriber",
        output="screen",
    )

    return LaunchDescription([publisher_node, subscriber_node])
