"""Day 5 (Part 2): ROS 2 TF2 — Automatic 3D Coordinate Frame Tree across Nodes.

Run inside Docker:
  python3 /workspace/learning/tf2_demo.py

What this demonstrates:
1. StaticTransformBroadcaster (/tf_static): publishes fixed frames (world -> target_cube)
   once using TRANSIENT_LOCAL QoS so any late-joining node still sees them.
2. TransformBroadcaster (/tf): publishes moving frames (world -> hand) continuously.
3. Buffer + TransformListener: asks ONE simple question:
     "Where is `target_cube` relative to `hand` right now?"
   TF2 automatically walks the tree (hand -> world -> target_cube), handles 3D
   rotations + matrix inverses, and hands back (x, y, z) in the hand's own eyes!
"""

import math

from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformBroadcaster, TransformListener


def make_tf(
    parent: str,
    child: str,
    x: float,
    y: float,
    z: float,
    yaw_deg: float,
    stamp,
) -> TransformStamped:
    """Pack (x, y, z) position + Z-axis yaw rotation (as a 3D quaternion) into TransformStamped."""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.translation.z = z

    # ROS 2 represents 3D rotation as a quaternion (x, y, z, w) to avoid gimbal lock.
    # For a pure Z-axis yaw rotation by angle `a`: z = sin(a/2), w = cos(a/2).
    half = math.radians(yaw_deg) / 2.0
    t.transform.rotation.z = math.sin(half)
    t.transform.rotation.w = math.cos(half)
    return t


def main() -> None:
    rclpy.init()
    node = Node("tf2_demo")
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    # Broadcasters (send frames onto /tf_static and /tf)
    static_broadcaster = StaticTransformBroadcaster(node)
    dynamic_broadcaster = TransformBroadcaster(node)

    # Listener (automatically collects all /tf and /tf_static into a searchable 3D tree)
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)
    _ = tf_listener

    # 1. Cube sits fixed on the table at (0.50, 0.80, 0.20) in `world` frame.
    static_broadcaster.sendTransform(
        make_tf("world", "target_cube", x=0.50, y=0.80, z=0.20, yaw_deg=0.0, stamp=node.get_clock().now().to_msg())
    )

    try:
        # 2. Move the hand from y=0.50 toward the cube at y=0.80 (pointing +Y, i.e. yaw=90°).
        for step, hand_y in enumerate([0.50, 0.60, 0.70, 0.80], start=1):
            dynamic_broadcaster.sendTransform(
                make_tf("world", "hand", x=0.50, y=hand_y, z=0.20, yaw_deg=90.0, stamp=node.get_clock().now().to_msg())
            )
            # Let TF2 receive the freshly broadcast transform.
            for _ in range(5):
                executor.spin_once(timeout_sec=0.05)

            # 3. Ask TF2: "Where is `target_cube` seen FROM `hand`'s eyes?"
            #    Notice: target_frame="hand", source_frame="target_cube"
            rel = tf_buffer.lookup_transform("hand", "target_cube", Time())
            tx = rel.transform.translation.x  # straight ahead of the hand!
            ty = rel.transform.translation.y  # left/right of the hand
            tz = rel.transform.translation.z  # up/down from the hand

            node.get_logger().info(
                f"[Step {step}] hand_world=(0.50, {hand_y:.2f}, 0.20) | "
                f"TF2 cube-in-hand -> forward={tx:.2f}m, left={ty:.2f}m, up={tz:.2f}m"
            )
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
