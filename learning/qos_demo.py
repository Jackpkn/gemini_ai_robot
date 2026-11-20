"""Day 4 (Part 2): ROS 2 QoS (Quality of Service) — Compatibility & Late Joiners.

Run inside Docker:
  python3 /workspace/learning/qos_demo.py --mode mismatch     # BEST_EFFORT pub vs RELIABLE sub -> 0 msgs!
  python3 /workspace/learning/qos_demo.py --mode compatible   # RELIABLE pub vs BEST_EFFORT sub -> works!
  python3 /workspace/learning/qos_demo.py --mode late-joiner  # VOLATILE (misses) vs TRANSIENT_LOCAL (gets cached!)

Why Mind Robotics cares about QoS:
- High-rate cameras / joint sensors use BEST_EFFORT (drop stale frames, never stall).
- Safety commands & task state use RELIABLE.
- Late-joining Operator UIs use TRANSIENT_LOCAL so they immediately see the robot's last config/state!
"""

import argparse
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


def make_qos(reliability: ReliabilityPolicy, durability: DurabilityPolicy = DurabilityPolicy.VOLATILE) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=reliability,
        durability=durability,
    )


def run_reliability_test(node: Node, executor: SingleThreadedExecutor, pub_reliable: bool, sub_reliable: bool) -> None:
    pub_qos = make_qos(ReliabilityPolicy.RELIABLE if pub_reliable else ReliabilityPolicy.BEST_EFFORT)
    sub_qos = make_qos(ReliabilityPolicy.RELIABLE if sub_reliable else ReliabilityPolicy.BEST_EFFORT)

    received: list[str] = []
    pub = node.create_publisher(String, "/qos/sensor", pub_qos)
    sub = node.create_subscription(String, "/qos/sensor", lambda m: received.append(m.data), sub_qos)
    _ = sub  # keep reference alive

    # Spin briefly so DDS discovery runs, then publish 5 messages.
    for i in range(5):
        executor.spin_once(timeout_sec=0.1)
        pub.publish(String(data=f"frame-{i}"))
        time.sleep(0.05)
    for _ in range(5):
        executor.spin_once(timeout_sec=0.05)

    pub_label = "RELIABLE" if pub_reliable else "BEST_EFFORT"
    sub_label = "RELIABLE" if sub_reliable else "BEST_EFFORT"
    node.get_logger().info(
        f"[QoS Reliability] Pub={pub_label}, Sub={sub_label} -> "
        f"matched_subs={pub.get_subscription_count()}, received={len(received)}/5"
    )


def run_late_joiner_test(node: Node, executor: SingleThreadedExecutor) -> None:
    volatile_qos = make_qos(ReliabilityPolicy.RELIABLE, DurabilityPolicy.VOLATILE)
    latched_qos = make_qos(ReliabilityPolicy.RELIABLE, DurabilityPolicy.TRANSIENT_LOCAL)

    # 1. Publishers send robot config BEFORE the Operator UI subscribes!
    vol_pub = node.create_publisher(String, "/config/volatile", volatile_qos)
    lat_pub = node.create_publisher(String, "/config/latched", latched_qos)

    vol_pub.publish(String(data="MAX_SPEED=0.5 m/s"))
    lat_pub.publish(String(data="MAX_SPEED=0.5 m/s"))
    executor.spin_once(timeout_sec=0.2)
    node.get_logger().info("Published robot config BEFORE Operator UI joined...")

    # 2. Operator UI joins 0.3s LATER!
    time.sleep(0.3)
    got_volatile: list[str] = []
    got_latched: list[str] = []
    s1 = node.create_subscription(String, "/config/volatile", lambda m: got_volatile.append(m.data), volatile_qos)
    s2 = node.create_subscription(String, "/config/latched", lambda m: got_latched.append(m.data), latched_qos)
    _ = (s1, s2)

    for _ in range(10):
        executor.spin_once(timeout_sec=0.05)

    node.get_logger().info(f"[Late Joiner] VOLATILE received:        {got_volatile}")
    node.get_logger().info(f"[Late Joiner] TRANSIENT_LOCAL received: {got_latched}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mismatch", "compatible", "late-joiner"), required=True)
    args = parser.parse_args()

    rclpy.init()
    node = Node("qos_demo")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        if args.mode == "mismatch":
            run_reliability_test(node, executor, pub_reliable=False, sub_reliable=True)
        elif args.mode == "compatible":
            run_reliability_test(node, executor, pub_reliable=True, sub_reliable=False)
        else:
            run_late_joiner_test(node, executor)
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
