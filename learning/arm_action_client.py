"""Day 3 (Part 2): ROS 2 Action Client — sends a MoveArm goal, prints feedback, optionally cancels.

Usage inside Docker:
  # Normal completed move:
  python3 /workspace/learning/arm_action_client.py 0.4 0.2 0.3

  # Send a goal and cancel it after 0.9 seconds mid-motion:
  python3 /workspace/learning/arm_action_client.py 0.4 0.2 0.3 --cancel-after 0.9
"""

import argparse
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from robot_interfaces.action import MoveArm


class ArmActionClient(Node):
    def __init__(self) -> None:
        super().__init__("arm_action_client")
        self.client = ActionClient(self, MoveArm, "/arm/move")

    def on_feedback(self, feedback_msg) -> None:
        fb = feedback_msg.feedback
        self.get_logger().info(
            f"[Feedback] {fb.progress_pct:5.1f}% complete | {fb.distance_remaining_m:.3f}m left"
        )

    def send_move(self, x: float, y: float, z: float, cancel_after: float | None = None) -> None:
        # 1. Wait for the action server to appear on DDS.
        if not self.client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("Action server /arm/move not available!")

        # 2. Send the Goal + register our feedback callback.
        goal = MoveArm.Goal(target_x=x, target_y=y, target_z=z)
        send_future = self.client.send_goal_async(goal, feedback_callback=self.on_feedback)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Goal was REJECTED by /arm/move!")
            return

        self.get_logger().info("Goal ACCEPTED! Waiting for motion...")
        result_future = goal_handle.get_result_async()

        # 3. Either wait until completion, or cancel after `cancel_after` seconds!
        start_t = time.monotonic()
        cancel_sent = False
        while rclpy.ok() and not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if cancel_after is not None and not cancel_sent and (time.monotonic() - start_t) >= cancel_after:
                self.get_logger().warning(f"Sending CANCEL after {cancel_after:.1f}s!")
                goal_handle.cancel_goal_async()
                cancel_sent = True

        wrapped = result_future.result()
        if wrapped is not None:
            res = wrapped.result
            self.get_logger().info(
                f"[Result] success={res.success} | message='{res.message}' | "
                f"final_distance={res.final_distance_m:.3f}m"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument("z", type=float)
    parser.add_argument("--cancel-after", type=float, default=None)
    args = parser.parse_args()

    rclpy.init()
    node = ArmActionClient()
    try:
        node.send_move(args.x, args.y, args.z, cancel_after=args.cancel_after)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
