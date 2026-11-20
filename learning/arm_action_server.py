"""Day 3 (Part 2): ROS 2 Action Server — cancellable arm motion with live feedback.

Why an Action instead of a Service?
-----------------------------------
Moving a physical robot arm takes seconds. Unlike a Service (which blocks with zero
progress and no cancel button), a ROS 2 Action has 3 channels built in:
  1. Goal     — client asks: "Move arm to (x, y, z)"
  2. Feedback — server streams live updates: "40% done, 0.32m left..."
  3. Cancel   — client (or safety watchdog / AI planner) can abort mid-motion!
  4. Result   — final outcome: SUCCEEDED or CANCELED + final distance.
"""

import math
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from robot_interfaces.action import MoveArm


class ArmActionServer(Node):
    def __init__(self) -> None:
        super().__init__("arm_action_server")

        # Simulated current end-effector position (starts at origin).
        self.pos = [0.0, 0.0, 0.0]

        # ReentrantCallbackGroup + MultiThreadedExecutor lets `cancel_callback`
        # run IN PARALLEL while `execute_callback` is stepping the arm!
        # Without this, Cancel would sit stuck in a queue until motion finished.
        self.cb_group = ReentrantCallbackGroup()

        self.action_server = ActionServer(
            self,
            MoveArm,
            "/arm/move",
            execute_callback=self.execute_move,
            goal_callback=self.on_goal,
            cancel_callback=self.on_cancel,
            callback_group=self.cb_group,
        )
        self.get_logger().info("ArmActionServer ready on /arm/move")

    def on_goal(self, goal_request: MoveArm.Goal) -> GoalResponse:
        # Reject obviously unsafe targets outside a 1.0m arm workspace radius!
        dist_from_base = math.dist((0.0, 0.0, 0.0), (goal_request.target_x, goal_request.target_y, goal_request.target_z))
        if dist_from_base > 1.0:
            self.get_logger().warning(f"Rejecting out-of-reach goal (radius={dist_from_base:.2f}m > 1.0m)")
            return GoalResponse.REJECT

        self.get_logger().info(
            f"Accepted goal: target=({goal_request.target_x:.2f}, {goal_request.target_y:.2f}, {goal_request.target_z:.2f})"
        )
        return GoalResponse.ACCEPT

    def on_cancel(self, _goal_handle: ServerGoalHandle) -> CancelResponse:
        self.get_logger().warning("Cancel requested! Stopping arm safely...")
        return CancelResponse.ACCEPT

    def execute_move(self, goal_handle: ServerGoalHandle) -> MoveArm.Result:
        goal = goal_handle.request
        start = list(self.pos)
        target = [goal.target_x, goal.target_y, goal.target_z]
        total_steps = 10

        feedback = MoveArm.Feedback()

        for step in range(1, total_steps + 1):
            # ALWAYS check for cancellation inside long-running robotics loops!
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                remaining = math.dist(self.pos, target)
                self.get_logger().warning(f"Goal CANCELED at step {step}/{total_steps}, remaining={remaining:.3f}m")
                return MoveArm.Result(
                    success=False,
                    message="Canceled safely mid-motion",
                    final_distance_m=remaining,
                )

            # Linearly interpolate our simulated arm position one step closer.
            alpha = step / total_steps
            self.pos = [s + alpha * (t - s) for s, t in zip(start, target)]

            feedback.distance_remaining_m = math.dist(self.pos, target)
            feedback.progress_pct = alpha * 100.0
            goal_handle.publish_feedback(feedback)

            self.get_logger().info(
                f"Moving... {feedback.progress_pct:5.1f}% | pos={[round(v, 2) for v in self.pos]} | "
                f"remaining={feedback.distance_remaining_m:.3f}m"
            )
            time.sleep(0.25)

        goal_handle.succeed()
        self.get_logger().info("Goal SUCCEEDED — arm reached target!")
        return MoveArm.Result(
            success=True,
            message="Reached target pose",
            final_distance_m=0.0,
        )


def main() -> None:
    rclpy.init()
    node = ArmActionServer()
    # MultiThreadedExecutor allows cancel_callback to interrupt execute_move!
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
