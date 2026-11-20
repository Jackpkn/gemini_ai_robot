"""Day 7 (Node 2): MuJoCo Jacobian IK Controller ("The Arm Brain").

Run inside Docker alongside `mujoco_ros_sim.py`:
  python3 /workspace/learning/arm_ik_controller.py

Then control your REAL simulated MuJoCo arm with the exact Day 3 clients you already built!
  python3 /workspace/learning/arm_action_client.py 0.35 0.00 0.12
  python3 /workspace/learning/gripper_client.py close
  python3 /workspace/learning/arm_action_client.py 0.00 0.35 0.15

How Inverse Kinematics (IK) works in 10 lines:
  - Forward Kinematics (Day 5): joint angles -> hand (x, y, z)
  - Inverse Kinematics (here!): desired hand (x, y, z) -> joint angles!
  - At 20 Hz we ask MuJoCo for the Jacobian matrix `J` ("how much does hand (x, y, z)
    move when each joint turns 1 radian?"), solve a tiny damped least-squares step `dq`,
    and publish updated joint targets to `/arm/joint_target` until `ee_tip` arrives!
"""

import math
from pathlib import Path
import time

import mujoco
import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from robot_interfaces.action import MoveArm

MODEL_PATH = Path(__file__).with_name("simple_arm.xml")
ARM_JOINTS = ("base_yaw", "shoulder_pitch", "elbow_pitch")


class ArmIKController(Node):
    def __init__(self) -> None:
        super().__init__("arm_ik_controller")

        # Lightweight MuJoCo model copy used ONLY for math (forward kinematics + Jacobian).
        self.ik_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.ik_data = mujoco.MjData(self.ik_model)
        self.site_id = mujoco.mj_name2id(self.ik_model, mujoco.mjtObj.mjOBJ_SITE, "ee_tip")
        self.qpos_ids = [self.ik_model.joint(n).qposadr[0] for n in ARM_JOINTS]
        self.dof_ids = [self.ik_model.joint(n).dofadr[0] for n in ARM_JOINTS]
        self.wrist_qpos_id = self.ik_model.joint("wrist_pitch").qposadr[0]
        self.wrist_dof_id = self.ik_model.joint("wrist_pitch").dofadr[0]
        self.jnt_limits = [self.ik_model.joint(n).range.copy() for n in ARM_JOINTS]

        self.current_q: np.ndarray | None = None
        self.cb_group = ReentrantCallbackGroup()

        # Subscribe to /joint_states with BEST_EFFORT QoS (matches mujoco_ros_sim.py!).
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self.on_joint_states, sensor_qos, callback_group=self.cb_group
        )
        self.cmd_pub = self.create_publisher(Float64MultiArray, "/arm/joint_target", 10)

        # Expose the EXACT same `/arm/move` Action interface from Day 3b!
        self.action_server = ActionServer(
            self,
            MoveArm,
            "/arm/move",
            execute_callback=self.execute_move,
            goal_callback=self.on_goal,
            cancel_callback=self.on_cancel,
            callback_group=self.cb_group,
        )
        self.get_logger().info("ArmIKController ready on /arm/move (waiting for /joint_states...)")

    def on_joint_states(self, msg: JointState) -> None:
        by_name = dict(zip(msg.name, msg.position))
        if all(n in by_name for n in ARM_JOINTS):
            self.current_q = np.array([by_name[n] for n in ARM_JOINTS], dtype=float)

    def _ee_pos_and_jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute current 3D end-effector position and 3x3 position Jacobian with level wrist."""
        for q_adr, val in zip(self.qpos_ids, q):
            self.ik_data.qpos[q_adr] = val
        # Coupled auto-leveling wrist: wrist_pitch = -(shoulder_pitch + elbow_pitch)
        self.ik_data.qpos[self.wrist_qpos_id] = -(q[1] + q[2])
        mujoco.mj_forward(self.ik_model, self.ik_data)

        ee_pos = self.ik_data.site_xpos[self.site_id].copy()
        jacp = np.zeros((3, self.ik_model.nv), dtype=float)
        mujoco.mj_jacSite(self.ik_model, self.ik_data, jacp, None, self.site_id)
        J = jacp[:, self.dof_ids].copy()
        # Chain rule for d(wrist_pitch)/d(shoulder_pitch) = -1 and d(wrist_pitch)/d(elbow_pitch) = -1
        J[:, 1] -= jacp[:, self.wrist_dof_id]
        J[:, 2] -= jacp[:, self.wrist_dof_id]
        return ee_pos, J

    def on_goal(self, goal_request: MoveArm.Goal) -> GoalResponse:
        if self.current_q is None:
            self.get_logger().warning("Rejecting goal: no /joint_states received from simulator yet!")
            return GoalResponse.REJECT
        dist = math.dist((0.0, 0.0, 0.14), (goal_request.target_x, goal_request.target_y, goal_request.target_z))
        if dist > 0.56 or goal_request.target_z < 0.025:
            self.get_logger().warning(f"Rejecting unreachable/unsafe goal (dist={dist:.2f}m, z={goal_request.target_z:.2f}m)")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def on_cancel(self, _goal_handle: ServerGoalHandle) -> CancelResponse:
        self.get_logger().warning("Cancel received! Holding current arm pose.")
        return CancelResponse.ACCEPT

    def execute_move(self, goal_handle: ServerGoalHandle) -> MoveArm.Result:
        goal = goal_handle.request
        target = np.array([goal.target_x, goal.target_y, goal.target_z], dtype=float)
        assert self.current_q is not None

        q_cmd = self.current_q.copy()
        ee_start, _ = self._ee_pos_and_jacobian(q_cmd)
        initial_dist = max(float(np.linalg.norm(target - ee_start)), 1e-6)

        feedback = MoveArm.Feedback()
        damping = 0.06
        tolerance_m = 0.015

        for step in range(1, 161):  # up to 8.0 seconds at 20 Hz
            if goal_handle.is_cancel_requested:
                # Hold right where the arm currently is!
                self.cmd_pub.publish(Float64MultiArray(data=self.current_q.tolist()))
                ee_now, _ = self._ee_pos_and_jacobian(self.current_q)
                rem = float(np.linalg.norm(target - ee_now))
                goal_handle.canceled()
                return MoveArm.Result(success=False, message="Canceled safely in MuJoCo", final_distance_m=rem)

            # Closed-loop: measure ACTUAL simulated end-effector pose from live /joint_states!
            ee_actual, _ = self._ee_pos_and_jacobian(self.current_q)
            err_actual = target - ee_actual
            dist_actual = float(np.linalg.norm(err_actual))

            feedback.distance_remaining_m = dist_actual
            feedback.progress_pct = float(np.clip((1.0 - dist_actual / initial_dist) * 100.0, 0.0, 100.0))
            goal_handle.publish_feedback(feedback)

            if dist_actual <= tolerance_m:
                goal_handle.succeed()
                self.get_logger().info(
                    f"Reached target! ee={np.round(ee_actual, 3).tolist()}, err={dist_actual:.3f}m"
                )
                return MoveArm.Result(success=True, message="Reached target in MuJoCo", final_distance_m=dist_actual)

            # Damped Least-Squares IK step on our commanded joint vector `q_cmd`:
            ee_cmd, J = self._ee_pos_and_jacobian(q_cmd)
            err_cmd = target - ee_cmd
            dq = J.T @ np.linalg.solve(J @ J.T + (damping**2) * np.eye(3), err_cmd)
            dq = np.clip(dq, -0.10, 0.10)  # smooth max speed per 20 Hz tick
            q_cmd = np.array([
                float(np.clip(q_cmd[i] + dq[i], self.jnt_limits[i][0], self.jnt_limits[i][1]))
                for i in range(3)
            ])
            self.cmd_pub.publish(Float64MultiArray(data=q_cmd.tolist()))
            time.sleep(0.05)  # 20 Hz control loop

        ee_final, _ = self._ee_pos_and_jacobian(self.current_q)
        rem = float(np.linalg.norm(target - ee_final))
        goal_handle.abort()
        return MoveArm.Result(success=False, message="IK move timed out before reaching target", final_distance_m=rem)


def main() -> None:
    rclpy.init()
    node = ArmIKController()
    executor = MultiThreadedExecutor(num_threads=3)
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
