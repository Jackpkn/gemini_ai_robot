"""Day 7 (Node 1): MuJoCo <-> ROS 2 Simulator Bridge ("The Robot Body & Senses").

Run inside Docker:
  python3 /workspace/learning/mujoco_ros_sim.py

What this node connects:
  - Runs real MuJoCo physics (`simple_arm.xml`) at 50 Hz (10 physics substeps per tick).
  - Publishes `/joint_states` (`sensor_msgs/JointState`, BEST_EFFORT QoS -> Day 2 & 4!).
  - Broadcasts `/tf` (`world -> ee_tip` and `world -> target_cube` -> Day 5!).
  - Subscribes to `/arm/joint_target` (`std_msgs/Float64MultiArray`, 3 arm joint targets).
  - Exposes `/gripper/set_closed` (`std_srvs/SetBool` -> Day 3a! Same client works!).
"""

from io import BytesIO
import math
from pathlib import Path
import random
import struct
import zlib

from geometry_msgs.msg import TransformStamped
import mujoco
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool, Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

try:
    from PIL import Image as _PILImage  # fast JPEG if Pillow is installed
except ImportError:
    _PILImage = None

MODEL_PATH = Path(__file__).with_name("simple_arm.xml")
ARM_JOINTS = ("base_yaw", "shoulder_pitch", "elbow_pitch")
ALL_JOINTS = (
    "base_yaw",
    "shoulder_pitch",
    "elbow_pitch",
    "wrist_pitch",
    "left_finger_slide",
    "right_finger_slide",
)


def encode_rgb_frame(rgb: np.ndarray) -> tuple[str, bytes]:
    """Encode an HxWx3 uint8 RGB array to JPEG (if Pillow exists) or zero-dep PNG."""
    if _PILImage is not None:
        buf = BytesIO()
        _PILImage.fromarray(rgb).save(buf, format="JPEG", quality=80)
        return "jpeg", buf.getvalue()

    height, width, _ = rgb.shape
    raw_rows = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack("!I", len(payload)) + tag + payload + struct.pack("!I", crc)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw_rows, level=1))
        + _chunk(b"IEND", b"")
    )
    return "png", png


class MuJoCoRosSim(Node):
    def __init__(self) -> None:
        super().__init__("mujoco_ros_sim")

        # 1. Load MuJoCo physics model, live state, and offscreen RGB-D camera renderer.
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height=180, width=240)
        # Warm up software OpenGL shader cache once before starting the 50 Hz physics timer.
        self.renderer.update_scene(self.data, camera="overview")
        self.renderer.render()
        self.tick_count = 0

        # Actuator targets: [base_yaw, shoulder_pitch, elbow_pitch, wrist_pitch, left_finger, right_finger]
        self.target_ctrl = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 2. Sensor topics use BEST_EFFORT QoS (Day 4 rule: sensors & video drop stale frames!).
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        # Latched QoS (TRANSIENT_LOCAL) for static camera intrinsics so late-joining vision nodes get them!
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.joint_pub = self.create_publisher(JointState, "/joint_states", sensor_qos)
        self.cam_pub = self.create_publisher(CompressedImage, "/sim/camera", sensor_qos)
        self.rgb_pub = self.create_publisher(Image, "/sim/rgb", sensor_qos)
        self.depth_pub = self.create_publisher(Image, "/sim/depth", sensor_qos)
        self.info_pub = self.create_publisher(CameraInfo, "/sim/camera_info", latched_qos)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        self._publish_camera_calibration()

        # 3. Command interfaces (reliable by default!).
        self.cmd_sub = self.create_subscription(
            Float64MultiArray, "/arm/joint_target", self.on_joint_target, 10
        )
        self.gripper_srv = self.create_service(
            SetBool, "/gripper/set_closed", self.on_gripper_request
        )
        self.rand_cube_srv = self.create_service(
            Trigger, "/sim/randomize_cube", self.on_randomize_cube
        )

        # 4. Step physics at 50 Hz (0.02s per tick = 10 MuJoCo 0.002s substeps).
        self.create_timer(0.02, self.step_sim)
        self.get_logger().info(
            "MuJoCoRosSim running at 50 Hz (/joint_states, /sim/camera, /sim/rgb, /sim/depth, /tf)"
        )

    def _publish_camera_calibration(self) -> None:
        """Broadcast fixed camera pose on /tf_static and intrinsics on /sim/camera_info."""
        cam_id = self.model.camera("overview").id
        cam_pos = self.data.cam_xpos[cam_id]
        quat_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(quat_wxyz, self.data.cam_xmat[cam_id])

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = "overview_camera"
        t.transform.translation.x = float(cam_pos[0])
        t.transform.translation.y = float(cam_pos[1])
        t.transform.translation.z = float(cam_pos[2])
        t.transform.rotation.w = float(quat_wxyz[0])
        t.transform.rotation.x = float(quat_wxyz[1])
        t.transform.rotation.y = float(quat_wxyz[2])
        t.transform.rotation.z = float(quat_wxyz[3])
        self.static_tf.sendTransform(t)

        # Intrinsics for the 120x90 RGB-D vision stream (2x downsampled from 240x180)
        sw, sh = 120, 90
        fovy = float(self.model.cam_fovy[cam_id])
        fy = (sh / 2.0) / math.tan(math.radians(fovy) / 2.0)
        cx, cy = (sw - 1) / 2.0, (sh - 1) / 2.0
        info = CameraInfo()
        info.header.stamp = t.header.stamp
        info.header.frame_id = "overview_camera"
        info.width = sw
        info.height = sh
        info.k = [fy, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self.info_pub.publish(info)

    def on_randomize_cube(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Teleport the red cube to a random reachable spot on the table to test vision!"""
        radius = random.uniform(0.24, 0.39)
        angle = random.uniform(-0.85, 0.60)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        jnt_id = self.model.joint("cube_free").id
        qpos_adr = self.model.jnt_qposadr[jnt_id]
        qvel_adr = self.model.jnt_dofadr[jnt_id]
        self.data.qpos[qpos_adr : qpos_adr + 7] = [x, y, 0.03, 1.0, 0.0, 0.0, 0.0]
        self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        response.success = True
        response.message = f"Moved red cube to ({x:.2f}, {y:.2f}, 0.03)"
        self.get_logger().info(response.message)
        return response

    def on_joint_target(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != 3:
            self.get_logger().warning(f"Ignoring /arm/joint_target with {len(msg.data)} values (expected 3)")
            return
        self.target_ctrl[0] = float(msg.data[0])
        self.target_ctrl[1] = float(msg.data[1])
        self.target_ctrl[2] = float(msg.data[2])
        # Keep the wrist strictly vertical so downward gripper fingers always approach top-down!
        self.target_ctrl[3] = -(self.target_ctrl[1] + self.target_ctrl[2])

    def on_gripper_request(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        # 0.036m closes both vertical rubber jaws firmly; 0.0m opens them 8.0cm wide.
        finger_val = 0.036 if request.data else 0.0
        self.target_ctrl[4] = finger_val
        self.target_ctrl[5] = finger_val
        state = "CLOSED" if request.data else "OPEN"
        response.success = True
        response.message = f"MuJoCo gripper set to {state} ({finger_val:.3f}m)"
        self.get_logger().info(response.message)
        return response

    def _publish_tf(self, stamp, child_frame: str, pos) -> None:
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "world"
        t.child_frame_id = child_frame
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def step_sim(self) -> None:
        self.data.ctrl[:] = self.target_ctrl
        for _ in range(10):  # 10 * 0.002s = 0.02s simulated time (real-time 50 Hz)
            mujoco.mj_step(self.model, self.data)

        stamp = self.get_clock().now().to_msg()

        # Publish /joint_states
        js = JointState()
        js.header.stamp = stamp
        js.name = list(ALL_JOINTS)
        js.position = [float(self.data.joint(name).qpos[0]) for name in ALL_JOINTS]
        js.velocity = [float(self.data.joint(name).qvel[0]) for name in ALL_JOINTS]
        self.joint_pub.publish(js)

        # Publish /tf for end-effector tip and tabletop objects
        self._publish_tf(stamp, "ee_tip", self.data.site("ee_tip").xpos)
        self._publish_tf(stamp, "target_cube", self.data.body("target_cube").xpos)
        self._publish_tf(stamp, "yellow_block", self.data.body("yellow_block").xpos)
        self._publish_tf(stamp, "purple_cup", self.data.body("purple_cup").xpos)

        # Render & publish overhead camera at ~3.3 Hz (cuts OSMesa CPU load & speeds up physics!)
        self.tick_count += 1
        if self.tick_count % 15 == 0:
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(self.data, camera="overview")
            rgb = self.renderer.render()

            # 1) Compressed image for Chrome viewer
            fmt, payload = encode_rgb_frame(rgb)
            cam_msg = CompressedImage()
            cam_msg.header.stamp = stamp
            cam_msg.header.frame_id = "overview_camera"
            cam_msg.format = fmt
            cam_msg.data = payload
            if rclpy.ok():
                self.cam_pub.publish(cam_msg)

            # 2) Raw 160x120 RGB + Metric Depth for ROS 2 vision nodes (like a RealSense camera!)
            rgb_small = np.ascontiguousarray(rgb[::2, ::2])
            rgb_msg = Image()
            rgb_msg.header.stamp = stamp
            rgb_msg.header.frame_id = "overview_camera"
            rgb_msg.height, rgb_msg.width = rgb_small.shape[:2]
            rgb_msg.encoding = "rgb8"
            rgb_msg.step = rgb_msg.width * 3
            rgb_msg.data = rgb_small.tobytes()
            if rclpy.ok():
                self.rgb_pub.publish(rgb_msg)

            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(self.data, camera="overview")
            depth = self.renderer.render()
            # RealSense-style 16UC1 depth in millimeters (38 KB -> fits cleanly under 64 KB UDP!)
            depth_mm = np.ascontiguousarray(np.clip(depth[::2, ::2] * 1000.0, 0, 65535).astype(np.uint16))
            depth_msg = Image()
            depth_msg.header.stamp = stamp
            depth_msg.header.frame_id = "overview_camera"
            depth_msg.height, depth_msg.width = depth_mm.shape[:2]
            depth_msg.encoding = "16UC1"
            depth_msg.step = depth_msg.width * 2
            depth_msg.data = depth_mm.tobytes()
            if rclpy.ok():
                self.depth_pub.publish(depth_msg)


def main() -> None:
    rclpy.init()
    node = MuJoCoRosSim()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.renderer.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
