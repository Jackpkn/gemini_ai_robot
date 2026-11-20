"""Vision Node ("The Robot's Eyes"): Automatically finds the red cube using ONLY camera pixels!

Notice what this node does NOT touch:
  - NO MuJoCo import!
  - NO hardcoded cube coordinates!
  - It could plug into a real Intel RealSense camera on a physical robot unchanged.

How it works in 3 simple steps:
  1. Looks at `/sim/rgb` pixels and finds where the red pixels are -> pixel (u, v).
  2. Reads `/sim/depth` at those pixels -> distance `d` in meters from the camera lens.
  3. Uses `/sim/camera_info` + TF2 (`world <- overview_camera`) to convert (u, v, d)
     into 3D robot coordinates (x, y, z) and publishes `/detected_cube`!
"""

from geometry_msgs.msg import PointStamped
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener


class CubeDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("cube_detector")
        self.latest_rgb: np.ndarray | None = None
        self.latest_depth: np.ndarray | None = None
        self.camera_k: tuple[float, float, float, float] | None = None  # (fx, fy, cx, cy)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(Image, "/sim/rgb", self.on_rgb, sensor_qos)
        self.create_subscription(Image, "/sim/depth", self.on_depth, sensor_qos)
        self.create_subscription(CameraInfo, "/sim/camera_info", self.on_info, latched_qos)
        self.pub = self.create_publisher(PointStamped, "/detected_cube", 10)
        self.get_logger().info("CubeDetectorNode watching /sim/rgb + /sim/depth -> /detected_cube")

    def on_info(self, msg: CameraInfo) -> None:
        # K = [fx, 0, cx,  0, fy, cy,  0, 0, 1]
        self.camera_k = (float(msg.k[0]), float(msg.k[4]), float(msg.k[2]), float(msg.k[5]))

    def on_rgb(self, msg: Image) -> None:
        self.latest_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

    def on_depth(self, msg: Image) -> None:
        # 16UC1 stores depth in millimeters -> divide by 1000.0 for meters
        self.latest_depth = (
            np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32)
            / 1000.0
        )
        self.detect_and_publish(msg.header.stamp, msg.header.frame_id)

    def detect_and_publish(self, stamp, camera_frame: str) -> None:
        if self.latest_rgb is None or self.latest_depth is None or self.camera_k is None:
            return

        # Step 1: Find red pixels in the camera image (no AI model needed for a red cube!)
        r = self.latest_rgb[:, :, 0].astype(np.int16)
        g = self.latest_rgb[:, :, 1].astype(np.int16)
        b = self.latest_rgb[:, :, 2].astype(np.int16)
        ys, xs = np.where((r > 120) & (r > g * 1.8) & (r > b * 1.8))
        if len(xs) < 5:
            return  # Cube not visible (e.g. temporarily blocked by arm)

        # Step 2: Turn pixel center (u, v) + depth `d` into 3D point in CAMERA coordinates
        u, v = float(xs.mean()), float(ys.mean())
        d = float(np.median(self.latest_depth[ys, xs]))
        fx, fy, cx, cy = self.camera_k

        pt_cam = PointStamped()
        pt_cam.header.stamp = stamp
        pt_cam.header.frame_id = camera_frame
        pt_cam.point.x = (u - cx) / fx * d
        pt_cam.point.y = -(v - cy) / fy * d  # image Y points down; OpenGL camera +Y points up
        pt_cam.point.z = -d  # OpenGL camera looks along -Z

        # Step 3: Ask TF2 to convert from `overview_camera` frame -> `world` (robot base) frame!
        try:
            tf = self.tf_buffer.lookup_transform("world", camera_frame, Time(), timeout=Duration(seconds=0.1))
            pt_world = do_transform_point(pt_cam, tf)
        except TransformException as exc:
            self.get_logger().warning(f"Waiting for TF world<-{camera_frame}: {exc}", throttle_duration_sec=2.0)
            return

        pt_world.header.stamp = stamp
        pt_world.header.frame_id = "world"
        self.pub.publish(pt_world)


def main() -> None:
    rclpy.init()
    node = CubeDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
