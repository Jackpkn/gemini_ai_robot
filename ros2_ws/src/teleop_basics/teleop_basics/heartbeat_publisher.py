"""First ROS 2 Publisher Node: emits periodic heartbeat telemetry on `/teleop/heartbeat`.

Key ROS 2 Concepts Demonstrated Here
-----------------------------------
1. Node (`rclpy.node.Node`): A single-purpose process/component in the ROS 2 graph.
2. Parameters: Runtime settings (`node_id`, `rate_hz`) you can override from the CLI
   or a launch file without editing Python source code.
3. Topic (`create_publisher`): A named, typed message bus (`/teleop/heartbeat`) where
   publishers broadcast data without knowing who (or how many subscribers) are listening.
4. Timer (`create_timer`): Fires a callback at a fixed period inside ROS 2's event loop
   (`rclpy.spin`), replacing manual `while True: time.sleep(...)` loops.
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teleop_basics.telemetry_payload import (
    build_heartbeat_payload,
    validate_publisher_config,
)

HEARTBEAT_TOPIC = "/teleop/heartbeat"


class HeartbeatPublisher(Node):
    """Publishes timestamped heartbeat messages so monitors know a node is alive."""

    def __init__(self) -> None:
        # Register this node in the ROS 2 graph under the name "heartbeat_publisher".
        super().__init__("heartbeat_publisher")

        # Declare configurable parameters with sensible defaults.
        # In ROS 2, you can override these at launch time, e.g.:
        #   ros2 run teleop_basics heartbeat_publisher --ros-args -p rate_hz:=5.0
        self.declare_parameter("node_id", "arm_sim")
        self.declare_parameter("rate_hz", 2.0)

        self._node_id = str(self.get_parameter("node_id").value)
        self._rate_hz = float(self.get_parameter("rate_hz").value)

        # Fail fast if misconfigured (e.g., blank node_id or rate_hz <= 0).
        validate_publisher_config(self._node_id, self._rate_hz)

        # Create a publisher on `/teleop/heartbeat` carrying `std_msgs/msg/String`.
        # `qos_profile=10` means: keep up to 10 outgoing messages in the queue if
        # the network/subscriber momentarily lags. (In Phase 2 we'll tune QoS policies.)
        self._publisher = self.create_publisher(String, HEARTBEAT_TOPIC, 10)
        self._seq = 0

        # Schedule `_on_timer` every `1.0 / rate_hz` seconds inside the ROS 2 executor.
        period_sec = 1.0 / self._rate_hz
        self._timer = self.create_timer(period_sec, self._on_timer)

        self.get_logger().info(
            f"HeartbeatPublisher started: node_id={self._node_id!r}, "
            f"rate={self._rate_hz:.2f} Hz, topic={HEARTBEAT_TOPIC}"
        )

    def _on_timer(self) -> None:
        """Build one heartbeat JSON payload and publish it over DDS/Zenoh."""
        self._seq += 1
        pub_ns = time.monotonic_ns()

        msg = String()
        msg.data = build_heartbeat_payload(
            node_id=self._node_id,
            seq=self._seq,
            pub_ns=pub_ns,
            status="OK",
        )
        self._publisher.publish(msg)
        self.get_logger().info(f"Published heartbeat seq={self._seq}")


def main(args: list[str] | None = None) -> None:
    """Standard ROS 2 Python node entry point invoked by `ros2 run` / `ros2 launch`."""
    # Initialize the rclpy middleware context (sets up DDS/Zenoh participant).
    rclpy.init(args=args)
    node: HeartbeatPublisher | None = None
    try:
        # Construct inside `try` so parameter validation errors still trigger `rclpy.shutdown()`.
        node = HeartbeatPublisher()
        # `spin` hands control to ROS 2's executor so timer callbacks keep firing
        # until Ctrl+C (KeyboardInterrupt) or `rclpy.shutdown()` stops the node.
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Always clean up the node and middleware context cleanly to avoid zombie DDS entities.
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
