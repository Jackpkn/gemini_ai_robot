"""First ROS 2 Subscriber Node: listens on `/teleop/heartbeat` and measures latency.

Key ROS 2 Concepts Demonstrated Here
-----------------------------------
1. Subscription (`create_subscription`): Registers a callback that ROS 2 invokes
   automatically whenever a matching message arrives on `/teleop/heartbeat`.
2. Defensive Parsing: Real robotics topics can receive corrupt or unexpected payloads.
   Catching `PayloadValidationError` prevents one bad packet from killing the node.
3. Transport Telemetry: Computing `recv_ns - pub_ns` and tracking sequence gaps lets
   us observe real middleware latency and message loss from Day 2 onward.
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teleop_basics.telemetry_payload import (
    HeartbeatMessage,
    MultiNodeSequenceTracker,
    PayloadValidationError,
    compute_latency_ms,
    parse_heartbeat_payload,
)

HEARTBEAT_TOPIC = "/teleop/heartbeat"


class HeartbeatSubscriber(Node):
    """Consumes `/teleop/heartbeat` messages and tracks transport latency and drops."""

    def __init__(self) -> None:
        super().__init__("heartbeat_subscriber")

        self._trackers = MultiNodeSequenceTracker()
        self.received_messages: list[tuple[HeartbeatMessage, float]] = []
        self.invalid_count = 0

        # Subscribe to `/teleop/heartbeat` with queue depth 10.
        # Every arriving `std_msgs/msg/String` triggers `_on_heartbeat`.
        self._subscription = self.create_subscription(
            String,
            HEARTBEAT_TOPIC,
            self._on_heartbeat,
            10,
        )
        self.get_logger().info(f"HeartbeatSubscriber listening on {HEARTBEAT_TOPIC}")

    def _on_heartbeat(self, msg: String) -> None:
        """Validate each incoming message, measure one-way latency, and check sequence gaps."""
        recv_ns = time.monotonic_ns()

        try:
            parsed = parse_heartbeat_payload(msg.data)
        except PayloadValidationError as exc:
            self.invalid_count += 1
            self.get_logger().warning(f"Ignoring malformed heartbeat message: {exc}")
            return

        latency_ms, clock_skew = compute_latency_ms(parsed.pub_ns, recv_ns)
        seq_update = self._trackers.observe(parsed.node_id, parsed.seq)
        self.received_messages.append((parsed, latency_ms))

        if clock_skew:
            self.get_logger().warning(
                f"Clock skew detected for node={parsed.node_id} seq={parsed.seq}"
            )
        if seq_update.gap > 0:
            self.get_logger().warning(
                f"Detected {seq_update.gap} dropped message(s) from node={parsed.node_id} "
                f"before seq={parsed.seq} (node_dropped={seq_update.dropped_total})"
            )
        elif seq_update.is_out_of_order:
            self.get_logger().warning(
                f"Out-of-order heartbeat from node={parsed.node_id} seq={parsed.seq}"
            )

        self.get_logger().info(
            f"Received node={parsed.node_id} seq={parsed.seq} status={parsed.status} "
            f"latency={latency_ms:.3f} ms dropped_total={self._trackers.dropped_total}"
        )


def main(args: list[str] | None = None) -> None:
    """Standard ROS 2 Python node entry point invoked by `ros2 run` / `ros2 launch`."""
    rclpy.init(args=args)
    node: HeartbeatSubscriber | None = None
    try:
        node = HeartbeatSubscriber()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
