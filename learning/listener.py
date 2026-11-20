"""Lesson 2: The smallest possible ROS 2 Subscriber node.

What it does:
- Connects to ROS 2 middleware (CycloneDDS)
- Creates a node named "listener"
- Subscribes to "/chatter" and runs `on_message()` every time `talker.py` publishes
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


class Listener(Node):
    def __init__(self) -> None:
        # 1. Register this process in the ROS 2 graph as "/listener".
        super().__init__("listener")

        # 2. Track how many messages we've received so far.
        self.received = 0

        # 3. Subscribe to topic "/chatter".
        #    - String: MUST match the exact message type talker.py publishes!
        #    - "/chatter": MUST match the exact topic name talker.py publishes to!
        #    - self.on_message: callback function ROS 2 calls automatically on each message
        #    - 10: queue depth (incoming message buffer)
        #    Notice we save it to `self.sub` so Python's garbage collector doesn't destroy it.
        self.sub = self.create_subscription(String, "/chatter", self.on_message, 10)
        self.get_logger().info("Listener ready — waiting for messages on /chatter...")

    def on_message(self, msg: String) -> None:
        # ROS 2 passes the deserialized `String` object directly into `msg`.
        self.received += 1
        self.get_logger().info(f"Heard #{self.received}: {msg.data}")


def main() -> None:
    rclpy.init()
    node = Listener()
    try:
        # Keeps the node awake; whenever CycloneDDS delivers a message on /chatter,
        # rclpy's executor automatically invokes `node.on_message(msg)`.
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
