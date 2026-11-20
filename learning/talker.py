"""Lesson 1: The smallest possible ROS 2 Publisher node.

What it does:
- Connects to ROS 2 middleware (CycloneDDS)
- Creates a node named "talker"
- Broadcasts a text message onto the "/chatter" topic every 1 second
"""

import rclpy                                    # ROS 2's official Python library
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node                     # Base class every ROS 2 node inherits from
from std_msgs.msg import String                 # Standard ROS 2 message type carrying a single `.data` string


class Talker(Node):
    def __init__(self) -> None:
        # 1. Register this process in the ROS 2 graph as "/talker".
        super().__init__("talker")

        # 2. Create a Publisher on topic "/chatter".
        #    - String: every message on this topic must be a std_msgs/msg/String
        #    - "/chatter": the topic name any subscriber can listen to
        #    - 10: queue depth (buffer up to 10 messages if a subscriber momentarily lags)
        self.pub = self.create_publisher(String, "/chatter", 10)

        # 3. Counter so each message is numbered (0, 1, 2, ...).
        self.count = 0

        # 4. Ask ROS 2's event loop to call `self.send_message` every 1.0 second.
        #    (Instead of `while True: time.sleep(1)`, timers keep the node responsive!)
        self.create_timer(1.0, self.send_message)

    def send_message(self) -> None:
        msg = String()
        msg.data = f"hello from robot #{self.count}"
        self.pub.publish(msg)  # Pushes the message into CycloneDDS
        self.get_logger().info(f"Sent: {msg.data}")
        self.count += 1


def main() -> None:
    rclpy.init()          # Start the ROS 2 / DDS communication engine
    
    node = Talker()
    try:
        rclpy.spin(node)  # Keep running forever, firing the 1-second timer until Ctrl+C
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()  # Cleanly disconnect from DDS


if __name__ == "__main__":
    main()
