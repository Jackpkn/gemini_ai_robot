"""Day 3 (Part 1): ROS 2 Service Client — calls `/gripper/set_closed` from Python.

Usage inside Docker:
  python3 /workspace/learning/gripper_client.py close
  python3 /workspace/learning/gripper_client.py open
"""

import sys

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool


class GripperClient(Node):
    def __init__(self) -> None:
        super().__init__("gripper_client")

        # Create a client matching the EXACT service type and name of the server.
        self.client = self.create_client(SetBool, "/gripper/set_closed")

    def send_command(self, close: bool) -> SetBool.Response:
        # 1. Always wait until the server node is discovered on DDS!
        if not self.client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("Service /gripper/set_closed is not available!")

        # 2. Build the Request object.
        req = SetBool.Request()
        req.data = close

        # 3. Call asynchronously (`call_async`) and spin until the Future finishes.
        #    Why NOT blocking `call()`? Inside a running ROS 2 executor, a blocking
        #    `call()` can deadlock because the thread waiting for the reply is the
        #    same thread supposed to process incoming DDS packets!
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done():
            raise TimeoutError("Timed out waiting for /gripper/set_closed response")
        result = future.result()
        if result is None:
            raise RuntimeError(f"Service call failed: {future.exception()}")
        return result


def main() -> None:
    # Read "close" or "open" from the command line (default: "close").
    action = sys.argv[1].lower() if len(sys.argv) > 1 else "close"
    if action not in ("close", "open"):
        print("Usage: python3 gripper_client.py [close|open]")
        sys.exit(1)

    rclpy.init()
    node = GripperClient()
    try:
        res = node.send_command(close=(action == "close"))
        node.get_logger().info(f"Reply -> success={res.success}, message='{res.message}'")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
