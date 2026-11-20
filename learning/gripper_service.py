"""Day 3 (Part 1): ROS 2 Service Server — controls a simulated robot gripper.

Why a Service instead of a Topic?
---------------------------------
- Topics (Pub/Sub) are 1-way broadcasts: you shout and don't get an answer back.
- Services are 2-way Request/Response (like an HTTP API call): the caller asks
  "close the gripper" and gets back `success: True/False` + a `message`.

Service type used: `std_srvs/srv/SetBool`
  Request:  bool data        (True = close gripper, False = open gripper)
  ---
  Response: bool success     (did the command succeed?)
            string message   (human-readable status)
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_srvs.srv import SetBool


class GripperService(Node):
    def __init__(self) -> None:
        super().__init__("gripper_service")

        # Pretend this tracks our robot gripper's physical state.
        self.is_closed = False

        # Expose a Service named "/gripper/set_closed".
        # Whenever a client (Python script or `ros2 service call`) sends a request,
        # ROS 2 runs `self.handle_request(request, response)`.
        self.srv = self.create_service(
            SetBool,
            "/gripper/set_closed",
            self.handle_request,
        )
        self.get_logger().info("GripperService ready on /gripper/set_closed (currently OPEN)")

    def handle_request(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        want_closed = request.data
        state_name = "CLOSED" if want_closed else "OPEN"

        if self.is_closed == want_closed:
            response.success = True
            response.message = f"Gripper was already {state_name}"
        else:
            self.is_closed = want_closed
            response.success = True
            response.message = f"Gripper moved to {state_name}"

        self.get_logger().info(f"Request data={want_closed} -> {response.message}")

        # IMPORTANT: Every ROS 2 service callback MUST return `response`!
        return response


def main() -> None:
    rclpy.init()
    node = GripperService()
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
