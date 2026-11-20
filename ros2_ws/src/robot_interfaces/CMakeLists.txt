cmake_minimum_required(VERSION 3.8)
project(robot_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generates Python (and C++) classes from action/MoveArm.action:
#   from robot_interfaces.action import MoveArm
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/MoveArm.action"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
