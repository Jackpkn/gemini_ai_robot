from glob import glob
import os
from setuptools import find_packages, setup

package_name = "teleop_basics"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # Registers this package in the ROS 2 ament resource index so `ros2 pkg list` finds it.
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        # Installs package.xml into share/<package_name> as required by ROS 2 conventions.
        ("share/" + package_name, ["package.xml"]),
        # Installs all ROS 2 launch files so `ros2 launch teleop_basics ...` can find them.
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jackpkn",
    maintainer_email="pawankumarbairwa879@gmail.com",
    description="Phase 1 foundational ROS 2 Python Pub/Sub package.",
    license="MIT",
    # Registers CLI entry points that `ros2 run teleop_basics <executable>` invokes.
    entry_points={
        "console_scripts": [
            "heartbeat_publisher = teleop_basics.heartbeat_publisher:main",
            "heartbeat_subscriber = teleop_basics.heartbeat_subscriber:main",
        ],
    },
)
