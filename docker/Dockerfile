# Base image: Official ROS 2 Jazzy Jalisco image built on Ubuntu 24.04 (Noble).
# Notice we do NOT hardcode --platform=linux/amd64, so on Apple Silicon Macs
# Docker pulls the native arm64 image for fast execution without QEMU emulation.
FROM ros:jazzy-ros-base

# Prevent apt-get from popping up interactive prompts (like timezone selection)
# during automated image builds.
ENV DEBIAN_FRONTEND=noninteractive

# Use OSMesa (offscreen CPU OpenGL) so MuJoCo can render robot camera frames
# inside a headless Linux container without needing a Mac X11 window or GPU passthrough.
ENV MUJOCO_GL=osmesa

# Install ROS 2 middleware implementations, TF2 tools, Python build tools, and OSMesa GL libraries:
# - python3-colcon-common-extensions: ROS 2's build tool (`colcon build`) for compiling workspaces
# - python3-pip / python3-pytest: Python package installer & test runner inside the container
# - ros-jazzy-rmw-cyclonedds-cpp: CycloneDDS middleware (standard DDS implementation for ROS 2)
# - ros-jazzy-rmw-zenoh-cpp: Eclipse Zenoh RMW middleware (for DDS vs Zenoh experiments in Phase 4)
# - ros-jazzy-tf2-ros / tf2-geometry-msgs: Coordinate transform library used starting in Phase 3
# - libosmesa6-dev / libgl1 / libglfw3: Offscreen OpenGL libraries required by MuJoCo's Renderer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-pytest \
    python3-colcon-common-extensions \
    ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-rmw-zenoh-cpp \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-geometry-msgs \
    libosmesa6-dev \
    libgl1 \
    libglfw3 \
    && rm -rf /var/lib/apt/lists/*

# Install Python robotics/runtime libraries into Ubuntu 24.04's Python 3.12 environment.
# --break-system-packages is required on Ubuntu 24.04 (PEP 668) inside single-purpose containers.
# --no-cache-dir keeps the Docker image smaller by skipping pip's download cache.
# Notice the pins:
# - mujoco>=3.2,<4.0 matches our host pyproject.toml and supports Python 3.12 arm64 wheels
# - numpy>=1.26,<2.0 preserves compatibility with Ubuntu 24.04's ROS 2 apt Python bindings
RUN pip3 install --no-cache-dir --break-system-packages \
    "mujoco>=3.2,<4.0" \
    "numpy>=1.26,<2.0" \
    "fastapi>=0.115" \
    "uvicorn>=0.32" \
    "eclipse-zenoh>=1.0"

# Set the working directory inside the container where our ROS 2 workspace and tests are mounted.
WORKDIR /workspace

# Copy the container startup script that automatically activates ROS 2 environment variables.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Every `docker compose run` or `docker compose up` command passes through /entrypoint.sh first.
ENTRYPOINT ["/entrypoint.sh"]

# Default command when no command is passed: start an interactive bash shell.
CMD ["bash"]
