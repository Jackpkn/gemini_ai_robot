# Phase 1: Docker + ROS 2 Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the empty starter scaffold with a clean Python 3.12 + Linux `arm64` ROS 2 Jazzy Docker environment, a thoroughly commented first ROS 2 Python Pub/Sub package (`teleop_basics`), automated unit + container tests, and a Phase 1 teaching guide.

**Architecture:** All ROS 2 middleware (`rclpy`, `CycloneDDS`, `rmw_zenoh_cpp`) runs inside an Ubuntu 24.04 Linux Docker container while host `uv` tests share the exact Python 3.12 + NumPy 1.x + MuJoCo 3.x dependency baseline. Pure payload/sequence-tracking logic lives in `telemetry_payload.py` (testable fast without ROS 2), while `heartbeat_publisher.py` and `heartbeat_subscriber.py` wrap it in real `rclpy.node.Node` classes inside `ros2_ws/src/teleop_basics`. (Phases 2–6 will each have their own focused, independently testable plan.)

**Tech Stack:** Docker & Docker Compose (`ubuntu:24.04`, `ros:jazzy-ros-base`), Python 3.12, `uv`, `rclpy`, `ament_python`, `colcon`, `std_msgs`, `rmw_cyclonedds_cpp`, `rmw_zenoh_cpp`, `pytest`.

**Spec:** [2026-11-17-robotics-middleware-platform-design.md](file:///Users/pawankumar/Projects/teleop-lab/docs/superpowers/specs/2026-11-17-robotics-middleware-platform-design.md)

## Global Constraints

- Host Python and container Python MUST both align on **Python 3.12 (`>=3.12,<3.13`)**, **`mujoco>=3.2,<4.0`**, and **`numpy>=1.26,<2.0`** (for ROS 2 Jazzy Ubuntu 24.04 ABI parity).
- Native `arm64` Linux containers (`ros:jazzy-ros-base` on Ubuntu 24.04) MUST be used—no hardcoded `linux/amd64` emulation.
- All ROS 2 IPC MUST use genuine `rclpy` + DDS/Zenoh (no fake in-memory Python queue substitutes).
- `.superpowers/` MUST be added to `.gitignore`, and the empty starter scaffold (`main.py`, unused empty directories) MUST be cleaned up.
- Every terminal command, Docker flag, config file, and Python module MUST be thoroughly explained and heavily commented for a robotics beginner.

## Review Focus

- **Malformed or truncated heartbeat JSON:** If a publisher emits invalid JSON or misses required fields (`node_id`, `seq`, `pub_ns`), the parser/subscriber must raise a typed `PayloadValidationError` and log a warning without crashing the ROS 2 spin loop (tested in Task 3).
- **Sequence gaps & out-of-order packets:** When `seq` jumps (e.g., `2 -> 5`) or arrives late (`seq <= last_seq`), `SequenceTracker` must count drops accurately and flag stale packets without negative counts (tested in Task 3).
- **Clock skew / negative delta latency:** If `recv_ns < pub_ns`, latency computation must return `0.0` (and flag `clock_skew=True`) instead of reporting nonsensical negative milliseconds (tested in Task 3).
- **Invalid node configuration (`rate_hz <= 0` or blank `node_id`):** Node/config validation must fail fast with `ValueError` instead of dividing by zero in timer setup (tested in Task 3).
- **Docker container architecture & dual-RMW readiness:** `docker/Dockerfile` and `docker/docker-compose.yml` must avoid `amd64`-only pinning and install both `rmw_cyclonedds_cpp` and `rmw_zenoh_cpp` plus `osmesa` headless GL dependencies (tested in Task 2).

---

### Task 1: Repo Hygiene, Python 3.12 Parity & Initial Spec Commit

**Files:**
- Modify: `.gitignore`
- Modify: `.python-version`
- Modify: `pyproject.toml`
- Delete: `main.py` (empty scaffold hello-world) and unused empty placeholder directories (`evaluator/`, `experiments/`, `operator/`, `recorder/`, `reports/`, `robot/`, `simulation/` — `recorder/` will be created properly in Phase 6)
- Create: `tests/test_repo_config.py`

**Interfaces:**
- Consumes: Design spec Python parity constraint (`>=3.12,<3.13`, `mujoco>=3.2,<4.0`, `numpy>=1.26,<2.0`)
- Produces: Clean Python 3.12 `pyproject.toml`, `.python-version`, `.gitignore`, and `tests/test_repo_config.py`

- [ ] **Step 1: Write the failing repository configuration test**

Create `tests/test_repo_config.py`:

```python
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_python_version_pinned_to_312():
    assert (ROOT / ".python-version").read_text().strip() == "3.12"


def test_pyproject_matches_jazzy_parity_constraints():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    proj = data["project"]
    assert proj["requires-python"] == ">=3.12,<3.13"
    deps = proj["dependencies"]
    assert any(d.startswith("mujoco>=3.2,<4.0") for d in deps)
    assert any(d.startswith("numpy>=1.26,<2.0") for d in deps)
    assert any(d.startswith("pytest>=") for d in data["dependency-groups"]["dev"])


def test_gitignore_excludes_superpowers_and_ros2_build_artifacts():
    lines = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert ".superpowers/" in lines
    assert "ros2_ws/build/" in lines
    assert "ros2_ws/install/" in lines
    assert "ros2_ws/log/" in lines
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_repo_config.py -v` (or `.venv/bin/python -m pytest` once pytest installed)
Expected: FAIL (`3.13 != 3.12`, missing `dev` group, missing `.superpowers/` ignore rules).

- [ ] **Step 3: Update `.python-version`, `pyproject.toml`, `.gitignore`, and remove empty scaffold files**

- Set `.python-version` to `3.12`.
- Update `pyproject.toml` with `requires-python = ">=3.12,<3.13"`, `dependencies = ["mujoco>=3.2,<4.0", "numpy>=1.26,<2.0"]`, and `[dependency-groups] dev = ["pytest>=8.0"]`, plus `[tool.pytest.ini_options]` setting `testpaths = ["tests"]` and `pythonpath = ["ros2_ws/src/teleop_basics"]`.
- Update `.gitignore` to keep `.venv`, `__pycache__/`, and add `.superpowers/`, `ros2_ws/build/`, `ros2_ws/install/`, `ros2_ws/log/`.
- Remove placeholder `main.py` and empty directories (`evaluator`, `experiments`, `operator`, `recorder`, `reports`, `robot`, `simulation`).
- Run `uv sync --python 3.12` (outside sandbox if PyPI resolution is needed, explaining the command to the user first).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_repo_config.py -v`
Expected: PASS (`3 passed`).

- [ ] **Step 5: Commit baseline spec, plan, and Python 3.12 config**

```bash
git add .gitignore .python-version pyproject.toml uv.lock tests/test_repo_config.py docs/superpowers/specs/2026-11-17-robotics-middleware-platform-design.md docs/superpowers/plans/2026-11-17-phase-1-docker-ros2-foundations.md
git commit -m "chore: align repo to Python 3.12 and commit design spec and phase 1 plan"
```

---

### Task 2: Linux `arm64` ROS 2 Jazzy Docker Environment

**Files:**
- Create: `docker/Dockerfile`
- Create: `docker/entrypoint.sh`
- Create: `docker/docker-compose.yml`
- Create: `tests/test_docker_config.py`

**Interfaces:**
- Consumes: `pyproject.toml` dependency constraints (`mujoco>=3.2,<4.0`, `numpy>=1.26,<2.0`)
- Produces: Buildable `teleop-lab:jazzy` Docker image & `docker/docker-compose.yml` service `teleop` with ROS 2 Jazzy, `rmw_cyclonedds_cpp`, `rmw_zenoh_cpp`, `mujoco`, `pytest`, and `ros2_ws` mounted at `/workspace/ros2_ws`.

- [ ] **Step 1: Write the failing Docker configuration test**

Create `tests/test_docker_config.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_uses_ros_jazzy_and_installs_both_rmws_and_osmesa():
    dockerfile = (ROOT / "docker/Dockerfile").read_text()
    assert "FROM ros:jazzy-ros-base" in dockerfile
    assert "platform=linux/amd64" not in dockerfile
    assert "ros-jazzy-rmw-cyclonedds-cpp" in dockerfile
    assert "ros-jazzy-rmw-zenoh-cpp" in dockerfile
    assert "libosmesa6-dev" in dockerfile
    assert "ENV MUJOCO_GL=osmesa" in dockerfile
    assert "mujoco>=3.2,<4.0" in dockerfile
    assert "numpy>=1.26,<2.0" in dockerfile


def test_compose_configures_teleop_service_and_default_cyclonedds():
    compose = (ROOT / "docker/docker-compose.yml").read_text()
    assert "teleop:" in compose
    assert "8000:8000" in compose
    assert "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}" in compose
    assert "../ros2_ws:/workspace/ros2_ws" in compose
    assert "../tests:/workspace/tests" in compose


def test_entrypoint_sources_ros_setup_and_workspace_overlay():
    entrypoint = (ROOT / "docker/entrypoint.sh").read_text()
    assert "source /opt/ros/jazzy/setup.bash" in entrypoint
    assert "/workspace/ros2_ws/install/setup.bash" in entrypoint
    assert 'exec "$@"' in entrypoint
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_docker_config.py -v`
Expected: FAIL (`FileNotFoundError: .../docker/Dockerfile`).

- [ ] **Step 3: Create `docker/Dockerfile`, `docker/entrypoint.sh`, and `docker/docker-compose.yml`**

- `docker/Dockerfile`: Base on `ros:jazzy-ros-base` (native multi-arch `arm64`/`amd64`), install apt packages (`python3-pip`, `python3-pytest`, `python3-colcon-common-extensions`, `ros-jazzy-rmw-cyclonedds-cpp`, `ros-jazzy-rmw-zenoh-cpp`, `ros-jazzy-tf2-ros`, `ros-jazzy-tf2-geometry-msgs`, `libosmesa6-dev`, `libgl1`, `libglfw3`), install Python packages with `--break-system-packages` (`"mujoco>=3.2,<4.0"`, `"numpy>=1.26,<2.0"`, `"fastapi>=0.115"`, `"uvicorn>=0.32"`, `"eclipse-zenoh>=1.0"`), set `ENV MUJOCO_GL=osmesa` and `WORKDIR /workspace`, copy `docker/entrypoint.sh`, and set `ENTRYPOINT ["/entrypoint.sh"]`. Add rich inline `#` comments explaining every instruction (`FROM`, `RUN`, `ENV`, `WORKDIR`, `ENTRYPOINT`) for the learner.
- `docker/entrypoint.sh`: POSIX/Bash script (`chmod +x`) that sources `/opt/ros/jazzy/setup.bash` and (if built) `/workspace/ros2_ws/install/setup.bash`, then runs `exec "$@"`.
- `docker/docker-compose.yml`: Defines `teleop` service building `../` with `docker/Dockerfile`, mapping `8000:8000`, mounting `../ros2_ws:/workspace/ros2_ws` and `../tests:/workspace/tests`, setting `RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}` and `ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}`, and `tty: true` / `stdin_open: true` for interactive learning.

- [ ] **Step 4: Run unit test + verify Docker daemon and build the image**

Run:
1. `.venv/bin/pytest tests/test_docker_config.py -v` -> Expected: PASS (`3 passed`).
2. `docker compose -f docker/docker-compose.yml build` -> Expected: builds `teleop` image cleanly.
3. `docker compose -f docker/docker-compose.yml run --rm teleop python3 -c "import rclpy, mujoco, numpy; print('OK', mujoco.__version__, numpy.__version__)"` -> Expected: prints `OK 3.x 1.26.x`.

- [ ] **Step 5: Commit Docker environment**

```bash
git add docker/Dockerfile docker/entrypoint.sh docker/docker-compose.yml tests/test_docker_config.py
git commit -m "build: add ROS 2 Jazzy arm64 Docker environment with DDS and Zenoh RMWs"
```

---

### Task 3: First ROS 2 Package (`teleop_basics`) — Payload Logic, Publisher, Subscriber & Launch

**Files:**
- Create: `ros2_ws/src/teleop_basics/package.xml`
- Create: `ros2_ws/src/teleop_basics/setup.py`
- Create: `ros2_ws/src/teleop_basics/setup.cfg`
- Create: `ros2_ws/src/teleop_basics/resource/teleop_basics`
- Create: `ros2_ws/src/teleop_basics/teleop_basics/__init__.py`
- Create: `ros2_ws/src/teleop_basics/teleop_basics/telemetry_payload.py`
- Create: `ros2_ws/src/teleop_basics/teleop_basics/heartbeat_publisher.py`
- Create: `ros2_ws/src/teleop_basics/teleop_basics/heartbeat_subscriber.py`
- Create: `ros2_ws/src/teleop_basics/launch/phase1_pubsub.launch.py`
- Create: `tests/test_phase1_payload.py`
- Create: `tests/integration/verify_phase1_pubsub.py`

**Interfaces:**
- Consumes: `std_msgs.msg.String`, `rclpy.node.Node`
- Produces:
  - `PayloadValidationError(ValueError)`
  - `HeartbeatMessage(dataclass: node_id: str, seq: int, pub_ns: int, status: str)`
  - `SequenceUpdate(dataclass: gap: int, is_out_of_order: bool, dropped_total: int)`
  - `validate_publisher_config(node_id: str, rate_hz: float) -> None`
  - `build_heartbeat_payload(node_id: str, seq: int, pub_ns: int, status: str = "OK") -> str`
  - `parse_heartbeat_payload(raw: str) -> HeartbeatMessage`
  - `compute_latency_ms(pub_ns: int, recv_ns: int) -> tuple[float, bool]` (returns `(latency_ms, clock_skew)`)
  - `SequenceTracker.observe(seq: int) -> SequenceUpdate`
  - ROS 2 console scripts & nodes: `heartbeat_publisher` (parameters `node_id: str = "arm_sim"`, `rate_hz: float = 2.0`), `heartbeat_subscriber` on topic `/teleop/heartbeat` (`std_msgs/msg/String`)

- [ ] **Step 1: Write the failing unit tests for payload serialization, validation, latency & sequence tracking**

Create `tests/test_phase1_payload.py`:

```python
import pytest

from teleop_basics.telemetry_payload import (
    HeartbeatMessage,
    PayloadValidationError,
    SequenceTracker,
    build_heartbeat_payload,
    compute_latency_ms,
    parse_heartbeat_payload,
    validate_publisher_config,
)


def test_roundtrip_heartbeat_payload():
    raw = build_heartbeat_payload(node_id="sim_node", seq=7, pub_ns=1_000_000, status="OK")
    msg = parse_heartbeat_payload(raw)
    assert msg == HeartbeatMessage(node_id="sim_node", seq=7, pub_ns=1_000_000, status="OK")


def test_parse_payload_rejects_malformed_json():
    with pytest.raises(PayloadValidationError):
        parse_heartbeat_payload("{not valid json")
    with pytest.raises(PayloadValidationError):
        parse_heartbeat_payload('{"node_id": "sim_node"}')


def test_sequence_tracker_detects_gaps_and_out_of_order():
    tracker = SequenceTracker()
    assert tracker.observe(1).gap == 0
    assert tracker.observe(2).gap == 0
    gap_update = tracker.observe(5)
    assert gap_update.gap == 2
    assert gap_update.is_out_of_order is False
    assert gap_update.dropped_total == 2
    late_update = tracker.observe(4)
    assert late_update.gap == 0
    assert late_update.is_out_of_order is True
    assert late_update.dropped_total == 2


def test_compute_latency_handles_clock_skew():
    assert compute_latency_ms(pub_ns=1_000_000, recv_ns=3_500_000) == (2.5, False)
    assert compute_latency_ms(pub_ns=5_000_000, recv_ns=4_000_000) == (0.0, True)


def test_validate_config_rejects_invalid_rate_and_node_id():
    validate_publisher_config("arm_sim", 2.0)
    with pytest.raises(ValueError):
        validate_publisher_config("  ", 2.0)
    with pytest.raises(ValueError):
        validate_publisher_config("arm_sim", 0.0)
    with pytest.raises(ValueError):
        validate_publisher_config("arm_sim", -5.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_phase1_payload.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'teleop_basics'`).

- [ ] **Step 3: Implement `telemetry_payload.py` and ROS 2 package files**

Implement:
- `ros2_ws/src/teleop_basics/teleop_basics/__init__.py`
- `ros2_ws/src/teleop_basics/teleop_basics/telemetry_payload.py` matching exact signatures above with clear teaching docstrings and inline comments explaining monotonic clocks (`time.monotonic_ns()`), sequence numbers, and why parsers must never crash a robotics control/telemetry loop.
- `package.xml`, `setup.py`, `setup.cfg`, `resource/teleop_basics` for `ament_python` with console script entry points `heartbeat_publisher = teleop_basics.heartbeat_publisher:main` and `heartbeat_subscriber = teleop_basics.heartbeat_subscriber:main`, plus installing `launch/*.launch.py`.

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `.venv/bin/pytest tests/test_phase1_payload.py -v`
Expected: PASS (`5 passed`).

- [ ] **Step 5: Implement `heartbeat_publisher.py`, `heartbeat_subscriber.py`, `phase1_pubsub.launch.py`, and containerized integration verification**

Implement:
- `heartbeat_publisher.py`: Subclasses `rclpy.node.Node`, declares ROS 2 parameters (`node_id`, `rate_hz`), validates via `validate_publisher_config`, creates publisher on `/teleop/heartbeat` (`std_msgs/msg/String`, depth `10`), and publishes `build_heartbeat_payload` using `time.monotonic_ns()` on a timer—with thorough beginner-friendly comments explaining `rclpy.init`, `Node`, parameters, timers, `spin`, and clean `try/finally` shutdown.
- `heartbeat_subscriber.py`: Subclasses `rclpy.node.Node`, subscribes to `/teleop/heartbeat`, parses messages safely (catching `PayloadValidationError` and logging a warning instead of crashing), computes `(latency_ms, clock_skew)`, updates `SequenceTracker`, and logs structured telemetry stats—with detailed comments explaining callbacks and event loops.
- `launch/phase1_pubsub.launch.py`: Launches both nodes cleanly with `LaunchDescription`.
- `tests/integration/verify_phase1_pubsub.py`: Python script run inside the Docker container that builds `teleop_basics` via `colcon build --symlink-install`, spins both nodes briefly via `launch` or subprocesses, verifies `ros2 topic list` contains `/teleop/heartbeat`, and asserts at least 3 valid heartbeat messages are received with non-negative latency and zero drops.

- [ ] **Step 6: Build and verify inside Docker container**

Run inside `teleop` container:
1. `docker compose -f docker/docker-compose.yml run --rm teleop bash -lc "cd /workspace/ros2_ws && colcon build --symlink-install --packages-select teleop_basics && source install/setup.bash && pytest /workspace/tests/test_phase1_payload.py -v"` -> Expected: `colcon build` succeeds and `pytest` passes inside container.
2. Run `docker compose -f docker/docker-compose.yml run --rm teleop python3 /workspace/tests/integration/verify_phase1_pubsub.py` -> Expected: exits `0` after receiving real ROS 2 DDS pub/sub messages.

- [ ] **Step 7: Commit `teleop_basics` and integration check**

```bash
git add ros2_ws/src/teleop_basics docker/docker-compose.yml tests/test_phase1_payload.py tests/integration/verify_phase1_pubsub.py
git commit -m "feat(phase1): add teleop_basics ROS 2 heartbeat pub/sub package and container verification"
```

---

### Task 4: Phase 1 Conceptual Learning Guide & Command Cheat Sheet

**Files:**
- Create: `docs/learning/phase-1-docker-and-ros2-foundations.md`

**Interfaces:**
- Consumes: Actual implemented files in `docker/` and `ros2_ws/src/teleop_basics/`
- Produces: Beginner-friendly study guide covering Day 1 & Day 2 concepts, exact commands, and interview Q&A.

- [ ] **Step 1: Write `docs/learning/phase-1-docker-and-ros2-foundations.md`**

Include clear, jargon-free explanations of:
- Why Linux Docker containers are used on Apple Silicon Mac for ROS 2
- ROS 2 Graph anatomy: Nodes, Topics, Messages, DDS Discovery (`ROS_DOMAIN_ID`), `ament_python`, `colcon`, and `install/setup.bash` overlays
- Annotated walkthrough of `telemetry_payload.py`, `heartbeat_publisher.py`, `heartbeat_subscriber.py`, and `phase1_pubsub.launch.py`
- Every command & flag from Phase 1 (`docker compose build`, `docker compose run --rm`, `colcon build --symlink-install`, `ros2 node list`, `ros2 topic list`, `ros2 topic info /teleop/heartbeat --verbose`, `ros2 topic echo /teleop/heartbeat`, `ros2 launch`)
- 5 interview check-yourself questions (with answers) tailored to Mind Robotics.

- [ ] **Step 2: Run full Phase 1 verification suite**

Run:
1. `.venv/bin/pytest tests/ -v`
2. `docker compose -f docker/docker-compose.yml run --rm teleop bash -lc "cd /workspace/ros2_ws && source install/setup.bash && python3 /workspace/tests/integration/verify_phase1_pubsub.py"`
Expected: All host unit tests pass and container ROS 2 pub/sub verification passes.

- [ ] **Step 3: Commit Phase 1 learning guide**

```bash
git add docs/learning/phase-1-docker-and-ros2-foundations.md
git commit -m "docs(phase1): add Docker and ROS 2 Pub/Sub learning guide"
```
