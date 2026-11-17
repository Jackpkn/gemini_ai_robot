# Teleop-Lab: Robotics Middleware & Task Runtime Platform Design

**Date:** 2026-11-17  
**Target Role:** Robotics Software Engineer (Mind Robotics — Physical AI, Middleware & Runtime Systems)  
**Status:** Ready for User Approval (Builder Check Completed)

---

## 1. Intent & Why

The goal of this project is to replace the empty `teleop-lab` scaffold with a **real, working robotics middleware and runtime systems engineering platform** written in Python and executed inside reproducible Linux Docker containers.

Rather than building a generic AI agent wrapper or toy script, this platform implements the core infrastructure Mind Robotics builds for factory-floor manipulation and covers every topic in the 14-day robotics + systems curriculum:
- **Robotics Middleware:** Real ROS 2 nodes, Topics (Pub/Sub), Services, Actions, Executors, Callback Groups, QoS policies, TF2 coordinate frames, and switchable DDS vs. Eclipse Zenoh transports.
- **Simulation & Kinematics:** A simulated 6-DOF industrial robotic arm and parallel-jaw gripper performing pick-and-place manipulation in **MuJoCo** (building on the `mujoco` dependency already in `pyproject.toml`).
- **Runtime & Reliability:** An explicit state-machine **Task Execution Engine** with resource locking, deadlines/timeouts, preemptive cancellation, fault injection, and safe recovery.
- **Observability & Operator Tooling:** Monotonic latency measurement (`p50`/`p95`/`p99`), node heartbeats, failure tracing, and a browser-based **Operator Console** with live visualization, control, fault injection, and episode data annotation.
- **Production Practices:** Docker Compose orchestration, automated `pytest` unit/integration tests, GitHub Actions CI, and architecture documentation.

Equally important is **how** this is built: the human partner comes from AI agents and simpler Python software with zero prior robotics middleware experience and wants to learn every concept thoroughly at an unhurried pace. Therefore, implementation will proceed in **6 progressive learning phases**, with detailed code comments, explanations of the *what*, *why*, and *how* for every robotics/systems concept, and explicit explanations for every shell/Docker command used.

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
1. Run authentic **ROS 2 Jazzy** (`rclpy`) processes inside `arm64`-compatible Linux Docker containers (`ubuntu:24.04`), communicating over real **DDS** (`CycloneDDS` / `FastDDS`) and **Eclipse Zenoh** (`rmw_zenoh_cpp` + `zenohd` router).
2. Simulate a 6-DOF robot arm + tabletop manipulation cell in **MuJoCo** with headless offscreen rendering, publishing joint states, RGB camera frames, and **TF2** coordinate transforms.
3. Demonstrate ROS 2 execution mechanics explicitly: Services, cancellable Actions with feedback, `MultiThreadedExecutor` + `CallbackGroup` concurrency, Lifecycle node states, and QoS reliability/durability trade-offs.
4. Build a **Task Execution Engine** that enforces exclusive arm ownership, deadlines/timeouts, cancellation preemption, fault injection, and safe-home recovery.
5. Measure and compare **DDS vs. Zenoh** latency, jitter, serialization overhead, and drop behavior using monotonic timestamps and correlation trace IDs.
6. Provide a web-based **Operator Console** (`localhost:8000`) for live arm camera viewing, joint/TF2 inspection, task dispatch/cancellation, fault injection, live latency/health status, and **episode recording + outcome annotation**.
7. Provide automated `pytest` unit + integration tests, GitHub Actions CI, rich per-phase learning guides, and final resume/interview prep materials.

### Non-Goals
- Building an LLM/VLA chatbot or generic AI-agent orchestration framework.
- Integrating physical robot hardware drivers or a real-time Linux kernel (`PREEMPT_RT`).
- Running Gazebo Desktop GUI inside macOS Docker.
- Multi-robot fleet scheduling (scope is one 6-DOF manipulation cell).
- Training ML policies (though episode recording + annotation supports downstream physical-AI data workflows).

### Anti-Goals (Failure Modes to Avoid)
- **Fake middleware:** Simulating ROS 2 topics/actions using Python in-memory queues instead of real `rclpy` + DDS/Zenoh IPC.
- **Code dumping:** Writing the entire codebase at once without explaining concepts, design trade-offs, and terminal commands step by step.
- **Silent blocking:** Blocking executor threads with synchronous sleeps so cancellation or heartbeat callbacks starve, or swallowing timeouts/errors without explicit state transitions.

---

## 3. Constraints & Environment

- **Host Machine:** macOS on Apple Silicon (`arm64`), with `uv`, `python3`, and `docker` CLI (`29.7.2`) installed; no native `ros2` or `zenohd` binaries.
- **Containerized Execution:** Because ROS 2 and robotics middleware target Linux, all ROS 2, DDS, Zenoh, and MuJoCo runtime components execute inside Linux Docker containers (`arm64` native images, avoiding slow `amd64` QEMU emulation).
- **Docker Daemon Prerequisite:** Docker Desktop or OrbStack must be running on the host Mac when launching containers (verified explicitly with the user in Phase 1).
- **Headless Visuals without X11:** MuJoCo renders frames offscreen inside Docker (`MUJOCO_GL=osmesa` or `egl` fallback) and streams them to the browser Operator Console on `localhost`, avoiding macOS X11/OpenGL forwarding issues.
- **Repo Replacement & Python Parity *(my call)*:** The current `teleop-lab` starter scaffold has no git commits, an empty `README.md`, a placeholder `main.py`, empty subdirectories (`evaluator`, `experiments`, `operator`, `recorder`, `reports`, `robot`, `simulation`), and `.python-version`/`pyproject.toml`/`uv.lock` pinned to Python `>=3.13` with `mujoco==4.0.1` and `numpy==2.5.5` (which lack Python 3.12 wheels and mismatch Ubuntu 24.04 ROS 2 Jazzy's Python 3.12 + NumPy 1.x ABI). Because the user authorized replacing the starter scaffold, we will align `.python-version` and `pyproject.toml` to **Python 3.12 (`>=3.12,<3.13`)**, **`mujoco>=3.2,<4.0`**, and **`numpy>=1.26,<2.0`** so macOS host `uv` unit tests and Docker ROS 2 Jazzy share identical Python and library APIs. Also add `.superpowers/` to `.gitignore`.
- **Teaching Cadence:** Never rush multiple phases into one turn; explain every command, flag, configuration, and code module clearly with comprehensive inline comments.

---

## 4. Approaches Considered & Selection

| Approach | Summary | Trade-offs | Decision |
|---|---|---|---|
| **1. All-Docker Linux Middleware + Headless MuJoCo + Web Operator UI** *(Selected)* | ROS 2, DDS, Zenoh, MuJoCo sim, Task Engine, Telemetry, and FastAPI Operator UI run in Linux Docker Compose; user opens UI in Mac browser. | Uses MuJoCo instead of Gazebo GUI (better fit for manipulation & avoids Mac Docker OpenGL issues); requires Docker daemon running on Mac. | **Picked by user** — cleanest single-environment workflow, matches Mind Robotics' Python/manipulation/middleware/UI/Docker stack, deterministic in CI. |
| **2. All-Docker Linux Middleware + Gazebo Harmonic** | Same middleware stack, using Gazebo + `ros_gz_bridge` inside Docker. | Follows "Install Gazebo" literally, but Gazebo 3D GUI inside Apple Silicon Docker relies on slow software rendering + VNC workarounds and ignores `mujoco` already in `pyproject.toml`. | Rejected. |
| **3. Hybrid Mac-native MuJoCo/UI + Docker ROS 2 via Zenoh** | Native Mac Python runs MuJoCo viewer & UI; Linux container runs ROS 2/DDS bridged via Zenoh. | Smooth native Mac 3D window, but splits debugging across two Python environments and host-container networking from Day 1, complicating early learning and CI. | Rejected. |

---

## 5. System Architecture & Component Design

### 5.1 High-Level 4-Layer Architecture

```text
+-------------------------------------------------------------------+
| Layer 4: Operator Console & Telemetry                             |
|  - Browser UI (localhost:8000): Live camera, TF2 pose, controls,  |
|    latency charts, fault toggles, episode recorder & annotator    |
|  - Watchdog / Heartbeat Monitor & Latency Percentile Aggregator   |
+-------------------------------^-----------------------------------+
                                | WebSocket / REST + Zenoh / ROS 2
+-------------------------------v-----------------------------------+
| Layer 3: Task Execution Engine (Runtime Core)                     |
|  - Explicit State Machine + Single-Arm Resource Lease Lock        |
|  - Step & Global Deadlines, Preemptive Cancellation, Safe Recovery|
|  - Fault Injection Interface (sensor_stall, timeout, grasp_slip)  |
+-------------------------------^-----------------------------------+
                                | ROS 2 Actions, Services, Lifecycle
+-------------------------------v-----------------------------------+
| Layer 2: Robotics Middleware Bus                                  |
|  - ROS 2 Jazzy (rclpy) Topics, Services, Actions, TF2, Executors  |
|  - Configurable QoS Profiles (Reliable vs Best-Effort, Depth)     |
|  - Switchable RMW Transport: CycloneDDS (DDS) <-> rmw_zenoh (Zenoh)|
+-------------------------------^-----------------------------------+
                                | Topics (/joint_states, /tf, /cmd)
+-------------------------------v-----------------------------------+
| Layer 1: MuJoCo 6-DOF Arm Simulation & Kinematics                 |
|  - Self-contained 6-DOF industrial arm + gripper MJCF + URDF      |
|  - Physics loop (50-100 Hz) + Offscreen camera renderer (15-20 Hz)|
|  - Kinematics/IK Controller node (Damped Least-Squares IK + TF2)  |
+-------------------------------------------------------------------+
```

### 5.2 Core Components

1. **Docker & Middleware Runtime (`docker/`)** *(Builder Calls: ROS 2 Jazzy & Single Runtime Container + Zenoh Router)*
   - **ROS 2 Distro *(my call)*:** Use **ROS 2 Jazzy** on `ubuntu:24.04` (`arm64`) because `ros-jazzy-rmw-zenoh-cpp` and `ros-jazzy-rmw-cyclonedds-cpp` are available cleanly as binary packages alongside Ubuntu 24.04's Python 3.12.
   - **Container Topology *(my call)*:** Run all ROS 2 nodes, MuJoCo, and the FastAPI Operator UI inside **one shared `teleop` runtime container** (exposing port `8000`) plus **one `zenohd` router container** (activated in the Compose `zenoh` profile), with an optional `bench_remote` peer container in Phase 4 to demonstrate cross-container Zenoh routing. Why: on macOS Docker Desktop, DDS UDP multicast does not cross separate Docker bridge containers reliably without static unicast peer XML; keeping the main nodes in one container keeps CycloneDDS loopback/SHM rock-solid in Phases 1–3 and guarantees identical `time.monotonic_ns()` kernel timestamps across nodes, while still letting `rmw_zenoh_cpp` cross container boundaries to `zenohd` over TCP.
   - Env/profile switching toggles `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` vs `RMW_IMPLEMENTATION=rmw_zenoh_cpp`. Shared `ros2_ws` volume mount in dev mode reflects code edits quickly while production/CI builds bake code into the image.

2. **MuJoCo Robot Cell & Sim Bridge (`ros2_ws/src/teleop_sim/`)** *(Builder Calls: Self-contained 6-DOF arm, LifecycleNode, Contact Grasp)*
   - **Robot Asset *(my call)*:** Author a clean, self-contained **6-DOF industrial arm + 2-finger parallel gripper** MJCF model (`arm_cell.xml`) and matching URDF (`arm_cell.urdf`) right in the repository (tabletop, target pick cube, drop bin, overhead camera) so the project has zero external asset download dependencies.
   - **Lifecycle *(my call)*:** `sim_node.py` implements `rclpy.lifecycle.LifecycleNode` (`Unconfigured -> Inactive -> Active -> Finalized`) using `LifecyclePublisher`s gated on `Active` (introduced as a plain `Node` in Phases 1–2 and upgraded to `LifecycleNode` in Phase 3/5).
   - Steps MuJoCo physics at `50–100 Hz`, renders overhead camera JPEG frames via `mujoco.Renderer` (`MUJOCO_GL=osmesa`), and publishes:
     - `/joint_states` (`sensor_msgs/msg/JointState`)
     - `/sim/camera/compressed` (`sensor_msgs/msg/CompressedImage`)
     - `/tf` & `/tf_static` (`tf2_msgs/msg/TFMessage` via `tf2_ros`) for `world -> base_link -> link1..6 -> ee_link`, plus `target_cube` and `drop_zone` frames
     - `/sim/grasp_state` (custom msg with gripper aperture, finger-cube contact flag, and `ee_link`-to-`target_cube` distance)
     - `/sim/telemetry_tick` (custom stamped message with `seq_id` and monotonic `pub_ns` for precise latency checks)
   - **Grasp & Slip Physics *(my call)*:** Grasping uses physical finger-cube contact friction (no fake weld cheat). Fault injection for `grasp_slip` temporarily relaxes finger grip force during lift so the cube physically slips.
   - Subscribes to `/arm/joint_command` (with selectable QoS profile) and exposes `/sim/reset` (`std_srvs/srv/Trigger`) to reset cube and robot pose.

3. **Arm Kinematics & Motion Primitives (`ros2_ws/src/teleop_control/`)** *(Builder Calls: LifecycleNode + Step-Level Primitives)*
   - **Lifecycle *(my call)*:** `arm_controller` implements `rclpy.lifecycle.LifecycleNode`, publishing joint commands only when `Active`.
   - Reads current robot state from `/joint_states` and target poses via `tf2_ros.Buffer` / `TransformListener`.
   - Implements Jacobian damped-least-squares Inverse Kinematics (IK) in Python + NumPy to move `ee_link` smoothly to requested Cartesian poses.
   - **Primitive Boundary *(my call)*:** Rather than one monolithic pick-and-place action, `teleop_control` exposes modular primitives so `teleop_runtime` owns multi-step task orchestration:
     - Service `/arm/home` (`std_srvs/srv/Trigger`)
     - Service `/arm/set_gripper` (`std_srvs/srv/SetBool`, open/close)
     - Action `/arm/move_to_pose` (custom cancellable `MoveToPose.action`: goal = `target_pose` + motion profile + step timeout; feedback = `distance_to_goal_m`, `progress_pct`; result = `success`, `elapsed_ms`, `failure_code`).
   - **Concurrency rule:** Uses `MultiThreadedExecutor` + `ReentrantCallbackGroup` / isolated callback groups so motion execution loops never starve incoming `cancel_goal` requests, lifecycle transitions, or `/heartbeat` timers.

4. **Task Execution Engine (`ros2_ws/src/teleop_runtime/`)** *(Builder Calls: Orchestrated Steps + Safe Fail on Slip)*
   - Manages task execution and orchestrates node lifecycle bringup/checks (`sim_node` and `arm_controller` must be `Active`, transitioned via a lifecycle bringup helper at startup and inspected/deactivated on severe faults).
   - Uses an explicit state machine calling `teleop_control` primitives (`MoveToPose`, `/arm/set_gripper`, `/arm/home`):
     - States: `IDLE -> VALIDATING -> ACQUIRING_LOCK -> RUNNING_PICK (pre_grasp -> approach -> close_gripper -> lift) -> RUNNING_PLACE (move_above_drop -> lower -> open_gripper -> retract) -> SUCCEEDED`
     - Failure/Preemption paths: `-> CANCELING -> RECOVERING -> CANCELED` or `-> RECOVERING -> FAILED`
   - **Resource Manager:** Enforces an exclusive lock (`arm_cell_1` lease token) so concurrent requests cannot collide; conflicting requests receive clear rejection/busy status.
   - **Timeouts, Cancellation & `grasp_slip` Recovery *(my call)*:**
     - Enforces both per-step timeouts (each `MoveToPose`/gripper step must finish within N seconds) and overall task deadlines.
     - Monitors `/sim/grasp_state` while carrying the cube: if `ee_link`–`target_cube` distance exceeds threshold or contact is lost while gripper is commanded closed, triggers `grasp_slip` recovery.
     - Upon user cancel, timeout, or `grasp_slip`, immediately cancels the active `/arm/move_to_pose` goal, enters `RECOVERING` (opens gripper safely, calls `/arm/home`, releases arm lock), and finishes in `CANCELED` or `FAILED` with a structured reason code (operator manually retries after inspecting/annotating).
   - **Fault Injection Hooks:** Service/API to simulate `sensor_stall` (sim stops publishing `/joint_states`), `action_timeout` (controller stalls), or `grasp_slip` (cube drops mid-lift), proving recovery works deterministically in tests and demos.

5. **DDS vs. Zenoh Middleware & Benchmark Suite (`ros2_ws/src/teleop_bench/`)**
   - Runs automated pub-sub and service/action round-trip benchmarks under identical payloads across:
     1. **DDS (`rmw_cyclonedds_cpp`)** with `RELIABLE` vs. `BEST_EFFORT` QoS
     2. **Zenoh (`rmw_zenoh_cpp` via `zenohd`)**
     3. **Direct Python `eclipse-zenoh` (`zenoh` package)** pub-sub + serialization comparison (e.g., CDR vs. JSON payload overhead)
   - Outputs reproducible latency percentile reports (`p50`, `p95`, `p99`, max jitter, message loss %) to JSON/Markdown.

6. **Telemetry, Tracing & Health Monitor (`ros2_ws/src/teleop_telemetry/`)**
   - Every task & control message carries a `trace_id` / `task_id` and monotonic nanosecond timestamps (`time.monotonic_ns()` inside same container/host clock or ROS `Clock`).
   - Node heartbeat monitor watches `/system/heartbeat` from `sim_node`, `arm_controller`, and `task_engine`; marks nodes `HEALTHY`, `DEGRADED`, or `DOWN` if missed beyond threshold.
   - Aggregates live latency histograms and recent structured trace events for the Operator UI and logs.

7. **Web Operator Console & Episode Recorder (`operator_ui/` & `recorder/`)** *(Builder Call: FastAPI + zero-build HTML/JS)*
   - **Choice made (my call):** Build the Operator UI with **FastAPI + WebSockets + clean static HTML/Vanilla JS/Canvas** rather than React/Vite so Docker builds stay lightweight, Python-centric, and require zero Node.js toolchain inside the container.
   - **Operator UI Capabilities:**
     - Live rendered MuJoCo arm video stream + joint angle table + TF2 end-effector `(x, y, z)` readout
     - Live middleware transport badge (`DDS` vs. `Zenoh`) & node health status indicators
     - Live Task Execution state machine display + controls (`Run Pick & Place`, `Home Arm`, `Cancel Task`, `Reset Cell`)
     - Fault Injection buttons (`Simulate Sensor Stall`, `Simulate Timeout`, `Simulate Grasp Slip`)
     - Live latency stats (`p50`, `p95`, `p99`)
     - **Episode Recorder & Data Annotation Panel:** Start/stop recording synchronized episodes (timestamps, joint states, commands, task state transitions, latency stats) saved as structured JSON/NPZ bundles, with one-click operator outcome tags (`success`, `grasp_slip`, `timeout_recovered`, `canceled`) and free-text notes.

---

## 6. Phased Teaching & Build Plan (6 Phases)

We will build and verify **one phase at a time**, never skipping ahead until the current phase runs, passes checks, and is clearly understood:

1. **Phase 1 — Docker + ROS 2 Foundations (Days 1–2):** Verify Docker on Mac, build base ROS 2 Jazzy image, understand ROS 2 graph/packages/DDS discovery, write heavily commented Python Publisher & Subscriber nodes, inspect with `ros2 topic` CLI.
2. **Phase 2 — Services, Actions, Executors & QoS (Days 3–4):** Learn synchronous Services vs. asynchronous cancellable Actions; understand why single-threaded callbacks block and fix it with `MultiThreadedExecutor` + `CallbackGroup`; run QoS `RELIABLE` vs `BEST_EFFORT` loss/latency experiments.
3. **Phase 3 — TF2 Kinematics + MuJoCo 6-DOF Arm Cell (Days 5–7):** Learn coordinate frames & transforms (`tf2_ros`), build the MuJoCo 6-DOF arm + gripper + tabletop scene, introduce `LifecycleNode` basics, and wire `/joint_states`, `/tf`, offscreen camera rendering, and Jacobian IK (`MoveToPose` action + `/arm/set_gripper` + `/arm/home`) into a multi-node ROS 2 cell.
4. **Phase 4 — DDS Architecture vs. Zenoh Integration (Days 8–9):** Understand DDS discovery/RTPS/CDR serialization, introduce `zenohd` + `rmw_zenoh_cpp` (plus an optional `bench_remote` peer container and direct Python `zenoh`), and run comparative latency/jitter benchmarks.
5. **Phase 5 — Task Execution Engine, Recovery & Telemetry (Days 10–12):** Build the explicit multi-step task state machine (`RUNNING_PICK` / `RUNNING_PLACE`), resource lock, per-step timeouts, preemption, lifecycle orchestration, fault-injection hooks (`sensor_stall`, `action_timeout`, `grasp_slip`), safe-home recovery, watchdog heartbeats, trace IDs, and latency percentile metrics.
6. **Phase 6 — Operator UI, Episode Annotator, CI/Tests & Resume (Days 13–14):** Build the FastAPI + WebSocket Operator Console & Episode Recorder/Annotator, complete `pytest` unit + integration suite, add GitHub Actions CI, write architecture docs, record demo walkthrough, and craft tailored Mind Robotics resume bullets & interview explanations.

---

## 7. Target Repository Layout

Proposed replacement for the empty scaffold:

```text
teleop-lab/
├── docker/                     # Dockerfile + docker-compose.yml (DDS & Zenoh profiles)
├── ros2_ws/
│   └── src/
│       ├── teleop_msgs/        # Custom .msg, .srv, .action definitions
│       ├── teleop_sim/         # MuJoCo 6-DOF arm MJCF/URDF + sim bridge node
│       ├── teleop_control/     # TF2 + Jacobian IK + Services + MoveToPose Action server
│       ├── teleop_runtime/     # Task Execution Engine, resource lock, fault injection, recovery
│       ├── teleop_telemetry/   # Heartbeat watchdog, trace collector, latency profiler
│       └── teleop_bench/       # DDS vs Zenoh + QoS + serialization benchmark runner
├── operator_ui/                # FastAPI + WebSocket backend & browser Operator Console
├── recorder/                   # Episode recording & operator data annotation storage
├── tests/                      # pytest unit & containerized integration tests
├── docs/
│   ├── learning/               # Per-phase conceptual study guides & command cheat sheets
│   └── superpowers/specs/      # Design specification
├── .github/workflows/          # CI workflow running lint + unit + container tests
├── pyproject.toml              # Shared Python / tooling config
└── README.md                   # Architecture overview, quickstart & demo guide
```

---

## 8. What Is Left to the Builder

The planning/implementation steps may freely decide the following without changing system intent:
- Exact MJCF/URDF geometry dimensions, joint limits, PID/position-actuator gains, and IK damping/step-size constants so pick-and-place converges reliably.
- Exact custom `.msg` / `.srv` / `.action` field names and package splitting (`teleop_msgs` as CMake/ament interface package vs. consolidated layout).
- Exact ROS 2 ↔ FastAPI bridge mechanism (running FastAPI as a ROS 2 node in a separate thread/executor or bridging via `zenoh` Python client).
- Exact visual styling of the HTML/JS Operator Console and chart widgets.
- Exact `pytest` mocking boundaries for fast unit tests vs. live container integration tests.
