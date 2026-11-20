"""Day 4 (Part 1): Executors & Callback Groups — why robot heartbeats freeze and how to fix them.

Run inside Docker:
  python3 /workspace/learning/executor_demo.py --mode single   # Watch heartbeats FREEZE for 1.5s!
  python3 /workspace/learning/executor_demo.py --mode multi    # Heartbeats stay steady at 0.2s!

Key Concepts:
- SingleThreadedExecutor (default inside rclpy.spin): runs ONE callback at a time.
  A slow 1.5s callback blocks EVERY timer, sensor, and cancel handler in that node!
- MultiThreadedExecutor + Callback Groups: runs callbacks from different groups on
  separate worker threads in parallel.
  * MutuallyExclusiveCallbackGroup: only 1 callback in THIS group runs at a time (thread-safe!).
  * ReentrantCallbackGroup: callbacks in this group can even overlap with themselves.
"""

import argparse
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node


class ExecutorDemoNode(Node):
    def __init__(self, use_separate_groups: bool) -> None:
        super().__init__("executor_demo")
        self.start_t = time.monotonic()
        self.beat_count = 0
        self.slow_ran = False
        self.max_gap_sec = 0.0
        self.last_beat_t = self.start_t

        # Putting fast heartbeat and slow task in TWO separate MutuallyExclusive groups
        # lets MultiThreadedExecutor run them on different threads in parallel, while
        # still preventing two copies of the same callback from racing with each other.
        heartbeat_group = MutuallyExclusiveCallbackGroup() if use_separate_groups else None
        slow_task_group = MutuallyExclusiveCallbackGroup() if use_separate_groups else None

        self.create_timer(0.2, self.on_heartbeat, callback_group=heartbeat_group)
        self.create_timer(0.5, self.on_slow_task, callback_group=slow_task_group)

    def on_heartbeat(self) -> None:
        now = time.monotonic()
        gap = now - self.last_beat_t
        self.max_gap_sec = max(self.max_gap_sec, gap)
        self.last_beat_t = now
        self.beat_count += 1
        self.get_logger().info(
            f"[HEARTBEAT #{self.beat_count:02d}] t={now - self.start_t:.2f}s (gap={gap:.2f}s)"
        )

    def on_slow_task(self) -> None:
        if self.slow_ran:
            return
        self.slow_ran = True
        elapsed = time.monotonic() - self.start_t
        self.get_logger().warning(f"[SLOW TASK] Starting 1.5s heavy compute at t={elapsed:.2f}s...")
        time.sleep(1.5)
        elapsed = time.monotonic() - self.start_t
        self.get_logger().warning(f"[SLOW TASK] Finished at t={elapsed:.2f}s!")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "multi"), required=True)
    args = parser.parse_args()

    rclpy.init()
    use_multi = args.mode == "multi"
    node = ExecutorDemoNode(use_separate_groups=use_multi)
    executor = MultiThreadedExecutor(num_threads=2) if use_multi else SingleThreadedExecutor()
    executor.add_node(node)

    # Stop automatically after 2.6 seconds so you can clearly compare both outputs.
    def stop_demo() -> None:
        node.get_logger().info(
            f"=== SUMMARY ({args.mode}) === beats={node.beat_count}, "
            f"worst_heartbeat_gap={node.max_gap_sec:.2f}s"
        )
        rclpy.shutdown()

    stop_group = MutuallyExclusiveCallbackGroup() if use_multi else None
    node.create_timer(2.6, stop_demo, callback_group=stop_group)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
