"""Days 8-9: Real Transport Benchmark — CycloneDDS vs. Eclipse Zenoh (rmw_zenoh_cpp).

Measures TRUE inter-process latency using the same Linux kernel monotonic clock:
  - Process 1 (`--role sub`): listens on `/bench/ping` and records (recv_ns - sent_ns).
  - Process 2 (`--role pub`): sends N messages with `time.monotonic_ns()` packed in the first 8 bytes.

Notice: ZERO DDS-specific or Zenoh-specific code! Switching transport is purely
changing the `RMW_IMPLEMENTATION` environment variable.
"""

import argparse
import os
import struct
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import UInt8MultiArray


def make_qos(reliable: bool) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
    )


class BenchPub(Node):
    def __init__(self, count: int, payload_bytes: int, rate_hz: float, reliable: bool) -> None:
        super().__init__("rmw_bench_pub")
        self.pub = self.create_publisher(UInt8MultiArray, "/bench/ping", make_qos(reliable))
        self.count = count
        self.pad = bytes(max(0, payload_bytes - 8))
        self.sent = 0
        self.done = False
        # Brief pause so DDS/Zenoh discovery connects before message #1.
        self.start_at = time.monotonic() + 0.8
        self.timer = self.create_timer(1.0 / rate_hz, self.on_tick)

    def on_tick(self) -> None:
        if time.monotonic() < self.start_at:
            return
        if self.sent >= self.count:
            self.done = True
            return
        msg = UInt8MultiArray()
        msg.data = struct.pack("!Q", time.monotonic_ns()) + self.pad
        self.pub.publish(msg)
        self.sent += 1


class BenchSub(Node):
    def __init__(self, expected: int, warmup: int, reliable: bool) -> None:
        super().__init__("rmw_bench_sub")
        self.expected = expected
        self.warmup = warmup
        self.latencies_ms: list[float] = []
        self.total_received = 0
        self.last_rx = time.monotonic()
        self.sub = self.create_subscription(
            UInt8MultiArray, "/bench/ping", self.on_msg, make_qos(reliable)
        )

    def on_msg(self, msg: UInt8MultiArray) -> None:
        now_ns = time.monotonic_ns()
        sent_ns = struct.unpack("!Q", bytes(msg.data[:8]))[0]
        self.total_received += 1
        self.last_rx = time.monotonic()
        if self.total_received > self.warmup:
            self.latencies_ms.append((now_ns - sent_ns) / 1e6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=("pub", "sub"), required=True)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--payload-bytes", type=int, default=64)
    ap.add_argument("--rate-hz", type=float, default=100.0)
    ap.add_argument("--reliable", action="store_true")
    args = ap.parse_args()

    rclpy.init()
    if args.role == "pub":
        node = BenchPub(args.count, args.payload_bytes, args.rate_hz, args.reliable)
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.2)
    else:
        node = BenchSub(args.count, args.warmup, args.reliable)
        deadline = time.monotonic() + (args.count / args.rate_hz) + 4.0
        while rclpy.ok() and node.total_received < args.count and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            # Stop early if sender finished and went quiet for 0.8s (e.g. BEST_EFFORT drop)
            if node.total_received > 0 and (time.monotonic() - node.last_rx) > 0.8:
                break

        rmw = os.environ.get("RMW_IMPLEMENTATION", "default")
        arr = np.array(node.latencies_ms, dtype=np.float64)
        if len(arr) == 0:
            print(f"[{rmw}] ERROR: received 0 messages!")
        else:
            p50, p95, p99 = np.percentile(arr, [50, 95, 99])
            print(
                f"[{rmw:22s}] payload={args.payload_bytes:6d}B | "
                f"rx={node.total_received:3d}/{args.count} | "
                f"p50={p50:6.3f}ms | p95={p95:6.3f}ms | p99={p99:6.3f}ms | max={arr.max():6.3f}ms"
            )

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
