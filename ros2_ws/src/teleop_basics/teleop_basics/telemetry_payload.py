"""Pure-Python heartbeat payload serialization, latency, and drop-tracking utilities.

Why separate this module from `rclpy`?
--------------------------------------
In production robotics systems, keeping pure data validation, clock math, and
state tracking in framework-independent Python modules gives two big wins:
1. Unit tests run in milliseconds on any machine without booting ROS 2 or DDS.
2. The exact same tested functions run inside real ROS 2 nodes in Docker.
"""

from dataclasses import dataclass
import json
import math


class PayloadValidationError(ValueError):
    """Raised when an incoming heartbeat JSON message is malformed or incomplete.

    A subscriber callback should catch this exception and log a warning rather
    than letting an unexpected message crash the whole ROS 2 node process.
    """


@dataclass(frozen=True)
class HeartbeatMessage:
    """Validated heartbeat payload sent over `/teleop/heartbeat`."""

    node_id: str
    seq: int
    pub_ns: int
    status: str


@dataclass(frozen=True)
class SequenceUpdate:
    """Result of observing the next sequence number on a heartbeat stream."""

    gap: int
    is_out_of_order: bool
    dropped_total: int


def _is_non_negative_int(value: object) -> bool:
    # Note: in Python `bool` is a subclass of `int` (`isinstance(True, int)` is True),
    # so we explicitly exclude booleans before checking integer range.
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_publisher_config(node_id: str, rate_hz: float) -> None:
    """Validate publisher parameters before creating timers or publishing messages."""
    if not isinstance(node_id, str) or not node_id.strip():
        raise ValueError("node_id must be a non-empty string")
    if (
        isinstance(rate_hz, bool)
        or not isinstance(rate_hz, (int, float))
        or not math.isfinite(rate_hz)
        or rate_hz <= 0
    ):
        raise ValueError("rate_hz must be a finite number greater than 0")


def _validate_heartbeat_fields(
    node_id: object,
    seq: object,
    pub_ns: object,
    status: object,
) -> HeartbeatMessage:
    """Validate field types/ranges and construct an immutable `HeartbeatMessage`."""
    if not isinstance(node_id, str) or not node_id.strip():
        raise PayloadValidationError("node_id must be a non-empty string")
    if not _is_non_negative_int(seq):
        raise PayloadValidationError("seq must be a non-negative integer")
    if not _is_non_negative_int(pub_ns):
        raise PayloadValidationError("pub_ns must be a non-negative integer")
    if not isinstance(status, str) or not status.strip():
        raise PayloadValidationError("status must be a non-empty string")
    return HeartbeatMessage(node_id=node_id, seq=seq, pub_ns=pub_ns, status=status)


def build_heartbeat_payload(
    node_id: str,
    seq: int,
    pub_ns: int,
    status: str = "OK",
) -> str:
    """Serialize a heartbeat record into a compact JSON string.

    `pub_ns` should come from `time.monotonic_ns()` (never `time.time()`):
    monotonic clocks never jump backwards if the OS syncs NTP wall time.
    """
    msg = _validate_heartbeat_fields(node_id=node_id, seq=seq, pub_ns=pub_ns, status=status)
    return json.dumps(
        {
            "node_id": msg.node_id,
            "seq": msg.seq,
            "pub_ns": msg.pub_ns,
            "status": msg.status,
        },
        separators=(",", ":"),
    )


def parse_heartbeat_payload(raw: str) -> HeartbeatMessage:
    """Parse and strictly validate a raw JSON heartbeat payload string."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise PayloadValidationError(f"Invalid heartbeat JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise PayloadValidationError("Heartbeat JSON must decode to an object")

    required_keys = {"node_id", "seq", "pub_ns", "status"}
    missing = required_keys - data.keys()
    if missing:
        raise PayloadValidationError(f"Missing required heartbeat keys: {sorted(missing)}")

    return _validate_heartbeat_fields(
        node_id=data["node_id"],
        seq=data["seq"],
        pub_ns=data["pub_ns"],
        status=data["status"],
    )


def compute_latency_ms(pub_ns: int, recv_ns: int) -> tuple[float, bool]:
    """Compute one-way transport latency in milliseconds.

    Returns `(latency_ms, clock_skew)` where `clock_skew=True` guards against
    negative elapsed times if timestamps come from unsynchronized clocks.
    """
    delta_ns = recv_ns - pub_ns
    if delta_ns < 0:
        return (0.0, True)
    return (delta_ns / 1_000_000.0, False)


class SequenceTracker:
    """Tracks monotonic sequence numbers to detect dropped or late messages."""

    def __init__(self) -> None:
        self._last_seq: int | None = None
        self._dropped_total: int = 0

    @property
    def dropped_total(self) -> int:
        return self._dropped_total

    def observe(self, seq: int) -> SequenceUpdate:
        """Record an incoming sequence number and report any newly detected gap."""
        if not _is_non_negative_int(seq):
            raise PayloadValidationError("seq must be a non-negative integer")

        if self._last_seq is None:
            self._last_seq = seq
            return SequenceUpdate(gap=0, is_out_of_order=False, dropped_total=self._dropped_total)

        if seq <= self._last_seq:
            return SequenceUpdate(gap=0, is_out_of_order=True, dropped_total=self._dropped_total)

        gap = seq - self._last_seq - 1
        self._dropped_total += gap
        self._last_seq = seq
        return SequenceUpdate(gap=gap, is_out_of_order=False, dropped_total=self._dropped_total)


class MultiNodeSequenceTracker:
    """Maintains an independent `SequenceTracker` per `node_id` stream.

    Why? Many nodes can publish on `/teleop/heartbeat` concurrently. Tracking
    sequence numbers per `node_id` prevents interleaving `node_a` (seq=10) and
    `node_b` (seq=1) from triggering false drop or out-of-order alerts.
    """

    def __init__(self) -> None:
        self._trackers: dict[str, SequenceTracker] = {}

    @property
    def dropped_total(self) -> int:
        return sum(tracker.dropped_total for tracker in self._trackers.values())

    def observe(self, node_id: str, seq: int) -> SequenceUpdate:
        tracker = self._trackers.setdefault(node_id, SequenceTracker())
        return tracker.observe(seq)
