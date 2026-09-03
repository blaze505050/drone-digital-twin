"""
telemetry_engine.source
=======================
TelemetrySource — the abstract base class that every data source must implement.

Architecture
------------
The TelemetryEngine is source-agnostic.  Any object that inherits
TelemetrySource and implements its three abstract methods can be plugged into
the engine:

    class MyMAVLinkSource(TelemetrySource):
        def connect(self) -> None:   ...
        def poll(self)   -> list[DroneStateUpdate]:   ...
        def disconnect(self) -> None:   ...

Built-in sources (implemented in later modules):

    MAVLinkSource   — Module 3: reads PX4 / real-hardware MAVLink UDP stream
    ROS2Source      — Module 4: subscribes to ROS2 topics
    SITLSource      — Module 6: reads PX4 SITL via MAVLink
    ReplaySource    — offline HDF5 log replay

This file also defines SourcePriority, SourceStatus, and the SourceInfo
metadata snapshot.
"""
from __future__ import annotations

import abc
import time
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

from drone_sdk.state_manager import DataSource, DroneStateUpdate


# ── Source priority ───────────────────────────────────────────────────────────

class SourcePriority(IntEnum):
    """Priority for source arbitration.

    When multiple sources provide data for the same vehicle, the arbiter
    keeps the one with the lowest numeric value (HARDWARE wins over SIMULATION).
    """
    HARDWARE    = 0   # Real Pixhawk / companion computer
    SITL        = 1   # PX4 SITL via MAVLink
    ROS2        = 2   # ROS2 bridge (may duplicate SITL)
    SIMULATION  = 3   # Internal Python physics engine
    REPLAY      = 4   # Offline log replay
    MANUAL      = 5   # Direct Python API / test injection


# ── Source status ─────────────────────────────────────────────────────────────

class SourceStatus(IntEnum):
    """Lifecycle state of a TelemetrySource."""
    DISCONNECTED = 0
    CONNECTING   = 1
    CONNECTED    = 2
    ERROR        = 3
    PAUSED       = 4


# ── Source metadata ───────────────────────────────────────────────────────────

@dataclass
class SourceInfo:
    """Immutable metrics snapshot for one source."""
    source_id:       str
    vehicle_id:      str
    data_source:     DataSource
    priority:        SourcePriority
    status:          SourceStatus
    total_packets:   int      = 0
    total_dropped:   int      = 0
    total_errors:    int      = 0
    update_rate_hz:  float    = 0.0
    last_packet_mono: Optional[float] = None

    @property
    def drop_rate(self) -> float:
        """Fraction of packets dropped (0.0–1.0)."""
        total = self.total_packets + self.total_dropped
        return self.total_dropped / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "source_id":        self.source_id,
            "vehicle_id":       self.vehicle_id,
            "data_source":      self.data_source.value,
            "priority":         int(self.priority),
            "status":           self.status.name,
            "total_packets":    self.total_packets,
            "total_dropped":    self.total_dropped,
            "total_errors":     self.total_errors,
            "update_rate_hz":   round(self.update_rate_hz, 2),
            "drop_rate":        round(self.drop_rate, 4),
        }


# ── Abstract base ─────────────────────────────────────────────────────────────

class TelemetrySource(abc.ABC):
    """Abstract base class for all telemetry data sources.

    Subclass and implement ``connect``, ``poll``, and ``disconnect`` to create
    a new data source that the TelemetryEngine can manage.

    Args:
        source_id:   Unique string identifier (e.g. "px4_sitl_0").
        vehicle_id:  Target StateStore vehicle ID.
        priority:    SourcePriority for arbitration.
        data_source: DataSource enum tag (stamped onto each DroneStateUpdate).

    Lifecycle::

        source.connect()         # Open socket / serial / subscribe to topic
        updates = source.poll()  # Called by engine at its tick rate
        source.disconnect()      # Clean up resources

    Thread safety
    -------------
    ``poll()`` is called from the engine's background thread.  ``connect()``
    and ``disconnect()`` are called from the calling thread.  Implementations
    must be re-entrant-safe for concurrent ``poll()`` and ``disconnect()`` calls
    (use an internal RLock if needed).
    """

    def __init__(
        self,
        source_id:   str,
        vehicle_id:  str,
        priority:    SourcePriority = SourcePriority.SIMULATION,
        data_source: DataSource     = DataSource.UNKNOWN,
    ) -> None:
        self._source_id   = source_id
        self._vehicle_id  = vehicle_id
        self._priority    = priority
        self._data_source = data_source
        self._status      = SourceStatus.DISCONNECTED
        self._lock        = threading.RLock()

        # Metrics
        self._total_packets:  int   = 0
        self._total_dropped:  int   = 0
        self._total_errors:   int   = 0
        self._rate_ema:       float = 0.0
        self._last_poll_mono: Optional[float] = None
        self._alpha: float = 0.1   # EMA smoothing

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def vehicle_id(self) -> str:
        return self._vehicle_id

    @property
    def priority(self) -> SourcePriority:
        return self._priority

    @property
    def data_source(self) -> DataSource:
        return self._data_source

    @property
    def status(self) -> SourceStatus:
        with self._lock:
            return self._status

    @property
    def is_connected(self) -> bool:
        return self.status == SourceStatus.CONNECTED

    # ── Abstract interface ────────────────────────────────────────────────────

    @abc.abstractmethod
    def connect(self) -> None:
        """Open the data source.

        Must set self._status to CONNECTED on success or ERROR on failure.
        Should be idempotent (safe to call when already CONNECTED).

        Raises:
            SourceError: If the connection cannot be established.
        """

    @abc.abstractmethod
    def poll(self) -> List[DroneStateUpdate]:
        """Retrieve all available updates since the last poll.

        Called by the engine at its configured tick rate.  Must be non-blocking;
        return an empty list if no new data is available.

        Returns:
            List of DroneStateUpdate objects (may be empty).

        Note:
            Implementations should set their own ``timestamp_wall`` and
            ``timestamp_mono`` as close to data-acquisition time as possible.
        """

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Release all resources.

        Must be safe to call even if connect() was never called or failed.
        Must set self._status to DISCONNECTED.
        """

    # ── Metric recording (called by engine) ───────────────────────────────────

    def _record_poll(self, n_packets: int, n_dropped: int = 0) -> None:
        """Update internal metrics.  Called by the engine after each poll."""
        mono = time.monotonic()
        with self._lock:
            self._total_packets += n_packets
            self._total_dropped += n_dropped
            if self._last_poll_mono is not None and n_packets > 0:
                delta = mono - self._last_poll_mono
                if delta > 0:
                    inst_rate = n_packets / delta
                    self._rate_ema = (
                        self._alpha * inst_rate
                        + (1 - self._alpha) * self._rate_ema
                    )
            if n_packets > 0:
                self._last_poll_mono = mono

    def _record_error(self) -> None:
        """Increment error counter."""
        with self._lock:
            self._total_errors += 1

    def get_info(self) -> SourceInfo:
        """Return an immutable metrics snapshot."""
        with self._lock:
            return SourceInfo(
                source_id        = self._source_id,
                vehicle_id       = self._vehicle_id,
                data_source      = self._data_source,
                priority         = self._priority,
                status           = self._status,
                total_packets    = self._total_packets,
                total_dropped    = self._total_dropped,
                total_errors     = self._total_errors,
                update_rate_hz   = self._rate_ema,
                last_packet_mono = self._last_poll_mono,
            )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"{self.__class__.__name__}("
            f"id={self._source_id!r}, "
            f"vehicle={self._vehicle_id!r}, "
            f"priority={self._priority.name}, "
            f"status={self._status.name})"
        )


# ── Built-in null source (for testing) ───────────────────────────────────────

class NullSource(TelemetrySource):
    """A no-op source that never produces data.  Useful for testing."""

    def connect(self) -> None:
        with self._lock:
            self._status = SourceStatus.CONNECTED

    def poll(self) -> List[DroneStateUpdate]:
        return []

    def disconnect(self) -> None:
        with self._lock:
            self._status = SourceStatus.DISCONNECTED


# ── Callable source (for testing / quick integration) ─────────────────────────

class CallableSource(TelemetrySource):
    """A source that calls a user-supplied function on each poll.

    Useful for injecting test data, demo loops, or bridging existing Python
    code without subclassing.

    Args:
        poll_fn: Callable() → List[DroneStateUpdate].  Called on every poll.

    Example::

        def my_sim_poll() -> list[DroneStateUpdate]:
            upd = DroneStateUpdate("drone_0", DataSource.MANUAL)
            upd.position = np.array([t, 0, -10])
            return [upd]

        source = CallableSource(
            "sim_source", "drone_0",
            priority=SourcePriority.SIMULATION,
            poll_fn=my_sim_poll,
        )
        engine.add_source(source)
    """

    def __init__(
        self,
        source_id:   str,
        vehicle_id:  str,
        poll_fn,                         # Callable[[], List[DroneStateUpdate]]
        priority:    SourcePriority = SourcePriority.SIMULATION,
        data_source: DataSource     = DataSource.MANUAL,
    ) -> None:
        super().__init__(source_id, vehicle_id, priority, data_source)
        self._poll_fn = poll_fn

    def connect(self) -> None:
        with self._lock:
            self._status = SourceStatus.CONNECTED

    def poll(self) -> List[DroneStateUpdate]:
        try:
            return self._poll_fn()
        except Exception:  # noqa: BLE001
            self._record_error()
            return []

    def disconnect(self) -> None:
        with self._lock:
            self._status = SourceStatus.DISCONNECTED
