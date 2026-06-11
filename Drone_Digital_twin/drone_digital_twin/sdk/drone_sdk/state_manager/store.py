"""
state_manager.store
===================
StateStore — thread-safe, singleton-per-vehicle state store.

This is the central object of the entire platform.  Every other module
(MAVLink, ROS2, Dashboard, AI/ML) interacts with drone state through a
StateStore instance.

Architecture
------------
* **Singleton per vehicle_id** — only one StateStore may exist for a given ID.
  Use ``StateStore.get_instance()`` to retrieve existing stores and
  ``StateStore.create()`` to create new ones.
* **Ring buffer history** — a ``collections.deque(maxlen=N)`` stores the last
  N states for offline analysis and ML training.
* **RLock** — a re-entrant lock guards all mutations.  Re-entrancy allows
  callbacks subscribed to state events to call read-only store methods without
  deadlocking.
* **Event bus** — state change events are fired *after* releasing the lock,
  so callback code cannot accidentally deadlock the store.
* **HealthMonitor** — tracks update rate, validation pass rate, and data age
  in a lock-free EMA.
* **PhysicsValidator** — every ``update()`` call is validated; invalid states
  are still stored but marked ``is_valid=False`` and fire a STATE_INVALID event.

Thread model
------------
The store is safe for concurrent reads and writes from multiple threads.  The
intended usage pattern is:

    Writer thread (100 Hz):  store.update(state)
    Reader thread (50 Hz):   state = store.get_latest()
    Subscriber thread:       @store.subscribe(EventType.STATE_UPDATED, cb)

Performance
-----------
``get_latest()`` acquires the RLock briefly to copy the reference.  On CPython
this is effectively a single LOAD_ATTR + INCREF under the GIL, so it is safe
to call at very high rates.  ``get_history()`` acquires the lock for the
duration of the list copy.

Python version: 3.9+
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import ClassVar, Dict, List, Optional

import numpy as np

from .event_bus import EventBus, EventCallback, EventType
from .exceptions import (
    DuplicateStoreError,
    StaleStateError,
    StoreNotFoundError,
)
from .factory import StateFactory
from .health import HealthMonitor
from .schema import (
    ArmingState,
    DroneStateUpdate,
    DroneStateVector,
    FlightMode,
    HealthStatus,
    VehicleConfig,
)
from .validator import PhysicsValidator, ValidationResult

logger = logging.getLogger(__name__)


class StateStore:
    """Thread-safe state store for one vehicle's digital twin.

    Do **not** instantiate directly — use the class-level factory methods.

    Quick-start::

        # Create once (typically at application startup)
        store = StateStore.create("drone_0")

        # Retrieve later from any module
        store = StateStore.get_instance("drone_0")

        # Write state (from MAVLink / SITL / simulation loop)
        state = StateFactory.create_initial("drone_0")
        store.update(state)

        # Read latest snapshot
        latest = store.get_latest()

        # Subscribe to events
        def on_update(event: Event) -> None:
            print("New state:", event.data)

        store.subscribe(EventType.STATE_UPDATED, on_update)

        # Analyse history
        history = store.get_history(window_seconds=10.0)
        arrays  = store.get_history_numpy(["x", "y", "z", "timestamp_wall"])

        # Tear down (tests / hot-reload)
        StateStore.destroy("drone_0")
    """

    # ── Class-level singleton registry ────────────────────────────────────────
    _instances: ClassVar[Dict[str, "StateStore"]] = {}
    _class_lock: ClassVar[threading.Lock] = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def __init__(self, vehicle_id: str, config: VehicleConfig) -> None:
        """Internal constructor — use create() or get_instance()."""
        self._vehicle_id = vehicle_id
        self._config     = config
        self._lock       = threading.RLock()
        self._shutdown   = False

        # State storage
        self._current: Optional[DroneStateVector] = None
        self._history: deque[DroneStateVector] = deque(maxlen=config.history_size)
        self._sequence: int = 0

        # Subsystems
        self._bus       = EventBus(vehicle_id)
        self._health    = HealthMonitor(config)
        self._validator = PhysicsValidator(config)

        # Transition tracking (for change-detection events)
        self._last_arming_state: Optional[ArmingState] = None
        self._last_flight_mode:  Optional[FlightMode]  = None
        self._last_health_status: Optional[HealthStatus] = None

        logger.info(
            "StateStore created for vehicle '%s' (history=%d, rate=%.0f Hz)",
            vehicle_id,
            config.history_size,
            config.expected_update_rate_hz,
        )

    # ── Singleton factory methods ─────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        vehicle_id: str,
        config: Optional[VehicleConfig] = None,
    ) -> "StateStore":
        """Create and register a new StateStore.

        Args:
            vehicle_id: Unique identifier for this vehicle.
            config:     VehicleConfig; a sensible default is created if omitted.

        Returns:
            The newly created StateStore.

        Raises:
            DuplicateStoreError: If a store for vehicle_id already exists.
        """
        with cls._class_lock:
            if vehicle_id in cls._instances:
                raise DuplicateStoreError(vehicle_id)
            cfg = config or VehicleConfig(vehicle_id=vehicle_id)
            instance = cls(vehicle_id, cfg)
            cls._instances[vehicle_id] = instance
            return instance

    @classmethod
    def get_instance(cls, vehicle_id: str) -> "StateStore":
        """Retrieve an existing StateStore.

        Args:
            vehicle_id: The vehicle identifier.

        Returns:
            The StateStore for vehicle_id.

        Raises:
            StoreNotFoundError: If no store exists for vehicle_id.
        """
        with cls._class_lock:
            if vehicle_id not in cls._instances:
                raise StoreNotFoundError(vehicle_id)
            return cls._instances[vehicle_id]

    @classmethod
    def get_or_create(
        cls,
        vehicle_id: str,
        config: Optional[VehicleConfig] = None,
    ) -> "StateStore":
        """Retrieve an existing store or create one if it does not exist.

        Useful when modules don't know the startup order.
        """
        with cls._class_lock:
            if vehicle_id not in cls._instances:
                cfg = config or VehicleConfig(vehicle_id=vehicle_id)
                instance = cls(vehicle_id, cfg)
                cls._instances[vehicle_id] = instance
                return instance
            return cls._instances[vehicle_id]

    @classmethod
    def destroy(cls, vehicle_id: str) -> bool:
        """Shutdown and deregister a StateStore.

        Args:
            vehicle_id: Vehicle to destroy.

        Returns:
            True if the store existed and was destroyed; False otherwise.
        """
        with cls._class_lock:
            store = cls._instances.pop(vehicle_id, None)
        if store is not None:
            store.shutdown()
            return True
        return False

    @classmethod
    def destroy_all(cls) -> int:
        """Shutdown and deregister all stores.  Returns count destroyed."""
        with cls._class_lock:
            ids = list(cls._instances.keys())
        count = 0
        for vid in ids:
            if cls.destroy(vid):
                count += 1
        return count

    @classmethod
    def list_vehicles(cls) -> list[str]:
        """Return a list of all currently registered vehicle IDs."""
        with cls._class_lock:
            return list(cls._instances.keys())

    @classmethod
    def exists(cls, vehicle_id: str) -> bool:
        """Return True if a store for vehicle_id is registered."""
        with cls._class_lock:
            return vehicle_id in cls._instances

    # ── Write API ─────────────────────────────────────────────────────────────

    def update(self, state: DroneStateVector) -> bool:
        """Store a new full state snapshot.

        This is the primary write method.  It:
        1. Validates the state with PhysicsValidator.
        2. Stamps the health fields (health_score, health_status, is_valid).
        3. Appends to the history ring buffer.
        4. Fires all relevant EventBus events (STATE_UPDATED and transitions).

        Args:
            state: The new state to store.  The ``sequence`` field is
                   overwritten with the store's internal counter.

        Returns:
            True if the state passed validation; False if it was stored but
            is_valid was set to False.

        Raises:
            RuntimeError: If the store has been shut down.
        """
        self._assert_not_shutdown()
        mono = time.monotonic()

        # Validate
        result: ValidationResult = self._validator.validate(state)

        # Update health monitor
        self._health.record_update(mono, is_valid=result.is_valid)
        metrics = self._health.get_metrics()

        # Stamp health, current time, and sequence onto the state.
        # timestamp_mono is refreshed to the store-receive time so that
        # get_history(window_seconds=...) filters correctly on store-write time.
        now_wall = time.time()
        stamped = state.copy_with(
            sequence       = self._sequence,
            timestamp_wall = now_wall,
            timestamp_mono = mono,
            is_valid       = result.is_valid,
            health_score   = metrics.overall_score,
            health_status  = metrics.status,
            latency_ms     = (now_wall - state.timestamp_wall) * 1_000.0,
        )

        # Detect state transitions before acquiring lock
        prev_arming  = self._last_arming_state
        prev_mode    = self._last_flight_mode
        prev_health  = self._last_health_status

        with self._lock:
            self._sequence += 1
            self._current = stamped
            self._history.append(stamped)
            self._last_arming_state  = stamped.arming_state
            self._last_flight_mode   = stamped.flight_mode
            self._last_health_status = stamped.health_status

        # Fire events AFTER releasing lock to prevent deadlocks in callbacks
        if not result.is_valid:
            self._bus.emit(EventType.STATE_INVALID, result.issues)

        self._bus.emit(EventType.STATE_UPDATED, stamped)

        # Arming transitions
        if prev_arming != stamped.arming_state:
            if stamped.arming_state == ArmingState.ARMED:
                self._bus.emit(EventType.VEHICLE_ARMED)
            elif stamped.arming_state == ArmingState.DISARMED:
                self._bus.emit(EventType.VEHICLE_DISARMED)

        # Flight mode transitions
        if prev_mode is not None and prev_mode != stamped.flight_mode:
            self._bus.emit(EventType.FLIGHT_MODE_CHANGED, (prev_mode, stamped.flight_mode))

        # Health transitions
        self._fire_health_transitions(prev_health, stamped.health_status, metrics.overall_score)

        if not result.is_valid:
            logger.debug(
                "StateStore[%s]: invalid state seq=%d — %s",
                self._vehicle_id,
                stamped.sequence,
                result.issues,
            )

        return result.is_valid

    def update_partial(self, update: DroneStateUpdate) -> bool:
        """Apply a partial DroneStateUpdate to the current state.

        If no state exists yet, creates an initial state first.

        Args:
            update: Partial update; only non-None fields are applied.

        Returns:
            True if the merged state passed validation.
        """
        self._assert_not_shutdown()

        with self._lock:
            base = self._current or StateFactory.create_initial(
                self._vehicle_id, self._config, source=update.source
            )

        merged = StateFactory.merge_update(base, update)
        result = self.update(merged)

        self._bus.emit(EventType.STATE_PARTIAL, update)
        return result

    # ── Read API ──────────────────────────────────────────────────────────────

    def get_latest(self) -> Optional[DroneStateVector]:
        """Return a snapshot of the most recent state, or None if no update yet.

        The returned object is the live state reference.  Callers that intend to
        modify the state (e.g. for what-if analysis) should call ``.copy_with()``
        first.
        """
        with self._lock:
            return self._current

    def get_latest_or_raise(self, max_age_ms: Optional[float] = None) -> DroneStateVector:
        """Return the latest state or raise if no data / data is stale.

        Args:
            max_age_ms: Staleness threshold in ms; uses config default if None.

        Raises:
            StaleStateError: If no state exists or it exceeds max_age_ms.
        """
        threshold = max_age_ms if max_age_ms is not None else self._config.max_state_age_ms
        if self._health.is_stale(threshold):
            age_ms = self._get_current_age_ms()
            raise StaleStateError(self._vehicle_id, age_ms, threshold)
        state = self.get_latest()
        if state is None:
            raise StaleStateError(self._vehicle_id, float("inf"), threshold)
        return state

    def get_history(
        self,
        *,
        n: Optional[int] = None,
        window_seconds: Optional[float] = None,
    ) -> List[DroneStateVector]:
        """Return a snapshot of recent history.

        Args:
            n:              Return the last N entries (default: all).
            window_seconds: Return entries from the last N seconds.
                            Evaluated on ``timestamp_mono``.  Overrides ``n``.

        Returns:
            List of DroneStateVector in chronological order (oldest first).
        """
        with self._lock:
            all_history = list(self._history)

        if window_seconds is not None:
            cutoff = time.monotonic() - window_seconds
            all_history = [s for s in all_history if s.timestamp_mono >= cutoff]
        elif n is not None:
            all_history = all_history[-n:]

        return all_history

    def get_history_numpy(
        self,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract history into numpy arrays, one per field.

        Useful for offline analysis, PINN training data preparation, and
        plotting.

        Args:
            fields: List of DroneStateVector attribute names.  Defaults to a
                    useful subset for trajectory analysis.

        Returns:
            Dict mapping field name → 1-D numpy array (float64 or int64).

        Example::

            arrays = store.get_history_numpy(["timestamp_wall", "x", "y", "z"])
            t = arrays["timestamp_wall"]
            traj = np.stack([arrays["x"], arrays["y"], arrays["z"]], axis=1)
        """
        default_fields = [
            "timestamp_wall", "timestamp_sim", "sequence",
            "x", "y", "z",
            "vx", "vy", "vz",
            "roll", "pitch", "yaw",
            "q0", "q1", "q2", "q3",
            "roll_rate", "pitch_rate", "yaw_rate",
            "omega1", "omega2", "omega3", "omega4",
            "battery_voltage", "battery_soc",
            "groundspeed", "altitude_agl",
            "health_score",
        ]
        target_fields = fields or default_fields

        history = self.get_history()
        if not history:
            return {f: np.array([], dtype=np.float64) for f in target_fields}

        result: Dict[str, np.ndarray] = {}
        for field_name in target_fields:
            if not hasattr(history[0], field_name):
                logger.warning(
                    "StateStore: field '%s' not found in DroneStateVector",
                    field_name,
                )
                continue
            values = [getattr(s, field_name) for s in history]
            # Attempt float conversion; fall back to int or object array
            try:
                result[field_name] = np.array(values, dtype=np.float64)
            except (TypeError, ValueError):
                result[field_name] = np.array(values)

        return result

    # ── Event subscription ────────────────────────────────────────────────────

    def subscribe(
        self,
        event_type: EventType | List[EventType],
        callback: EventCallback,
        subscriber_id: Optional[str] = None,
    ) -> str:
        """Subscribe to one or more event types on this vehicle's bus.

        Args:
            event_type:     EventType or list of EventTypes to listen for.
            callback:       Callable accepting a single Event argument.
            subscriber_id:  Optional stable ID; auto-generated if omitted.

        Returns:
            subscriber_id (use to unsubscribe).
        """
        return self._bus.subscribe(event_type, callback, subscriber_id)

    def subscribe_all(self, callback: EventCallback, subscriber_id: Optional[str] = None) -> str:
        """Subscribe to every EventType on this vehicle's bus."""
        return self._bus.subscribe_all(callback, subscriber_id)

    def unsubscribe(self, subscriber_id: str) -> bool:
        """Remove a subscriber by ID.  Returns True if it existed."""
        return self._bus.unsubscribe(subscriber_id)

    # ── Subsystem accessors ───────────────────────────────────────────────────

    @property
    def event_bus(self) -> EventBus:
        """Direct access to the underlying EventBus (for advanced use)."""
        return self._bus

    @property
    def health(self) -> HealthMonitor:
        """Direct access to the HealthMonitor."""
        return self._health

    @property
    def validator(self) -> PhysicsValidator:
        """Direct access to the PhysicsValidator."""
        return self._validator

    @property
    def config(self) -> VehicleConfig:
        """The VehicleConfig this store was created with."""
        return self._config

    @property
    def vehicle_id(self) -> str:
        """This store's vehicle identifier."""
        return self._vehicle_id

    # ── Diagnostics / status ──────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        """Return True when health_status == NOMINAL."""
        return self._health.get_status() == HealthStatus.NOMINAL

    def is_stale(self, max_age_ms: Optional[float] = None) -> bool:
        """Return True when data age exceeds the threshold."""
        return self._health.is_stale(max_age_ms)

    def get_status_summary(self) -> dict:
        """Return a JSON-serialisable status summary for monitoring dashboards."""
        metrics = self._health.get_metrics()
        current = self.get_latest()
        with self._lock:
            history_len = len(self._history)
            sequence    = self._sequence

        return {
            "vehicle_id":         self._vehicle_id,
            "sequence":           sequence,
            "history_length":     history_len,
            "history_capacity":   self._config.history_size,
            "subscriber_count":   self._bus.subscriber_count,
            "is_shutdown":        self._shutdown,
            "health": {
                "status":              metrics.status.name,
                "score":               round(metrics.overall_score, 3),
                "update_rate_hz":      round(metrics.update_rate_hz, 1),
                "expected_rate_hz":    metrics.expected_rate_hz,
                "data_age_ms":         round(metrics.data_age_ms, 1),
                "validation_pass_rate": round(metrics.validation_pass_rate, 3),
                "total_updates":       metrics.total_updates,
                "total_invalid":       metrics.total_invalid,
            },
            "current_state": current.to_dict() if current else None,
        }

    # ── Store control ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear history buffer and reset health statistics.

        Does NOT remove event subscriptions or shutdown the store.
        Fires STORE_RESET event.
        """
        self._assert_not_shutdown()
        with self._lock:
            self._history.clear()
            self._sequence = 0
            self._current  = None
            self._last_arming_state  = None
            self._last_flight_mode   = None
            self._last_health_status = None

        self._health.reset()
        self._bus.emit(EventType.STORE_RESET)
        logger.info("StateStore[%s]: reset", self._vehicle_id)

    def shutdown(self) -> None:
        """Release resources and mark the store as shut down.

        After shutdown, update() calls raise RuntimeError.
        """
        with self._lock:
            self._shutdown = True

        self._bus.emit(EventType.STORE_SHUTDOWN)
        self._bus.unsubscribe_all()
        logger.info("StateStore[%s]: shutdown complete", self._vehicle_id)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def history_length(self) -> int:
        """Current number of states in the ring buffer."""
        with self._lock:
            return len(self._history)

    @property
    def sequence(self) -> int:
        """Current monotonic sequence counter."""
        with self._lock:
            return self._sequence

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _assert_not_shutdown(self) -> None:
        if self._shutdown:
            raise RuntimeError(
                f"StateStore[{self._vehicle_id}] has been shut down and cannot "
                "accept further updates."
            )

    def _get_current_age_ms(self) -> float:
        """Return data age in ms, or inf if no data."""
        with self._lock:
            current = self._current
        if current is None:
            return float("inf")
        return (time.monotonic() - current.timestamp_mono) * 1_000.0

    def _fire_health_transitions(
        self,
        prev: Optional[HealthStatus],
        current: HealthStatus,
        score: float,
    ) -> None:
        """Fire health transition events when status changes."""
        if prev is None or prev == current:
            return
        if current == HealthStatus.CRITICAL:
            self._bus.emit(EventType.HEALTH_CRITICAL, score)
        elif current == HealthStatus.DEGRADED:
            self._bus.emit(EventType.HEALTH_DEGRADED, score)
        elif current == HealthStatus.NOMINAL and prev in (
            HealthStatus.DEGRADED, HealthStatus.CRITICAL
        ):
            self._bus.emit(EventType.HEALTH_RECOVERED, score)

    def __repr__(self) -> str:  # noqa: D105
        m = self._health.get_metrics()
        return (
            f"StateStore(vehicle_id={self._vehicle_id!r}, "
            f"seq={self._sequence}, "
            f"history={self.history_length}/{self._config.history_size}, "
            f"health={m.status.name}, "
            f"subscribers={self._bus.subscriber_count})"
        )
