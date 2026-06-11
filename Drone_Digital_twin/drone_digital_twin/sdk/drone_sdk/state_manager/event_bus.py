"""
state_manager.event_bus
=======================
Thread-safe publish-subscribe event bus for drone state notifications.

Design
------
* Subscribers register a Python callable plus an optional subscriber_id.
* The bus is re-entrant-lock protected — publishing from within a callback is
  safe (will not deadlock).
* Callbacks are invoked **synchronously** from the publishing thread, after
  all internal locks have been released.  Slow callbacks will delay subsequent
  state updates; keep them short or delegate to a worker thread.
* Subscriber IDs allow targeted unsubscription.  If no ID is given, one is
  generated automatically.

Thread safety
-------------
The subscriber registry is guarded by an RLock.  Snapshot-copy-before-call
ensures that subscriptions added or removed during delivery do not affect the
current delivery batch.
"""
from __future__ import annotations

import logging
import threading
import uuid
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

from .exceptions import EventBusError

logger = logging.getLogger(__name__)


# ── Event taxonomy ────────────────────────────────────────────────────────────

class EventType(str, Enum):
    """All event types emitted by the StateStore.

    Inherits from str for easy serialisation and comparison.
    """
    # State lifecycle
    STATE_UPDATED      = "state_updated"       # Every successful update
    STATE_PARTIAL      = "state_partial"       # Partial update merged
    STATE_STALE        = "state_stale"         # Data age exceeded threshold
    STATE_INVALID      = "state_invalid"       # Validation failed

    # Arming transitions
    VEHICLE_ARMED      = "vehicle_armed"
    VEHICLE_DISARMED   = "vehicle_disarmed"

    # Flight mode transitions
    FLIGHT_MODE_CHANGED = "flight_mode_changed"

    # Health transitions
    HEALTH_DEGRADED    = "health_degraded"     # Score dropped below min threshold
    HEALTH_CRITICAL    = "health_critical"     # Score dropped below critical threshold
    HEALTH_RECOVERED   = "health_recovered"    # Score rose above min threshold

    # Store lifecycle
    STORE_RESET        = "store_reset"         # History cleared
    STORE_SHUTDOWN     = "store_shutdown"      # Store shutting down


# ── Event dataclass ───────────────────────────────────────────────────────────

class Event:
    """Lightweight event object passed to subscriber callbacks.

    Attributes:
        event_type:  Which EventType fired.
        vehicle_id:  Vehicle this event belongs to.
        data:        Optional payload (type depends on event_type — see below).
        timestamp:   Monotonic time when the event was created.

    Payload conventions by EventType::

        STATE_UPDATED       → data = DroneStateVector (the new state)
        STATE_PARTIAL       → data = DroneStateUpdate (the merged update)
        STATE_STALE         → data = float (age in ms)
        STATE_INVALID       → data = list[str] (validation issues)
        VEHICLE_ARMED       → data = None
        VEHICLE_DISARMED    → data = None
        FLIGHT_MODE_CHANGED → data = (old: FlightMode, new: FlightMode)
        HEALTH_*            → data = float (current health_score)
        STORE_RESET         → data = None
        STORE_SHUTDOWN      → data = None
    """

    __slots__ = ("event_type", "vehicle_id", "data", "timestamp")

    def __init__(
        self,
        event_type: EventType,
        vehicle_id: str,
        data: object = None,
        timestamp: Optional[float] = None,
    ) -> None:
        import time
        self.event_type  = event_type
        self.vehicle_id  = vehicle_id
        self.data        = data
        self.timestamp   = timestamp if timestamp is not None else time.monotonic()

    def __repr__(self) -> str:  # noqa: D105
        return f"Event({self.event_type.value!r}, vehicle={self.vehicle_id!r})"


# Type alias for callback signature
EventCallback = Callable[[Event], None]

# Internal registry entry: (callback, set_of_event_types_it_listens_to)
_RegistryEntry = Tuple[EventCallback, Set[EventType]]


# ── EventBus ──────────────────────────────────────────────────────────────────

class EventBus:
    """Thread-safe pub/sub bus for one vehicle's state events.

    Typically accessed through ``StateStore.event_bus`` rather than
    instantiated directly.

    Example::

        bus = EventBus(vehicle_id="drone_0")

        def on_update(event: Event) -> None:
            print("State updated:", event.data)

        sub_id = bus.subscribe(EventType.STATE_UPDATED, on_update)
        # ... later ...
        bus.unsubscribe(sub_id)
    """

    def __init__(self, vehicle_id: str) -> None:
        self._vehicle_id = vehicle_id
        self._lock: threading.RLock = threading.RLock()
        # subscriber_id → (callback, event_type_set)
        self._registry: Dict[str, _RegistryEntry] = {}

    # ── Subscription management ───────────────────────────────────────────────

    def subscribe(
        self,
        event_type: EventType | list[EventType],
        callback: EventCallback,
        subscriber_id: Optional[str] = None,
    ) -> str:
        """Register a callback for one or more event types.

        Args:
            event_type:     A single EventType or a list of EventTypes.
            callback:       Callable that accepts a single Event argument.
            subscriber_id:  Optional stable ID; generated if omitted.

        Returns:
            The subscriber_id (use to unsubscribe later).

        Raises:
            EventBusError: If subscriber_id is already registered.
        """
        if not callable(callback):
            raise EventBusError(f"callback must be callable, got {type(callback)!r}")

        if isinstance(event_type, EventType):
            event_types: Set[EventType] = {event_type}
        else:
            event_types = set(event_type)

        if subscriber_id is None:
            subscriber_id = str(uuid.uuid4())

        with self._lock:
            if subscriber_id in self._registry:
                raise EventBusError(
                    f"Subscriber '{subscriber_id}' is already registered on "
                    f"vehicle '{self._vehicle_id}'."
                )
            self._registry[subscriber_id] = (callback, event_types)

        logger.debug(
            "EventBus[%s]: subscribed %s for %s",
            self._vehicle_id,
            subscriber_id,
            [e.value for e in event_types],
        )
        return subscriber_id

    def subscribe_all(
        self,
        callback: EventCallback,
        subscriber_id: Optional[str] = None,
    ) -> str:
        """Register a callback for every EventType.

        Convenience wrapper around subscribe() with all EventTypes.

        Returns:
            The subscriber_id.
        """
        return self.subscribe(list(EventType), callback, subscriber_id)

    def unsubscribe(self, subscriber_id: str) -> bool:
        """Remove a subscriber.

        Args:
            subscriber_id: ID returned by subscribe().

        Returns:
            True if the subscriber existed and was removed; False otherwise.
        """
        with self._lock:
            existed = subscriber_id in self._registry
            if existed:
                del self._registry[subscriber_id]
                logger.debug(
                    "EventBus[%s]: unsubscribed %s",
                    self._vehicle_id,
                    subscriber_id,
                )
        return existed

    def unsubscribe_all(self) -> int:
        """Remove all subscribers.

        Returns:
            Number of subscribers that were removed.
        """
        with self._lock:
            count = len(self._registry)
            self._registry.clear()
        logger.debug("EventBus[%s]: cleared all %d subscribers", self._vehicle_id, count)
        return count

    @property
    def subscriber_count(self) -> int:
        """Current number of active subscribers."""
        with self._lock:
            return len(self._registry)

    @property
    def subscriber_ids(self) -> list[str]:
        """Snapshot of all current subscriber IDs."""
        with self._lock:
            return list(self._registry.keys())

    # ── Publishing ────────────────────────────────────────────────────────────

    def publish(self, event: Event) -> int:
        """Deliver an event to all matching subscribers.

        Callbacks are called synchronously, in registration order, from the
        calling thread.  All internal locks are released before invoking
        callbacks to prevent deadlocks.

        Args:
            event: The Event to deliver.

        Returns:
            Number of subscribers that received the event.
        """
        # Snapshot the registry under lock, then call outside the lock.
        with self._lock:
            snapshot: List[_RegistryEntry] = [
                entry for entry in self._registry.values()
                if event.event_type in entry[1]
            ]

        delivered = 0
        for callback, _ in snapshot:
            try:
                callback(event)
                delivered += 1
            except Exception:  # noqa: BLE001
                logger.exception(
                    "EventBus[%s]: exception in subscriber callback for %s",
                    self._vehicle_id,
                    event.event_type.value,
                )
        return delivered

    def emit(
        self,
        event_type: EventType,
        data: object = None,
    ) -> int:
        """Convenience wrapper: create and publish an Event.

        Args:
            event_type: EventType to fire.
            data:       Optional payload.

        Returns:
            Number of subscribers notified.
        """
        return self.publish(Event(event_type, self._vehicle_id, data))

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return a snapshot of bus statistics for diagnostics."""
        with self._lock:
            per_type: Dict[str, int] = {}
            for _, (_, event_types) in self._registry.items():
                for et in event_types:
                    per_type[et.value] = per_type.get(et.value, 0) + 1
        return {
            "vehicle_id":        self._vehicle_id,
            "subscriber_count":  self.subscriber_count,
            "subscribers_by_event": per_type,
        }

    def __repr__(self) -> str:  # noqa: D105
        return f"EventBus(vehicle_id={self._vehicle_id!r}, subscribers={self.subscriber_count})"
