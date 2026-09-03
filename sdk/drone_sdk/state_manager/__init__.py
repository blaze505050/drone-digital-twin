"""
drone_sdk.state_manager
=======================
Digital Twin State Manager — Module 1 of the UAV Digital Twin Platform.

This module is the **single source of truth** for all drone state across the
entire platform.  Every other module reads and writes through this API.

Public API
----------
Core types::

    DroneStateVector    # Complete drone state at one instant
    DroneStateUpdate    # Partial update from one data source
    VehicleConfig       # Per-vehicle configuration and bounds

Enumerations::

    FlightMode          # PX4-aligned flight modes
    ArmingState         # Arming lifecycle
    HealthStatus        # NOMINAL | DEGRADED | CRITICAL | NO_DATA
    DataSource          # mavlink | ros2 | sitl | manual | replay

State store::

    StateStore          # Thread-safe singleton store per vehicle
    StateFactory        # Construct states from various sources

Validation::

    PhysicsValidator    # Physical bounds enforcement
    ValidationResult    # Outcome of a validation pass

Health::

    HealthMonitor       # Update-rate and data-freshness tracking
    HealthMetrics       # Immutable metrics snapshot

Events::

    EventBus            # Pub/sub for state transitions
    EventType           # All event categories
    Event               # Event payload

Exceptions::

    StateMgrError           # Base exception
    StateValidationError    # Physics bounds violation
    StateStoreError         # Store lifecycle error
    DuplicateStoreError     # Vehicle ID collision
    StoreNotFoundError      # Unknown vehicle ID
    StaleStateError         # Data age exceeded threshold
    EventBusError           # Subscription / publication error
    FactoryError            # State construction failure

Quick-start
-----------
::

    from drone_sdk.state_manager import StateStore, StateFactory, EventType

    # Create a store for one vehicle (call once at startup)
    store = StateStore.create("drone_0")

    # Write state (from any thread)
    state = StateFactory.create_initial("drone_0")
    store.update(state)

    # Read the latest snapshot (any thread, very fast)
    latest = store.get_latest()

    # Subscribe to events
    def on_state(event):
        print("Updated:", event.data)

    store.subscribe(EventType.STATE_UPDATED, on_state)

    # Analyse history as numpy arrays
    arrays = store.get_history_numpy(["x", "y", "z", "timestamp_wall"])

    # Tear down cleanly (tests / hot-reload)
    StateStore.destroy("drone_0")

Compatibility
-------------
Python 3.9+.  No external dependencies beyond numpy (already in requirements).
"""

from .event_bus import Event, EventBus, EventCallback, EventType
from .exceptions import (
    DuplicateStoreError,
    EventBusError,
    FactoryError,
    StateMgrError,
    StateStoreError,
    StateValidationError,
    StaleStateError,
    StoreNotFoundError,
)
from .factory import StateFactory
from .health import HealthMetrics, HealthMonitor
from .schema import (
    ArmingState,
    DataSource,
    DroneStateUpdate,
    DroneStateVector,
    FlightMode,
    HealthStatus,
    VehicleConfig,
)
from .store import StateStore
from .validator import PhysicsValidator, ValidationResult

__version__ = "1.0.0"
__author__  = "UAV Digital Twin Platform"

__all__ = [
    # Schema
    "DroneStateVector",
    "DroneStateUpdate",
    "VehicleConfig",
    # Enums
    "FlightMode",
    "ArmingState",
    "HealthStatus",
    "DataSource",
    # Store
    "StateStore",
    "StateFactory",
    # Validation
    "PhysicsValidator",
    "ValidationResult",
    # Health
    "HealthMonitor",
    "HealthMetrics",
    # Events
    "EventBus",
    "EventType",
    "Event",
    "EventCallback",
    # Exceptions
    "StateMgrError",
    "StateValidationError",
    "StateStoreError",
    "DuplicateStoreError",
    "StoreNotFoundError",
    "StaleStateError",
    "EventBusError",
    "FactoryError",
]
