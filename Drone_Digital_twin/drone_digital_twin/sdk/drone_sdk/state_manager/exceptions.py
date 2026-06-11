"""
state_manager.exceptions
========================
Typed exception hierarchy for the Digital Twin State Manager.

All exceptions inherit from StateMgrError so callers can catch the entire
module's exceptions with a single except clause when needed.
"""
from __future__ import annotations


class StateMgrError(Exception):
    """Base class for all state manager exceptions."""


class StateValidationError(StateMgrError):
    """Raised when a DroneStateVector fails physical bounds validation.

    Attributes:
        issues: List of human-readable validation failure messages.
    """

    def __init__(self, message: str, issues: list[str] | None = None) -> None:
        super().__init__(message)
        self.issues: list[str] = issues or []

    def __str__(self) -> str:
        base = super().__str__()
        if self.issues:
            detail = "\n  ".join(self.issues)
            return f"{base}\n  {detail}"
        return base


class StateStoreError(StateMgrError):
    """Raised for StateStore configuration or lifecycle errors."""


class DuplicateStoreError(StateStoreError):
    """Raised when attempting to create a store that already exists."""

    def __init__(self, vehicle_id: str) -> None:
        super().__init__(
            f"StateStore for vehicle '{vehicle_id}' already exists. "
            "Use StateStore.get_instance() to retrieve it."
        )
        self.vehicle_id = vehicle_id


class StoreNotFoundError(StateStoreError):
    """Raised when get_instance() is called for an unknown vehicle_id."""

    def __init__(self, vehicle_id: str) -> None:
        super().__init__(
            f"No StateStore found for vehicle '{vehicle_id}'. "
            "Create one with StateStore.create() first."
        )
        self.vehicle_id = vehicle_id


class StaleStateError(StateMgrError):
    """Raised when state data exceeds the configured maximum age."""

    def __init__(self, vehicle_id: str, age_ms: float, max_age_ms: float) -> None:
        super().__init__(
            f"State for '{vehicle_id}' is stale: {age_ms:.1f} ms old "
            f"(max allowed: {max_age_ms:.1f} ms)."
        )
        self.vehicle_id = vehicle_id
        self.age_ms = age_ms
        self.max_age_ms = max_age_ms


class EventBusError(StateMgrError):
    """Raised for EventBus subscription or publication errors."""


class FactoryError(StateMgrError):
    """Raised when StateFactory cannot construct a valid state."""
