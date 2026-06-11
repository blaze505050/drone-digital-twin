"""
telemetry_engine.exceptions
============================
Typed exception hierarchy for the Telemetry Engine module.
"""
from __future__ import annotations


class TelemetryError(Exception):
    """Base class for all telemetry engine exceptions."""


class SourceError(TelemetryError):
    """Raised for telemetry source lifecycle errors."""


class SourceNotFoundError(SourceError):
    """Raised when a named source has not been registered."""
    def __init__(self, source_id: str) -> None:
        super().__init__(f"Telemetry source '{source_id}' not registered.")
        self.source_id = source_id


class DuplicateSourceError(SourceError):
    """Raised when registering a source with an already-used ID."""
    def __init__(self, source_id: str) -> None:
        super().__init__(f"Telemetry source '{source_id}' already registered.")
        self.source_id = source_id


class EngineNotRunningError(TelemetryError):
    """Raised when an operation requires the engine to be running."""


class EngineAlreadyRunningError(TelemetryError):
    """Raised when start() is called on an already-running engine."""


class BufferOverflowError(TelemetryError):
    """Raised (internally) when a source buffer is full and must drop packets."""
    def __init__(self, source_id: str, dropped: int) -> None:
        super().__init__(
            f"Source '{source_id}' buffer overflow: {dropped} packets dropped."
        )
        self.source_id = source_id
        self.dropped   = dropped


class RecordingError(TelemetryError):
    """Raised for HDF5 recording errors."""


class ReplayError(TelemetryError):
    """Raised for telemetry replay errors."""
