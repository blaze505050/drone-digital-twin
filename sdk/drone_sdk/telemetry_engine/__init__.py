"""
drone_sdk.telemetry_engine
==========================
Telemetry Engine — Module 2 of the UAV Digital Twin Platform.

The Telemetry Engine is the real-time data ingestion layer.  It sits between
raw data sources (MAVLink sockets, ROS2 topics, simulation loops, file
replay) and the StateStore (Module 1).

Architecture::

    Source 1 ──poll()──┐
    Source 2 ──poll()──┤──► SourceArbiter ──► StateStore ──► Dashboard
    Source N ──poll()──┘                                 └──► HDF5 log

Public API
----------
Engine::

    TelemetryEngine     # Orchestrator — manages sources, runs at N Hz
    EngineConfig        # Tick rate, arbiter thresholds, auto-store creation

Sources::

    TelemetrySource     # Abstract base — implement to create a new source
    NullSource          # No-op source (testing)
    CallableSource      # Wraps a Python callable — for demos and quick bridging
    SourcePriority      # HARDWARE < SITL < ROS2 < SIMULATION < REPLAY < MANUAL
    SourceStatus        # DISCONNECTED | CONNECTING | CONNECTED | ERROR | PAUSED
    SourceInfo          # Immutable metrics snapshot per source

Arbitration::

    SourceArbiter       # Priority-based multi-source conflict resolution

Recording::

    TelemetryRecorder   # HDF5 flight log writer (requires h5py)

Replay::

    TelemetryReplayer   # HDF5 flight log playback

Exceptions::

    TelemetryError              # Base exception
    SourceError                 # Source lifecycle
    SourceNotFoundError
    DuplicateSourceError
    EngineNotRunningError
    EngineAlreadyRunningError
    BufferOverflowError
    RecordingError
    ReplayError

Quick-start
-----------
::

    import numpy as np
    from drone_sdk.state_manager import StateStore, DataSource, DroneStateUpdate
    from drone_sdk.telemetry_engine import (
        TelemetryEngine, CallableSource, SourcePriority
    )

    # 1. Create a StateStore for the vehicle
    StateStore.create("drone_0")

    # 2. Create a data source (simulation loop)
    t = [0.0]
    def my_sim_poll():
        t[0] += 0.01
        upd = DroneStateUpdate("drone_0", DataSource.SITL)
        upd.position = np.array([t[0], 0.0, -10.0])
        return [upd]

    source = CallableSource(
        "sim", "drone_0",
        poll_fn=my_sim_poll,
        priority=SourcePriority.SIMULATION,
    )

    # 3. Build and start the engine
    engine = TelemetryEngine()
    engine.add_source(source, auto_connect=True)
    engine.start()

    import time; time.sleep(5)
    engine.stop()

    # 4. Inspect results
    store  = StateStore.get_instance("drone_0")
    latest = store.get_latest()
    print("Position:", latest.position_ned())
    print("Health:",   latest.health_status.name)
"""

from .arbiter import SourceArbiter
from .engine import EngineConfig, TelemetryEngine
from .exceptions import (
    BufferOverflowError,
    DuplicateSourceError,
    EngineAlreadyRunningError,
    EngineNotRunningError,
    RecordingError,
    ReplayError,
    SourceError,
    SourceNotFoundError,
    TelemetryError,
)
from .metrics import EngineMetrics, MetricsCollector
from .recorder import TelemetryRecorder
from .replayer import PlaybackMode, TelemetryReplayer
from .source import (
    CallableSource,
    NullSource,
    SourceInfo,
    SourcePriority,
    SourceStatus,
    TelemetrySource,
)

__version__ = "1.0.0"

__all__ = [
    # Engine
    "TelemetryEngine",
    "EngineConfig",
    # Sources
    "TelemetrySource",
    "NullSource",
    "CallableSource",
    "SourcePriority",
    "SourceStatus",
    "SourceInfo",
    # Arbiter
    "SourceArbiter",
    # Metrics
    "EngineMetrics",
    "MetricsCollector",
    # Recording
    "TelemetryRecorder",
    # Replay
    "TelemetryReplayer",
    "PlaybackMode",
    # Exceptions
    "TelemetryError",
    "SourceError",
    "SourceNotFoundError",
    "DuplicateSourceError",
    "EngineNotRunningError",
    "EngineAlreadyRunningError",
    "BufferOverflowError",
    "RecordingError",
    "ReplayError",
]
