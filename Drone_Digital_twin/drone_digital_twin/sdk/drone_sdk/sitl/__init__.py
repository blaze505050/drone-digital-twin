"""
drone_sdk.sitl
==============
PX4 SITL Integration — Module 6 of the UAV Digital Twin Platform.

Provides configuration, process management, and health monitoring for
PX4 Software-In-The-Loop simulation instances.  Supports both Docker
(no native build required) and native (PX4 source checkout) launch modes.

Architecture::

    SITLConfig ──► SITLLauncher ──► PX4 SITL process (Docker / native)
                                          │ MAVLink UDP
                                          ▼
                                    MAVLinkSource ──► TelemetryEngine
                                                          │
                                                          ▼
                                                      StateStore ──► Dashboard

Multi-vehicle::

    FleetConfig(n=3) ──► SwarmLauncher ──► [PX4#0, PX4#1, PX4#2]
                                                │
                                    [drone_0, drone_1, drone_2] StateStores

Quick-start (single drone)::

    from drone_sdk.sitl import SITLConfig, SITLLauncher, VehicleType
    from drone_sdk.state_manager import StateStore
    from drone_sdk.telemetry_engine import TelemetryEngine

    cfg      = SITLConfig(vehicle=VehicleType.IRIS)
    launcher = SITLLauncher(cfg)
    result   = launcher.start()

    if result.success:
        StateStore.create("drone_0")
        source = launcher.get_mavlink_source("drone_0")
        source.connect()
        source.configure_default_streams(rate_hz=50)

        engine = TelemetryEngine()
        engine.add_source(source)
        engine.start()
    else:
        print("SITL failed:", result.error_message)

Quick-start (3-drone swarm)::

    from drone_sdk.sitl import FleetConfig, SwarmLauncher

    fleet  = FleetConfig(n_vehicles=3, home=HomePosition.bengaluru_hal())
    swarm  = SwarmLauncher(fleet.configs)
    results = swarm.start()
    print(swarm.status())
"""

from .config import (
    FleetConfig,
    HomePosition,
    SimBackend,
    SITLConfig,
    SITLPortConfig,
    VehicleType,
    WorldType,
)
from .launcher import (
    LaunchMode,
    LaunchResult,
    ProcessState,
    SITLLauncher,
    SwarmLauncher,
    PX4_DOCKER_IMAGE,
    PX4_READY_MARKER,
)

__version__ = "1.0.0"

__all__ = [
    # Config
    "SITLConfig",
    "SITLPortConfig",
    "FleetConfig",
    "HomePosition",
    "VehicleType",
    "SimBackend",
    "WorldType",
    "LaunchMode",
    # Launcher
    "SITLLauncher",
    "SwarmLauncher",
    "LaunchResult",
    "ProcessState",
    "PX4_DOCKER_IMAGE",
    "PX4_READY_MARKER",
]
