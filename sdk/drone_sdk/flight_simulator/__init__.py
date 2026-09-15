"""
drone_sdk.flight_simulator
===========================
Real-Time UAV Flight Dynamics Simulator.

Provides a complete simulation environment connecting:
  - Mathematical UAV models (quad/hex/octo/fixed-wing/VTOL)
  - 6-DOF rigid body physics with RK4/symplectic integration
  - ISA atmosphere, Dryden turbulence wind, WGS-84 gravity
  - PX4-style cascaded PID flight controller
  - WebSocket telemetry server for browser-based 3D visualisation

Usage::

    from drone_sdk.flight_simulator import FlightSimulator, SimulatorConfig, UAVType
    sim = FlightSimulator(SimulatorConfig(uav_type=UAVType.QUADROTOR))
    sim.set_target_position(np.array([0, 0, -10]))
    sim.run_websocket_server()  # Open simulator/index.html in browser

Python version: 3.9+
"""
from __future__ import annotations

from .environment import (
    Environment,
    WindField,
    AtmosphereState,
    isa_atmosphere,
    wgs84_gravity,
)

from .controller import (
    PIDController,
    PIDGains,
    MultirotorController,
    MultirotorControllerConfig,
)

from .simulator import (
    FlightSimulator,
    SimulatorConfig,
    SimulatorState,
    UAVType,
)


__all__ = [
    "Environment", "WindField", "AtmosphereState",
    "isa_atmosphere", "wgs84_gravity",
    "PIDController", "PIDGains",
    "MultirotorController", "MultirotorControllerConfig",
    "FlightSimulator", "SimulatorConfig", "SimulatorState", "UAVType",
]
