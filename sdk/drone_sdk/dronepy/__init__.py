"""
DronePy: RocketPy-style Multirotor UAV Engineering Simulation & Digital Twin Layer
=================================================================================

A high-fidelity, modular 6-DOF multirotor flight simulation, uncertainty quantification,
and digital twin synchronization framework built upon the UAV Digital Twin platform.

Key Capabilities:
- Clean, declarative RocketPy-style API:
    from dronepy import Drone, Environment, Flight, MonteCarlo
- Full 6-DOF nonlinear rigid-body dynamics (NED/FRD conventions)
- Actuator dynamics: first-order lag, voltage sensitivity, motor failure injection
- Aerodynamics: analytical drag, OpenFOAM CFD polar lookup, and PINN neural surrogates
- Atmosphere & Wind: ISA 1976, constant wind, shear gradients, 1-cosine gusts, Dryden turbulence
- Mission planning & discrete event scheduling (payload drop, CG shift, motor burnout)
- Parallel Monte Carlo stochastic dispersion analysis & landing footprint visualization
- Real-time digital twin synchronization, state residual tracking, and online calibration
- Multi-tier safety gateway ensuring SIMULATION_ONLY safety by default
"""
from __future__ import annotations

# Actuators
from drone_sdk.dronepy.motors import Motor, MotorGroup
from drone_sdk.dronepy.propellers import Propeller, UIUCPropeller, BEMTPropeller

# Aerodynamics
from drone_sdk.dronepy.aerodynamics import (
    AerodynamicsModel,
    AnalyticalDrag,
    OpenFOAMAeroDatabase,
    NeuralAeroModel,
)

# Environment & Weather
from drone_sdk.dronepy.environment import Environment, Wind

# Controllers
from drone_sdk.dronepy.controllers import (
    Controller,
    CustomController,
    PIDPositionController,
)

# Core Vehicle & Results
from drone_sdk.dronepy.drone import Drone
from drone_sdk.dronepy.results import FlightResult

# Events, Failures & Mission
from drone_sdk.dronepy.events import FlightEvent, payload_release, payload_add, set_target
from drone_sdk.dronepy.failures import Scenario
from drone_sdk.dronepy.mission import Mission, Waypoint

# Flight Simulator Engine
from drone_sdk.dronepy.flight import FlightSimulatorEngine, Flight

# Plotting & Visualization
from drone_sdk.dronepy.visualization import (
    plot_trajectory,
    plot_attitude,
    plot_velocity,
    plot_motor_rpm,
    plot_power,
    plot_energy,
    plot_forces,
    plot_moments,
    plot_dashboard,
)

# Stochastic & Parametric Studies
from drone_sdk.dronepy.monte_carlo import Distribution, MonteCarlo, MonteCarloResult
from drone_sdk.dronepy.dispersion import (
    plot_landing_dispersion,
    plot_trajectory_envelopes,
)
from drone_sdk.dronepy.sweep import Sweep, SweepResult

# Digital Twin & System ID
from drone_sdk.dronepy.twin import TwinComparison, DigitalTwinSynchronizer
from drone_sdk.dronepy.calibration import DigitalTwinCalibrator, CalibrationResult
from drone_sdk.dronepy.replay import FlightReplay

# Safety & Control Interface
from drone_sdk.dronepy.safety import (
    SafetyGateway,
    SystemExecutionMode,
    SafetyViolationType,
    SafetyLimits,
    GeofenceCylinder,
    CommandIntent,
    ValidationReport,
)
from drone_sdk.dronepy.inputs import LaptopController, ControllerState

__version__ = "1.0.0"
__author__ = "Drone Digital Twin Engineering Team"

__all__ = [
    # Vehicle & Actuators
    "Drone",
    "Motor",
    "MotorGroup",
    "Propeller",
    "UIUCPropeller",
    "BEMTPropeller",
    # Aerodynamics
    "AerodynamicsModel",
    "AnalyticalDrag",
    "OpenFOAMAeroDatabase",
    "NeuralAeroModel",
    # Environment
    "Environment",
    "Wind",
    # Controllers
    "Controller",
    "CustomController",
    "PIDPositionController",
    # Mission & Events
    "Mission",
    "Waypoint",
    "FlightEvent",
    "payload_release",
    "payload_add",
    "set_target",
    "Scenario",
    # Simulation
    "Flight",
    "FlightSimulatorEngine",
    "FlightResult",
    # Stochastic & Analysis
    "Distribution",
    "MonteCarlo",
    "MonteCarloResult",
    "plot_landing_dispersion",
    "plot_trajectory_envelopes",
    "Sweep",
    "SweepResult",
    # Digital Twin
    "TwinComparison",
    "DigitalTwinSynchronizer",
    "DigitalTwinCalibrator",
    "CalibrationResult",
    "FlightReplay",
    # Safety & HMI
    "SafetyGateway",
    "SystemExecutionMode",
    "SafetyViolationType",
    "SafetyLimits",
    "GeofenceCylinder",
    "CommandIntent",
    "ValidationReport",
    "LaptopController",
    "ControllerState",
    # Visualizations
    "plot_trajectory",
    "plot_attitude",
    "plot_velocity",
    "plot_motor_rpm",
    "plot_power",
    "plot_energy",
    "plot_forces",
    "plot_moments",
    "plot_dashboard",
]
