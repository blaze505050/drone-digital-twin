# DronePy Architecture & Engineering Design Specification

**RocketPy-Style Flight Mechanics & Stochastic Engineering Layer for the UAV Digital Twin Platform**

---

## 1. Executive Overview

The goal of **DronePy** is to provide an idiomatic, notebook-first, high-precision flight dynamics and trajectory simulation framework for multirotor and hybrid UAVs—analogous to what [RocketPy](https://github.com/RocketPy-Team/RocketPy) represents for high-power rocketry.

DronePy does **not** replace the existing UAV Digital Twin platform; rather, it introduces a clean, unified object-oriented Python layer (`dronepy`) directly on top of the repository's 31 modular subsystems. It enables rapid vehicle prototyping, analytical and surrogate aerodynamics, stochastic Monte Carlo trajectory envelopes, landing dispersion heatmaps, actuator failure injection, and hardware-in-the-loop (HIL) digital-twin synchronization with zero conceptual duplication.

---

## 2. Current Architecture vs. DronePy Proposed Architecture

### Current Subsystem Architecture
```
                                ┌──────────────────────────────────────┐
                                │       StateStore (Thread-Safe)       │
                                │         [DroneStateVector]           │
                                └──────────────────┬───────────────────┘
                                                   │
         ┌──────────────────┬──────────────────────┼──────────────────────┬──────────────────┐
         │                  │                      │                      │                  │
┌────────▼────────┐┌────────▼────────┐    ┌────────▼────────┐    ┌────────▼────────┐┌────────▼────────┐
│ MAVLink/ROS2    ││ 6-DOF Dynamics  │    │ 15-State MEKF   │    │ Battery / Health││ Mission Planner  │
│ Bridges         ││ (RigidBodySim)  │    │ State Estimator │    │ Digital Twins   ││ & Safety Gateway │
└─────────────────┘└─────────────────┘    └─────────────────┘    └─────────────────┘└─────────────────┘
```

### DronePy Layered Architecture
```
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   USER / JUPYTER NOTEBOOK                                   │
 │       drone = Drone.quadcopter(...)                                                         │
 │       flight = drone.simulate(environment=env, mission=mission)                             │
 │       flight.plot_trajectory(); flight.plot_energy(); mc = MonteCarlo(...).run()            │
 └──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                │
 ┌──────────────────────────────────────────────▼──────────────────────────────────────────────┐
 │                                     DRONEPY API LAYER                                       │
 │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐  │
 │  │     Drone     │ │  Environment  │ │ ActuatorGroup │ │ Aerodynamics  │ │    Mission    │  │
 │  │   (Vehicle)   │ │  (Atmosphere) │ │ (Motors/Props)│ │  (AeroSource) │ │   & Events    │  │
 │  └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └───────┬───────┘  │
 └──────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼──────────┘
            │                 │                 │                 │                 │
 ┌──────────▼─────────────────▼─────────────────▼─────────────────▼─────────────────▼──────────┐
 │                                AUTHORITATIVE 6-DOF CORE                                     │
 │                          Unified Dynamics Solver (ODE / RK4)                                │
 │                                   (RigidBodySimulator)                                      │
 └──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     │                                                     │
 ┌───────────────────▼───────────────────┐             ┌───────────────────▼───────────────────┐
 │       FLIGHT RESULTS & ANALYSIS       │             │       DIGITAL TWIN & HARDWARE         │
 │  - FlightResult (time-series, Pandas) │             │  - StateStore ingestion               │
 │  - MonteCarlo dispersion & envelopes  │             │  - Reality vs Twin comparison         │
 │  - Parameter sweep engine             │             │  - MAVLink / SITL command sink        │
 │  - Matplotlib / Plotly visualization  │             │  - 3D Visualizer streaming            │
 └───────────────────────────────────────┘             └───────────────────────────────────────┘
```

---

## 3. Modules to be Reused vs. Extended

| Functional Domain | Existing Module in Repository | Status in DronePy | Role in DronePy Architecture |
| :--- | :--- | :--- | :--- |
| **Rigid Body 6-DOF** | `sdk/drone_sdk/math_models/rigid_body.py` | **Reused directly** | Newton-Euler translational and rotational equations, quaternion kinematics, RK4/Symplectic Euler integration. |
| **Frame Transforms** | `sdk/drone_sdk/math_models/frames.py` & `contracts/coordinates.py` | **Reused directly** | NED, Body FRD, quaternion Hamilton operations, DCM conversions. |
| **Multirotor Physics** | `sdk/drone_sdk/math_models/quadrotor.py`, `hexarotor.py`, `octorotor.py` | **Reused & Wrapped** | Motor mixing matrices, ground effect (Cheeseman-Bennett), blade flapping, gyroscopic precession. |
| **Atmosphere & Wind** | `sdk/drone_sdk/flight_simulator/environment.py` | **Extended** | ISA 1976 atmosphere, Dryden turbulence, WGS-84 gravity. Extended with customizable Wind profiles and gusts. |
| **Propulsion & BEMT** | `sdk/drone_sdk/physics/actuators.py` & `cad_engine/bemt.py` | **Reused & Wrapped** | UIUC wind tunnel polars, Blade Element Momentum Theory, brushless DC motor electro-mechanical equations. |
| **Aero Models** | `sdk/drone_sdk/openfoam_bridge/` & `pinn_engine/` | **Modularized** | Analytical drag base class with interchangeable `AeroDatabase` and `PINN` surrogate providers. |
| **Battery Model** | `sdk/drone_sdk/battery_twin/pack_model.py` | **Reused & Coupled** | Thevenin 1-RC equivalent circuit model (OCV, polarization, terminal voltage, Wh integration). |
| **Mission Planning** | `sdk/drone_sdk/mission_planner/` | **Reused & Wrapped** | `Mission`, `MissionItem`, geofence validation, waypoint interpolation. |
| **State Authority** | `sdk/drone_sdk/state_manager/schema.py` | **Canonical Authority** | `DroneStateVector` is the single source of truth across DronePy, MAVLink, and 3D Visualizer. |
| **System Identification**| `sdk/drone_sdk/identification/identifier.py` & `digital_twin_core/recalibrator.py` | **Reused & Wrapped** | Bounded parameter estimation (mass, drag, thrust coefficients) from flight data. |
| **Residual Monitoring** | `sdk/drone_sdk/digital_twin_core/residual_monitor.py` | **Reused directly** | Real vs Twin residual tracking, divergence detection, health scoring. |
| **Flight Control & Sinks**| `sdk/drone_sdk/flight_simulator/controller.py` & `mission_planner/sinks.py` | **Extended** | Cascaded PID (Position $\to$ Velocity $\to$ Attitude $\to$ Rate), MAVLink and Gazebo command sinks. |
| **Safety Gateway** | `sdk/drone_sdk/safety/gateway.py` | **Reused directly** | Arming state, flight envelope limits, geofence, heartbeat watchdog. |

---

## 4. Authoritative State Representation & Data Flow

### The Single State Authority Principle
To prevent divergence across the notebook, 3D visualizer, and flight controller, **every module consumes or produces the canonical `DroneStateVector`**:

```
                  ┌───────────────────────────────┐
                  │    DroneStateVector (NED)     │
                  ├───────────────────────────────┤
                  │ pos_ned: [x, y, z]            │
                  │ vel_ned: [vx, vy, vz]         │
                  │ accel_body: [ax, ay, az]      │
                  │ quat: [q0, q1, q2, q3]        │
                  │ omega_body: [p, q, r]         │
                  │ motor_rpms: [Ω1, Ω2, ...]     │
                  │ battery: [V, I, SOC, Wh]      │
                  │ flight_mode, arming_state     │
                  └───────────────┬───────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
         ▼                        ▼                        ▼
    DronePy 6-DOF            3D Web Visualizer        MAVLink Telemetry
    Flight Analysis             (WebSockets)             (PX4 / QGC)
```

---

## 5. Coordinate Frame Conventions

DronePy strictly adheres to the established repository standard defined in `contracts/coordinates.py`:

1. **Inertial Navigation Frame (NED)**:
   - $+X$: Geodetic True North
   - $+Y$: Geodetic East
   - $+Z$: Downwards towards Earth center
   - Altitude above ground: $h = -Z$
   - Nominal gravity vector: $\mathbf{g}_{\text{NED}} = [0.0, 0.0, +9.80665] \text{ m/s}^2$

2. **Body-Fixed Frame (FRD)**:
   - $+X_b$: Longitudinal forward axis (nose)
   - $+Y_b$: Starboard lateral axis (right wing/arm)
   - $+Z_b$: Ventral normal axis (downwards through belly)
   - Multirotor hover thrust acts in the **$-Z_b$** direction (upwards).

3. **Attitude Quaternion**:
   - Hamilton convention: $\mathbf{q} = [q_w, q_x, q_y, q_z]$ where $q_w$ is the scalar component.
   - Vector transformation from body to inertial NED: $\mathbf{v}_{\text{NED}} = \mathbf{R}(\mathbf{q}) \mathbf{v}_{\text{body}}$.
   - Inverse transformation from NED to body: $\mathbf{v}_{\text{body}} = \mathbf{R}(\mathbf{q})^T \mathbf{v}_{\text{NED}}$.

---

## 6. Simulation Time Architecture

DronePy supports two complementary operational time paradigms:

1. **High-Speed Numerical Integration (ODE Mode)**:
   - Asynchronous, vectorized stepping for batch analysis, parameter sweeps, and Monte Carlo iterations.
   - Computes thousands of simulated seconds per real-time second.
   - Supports fixed-step Runge-Kutta 4th order (RK4) and symplectic semi-implicit Euler.

2. **Real-Time Wall-Clock Stepping (Digital Twin Mode)**:
   - Synchronized with real-time telemetry or hardware clocks at 50–200 Hz.
   - Steps in exact lockstep with physical drone streams (`MAVLinkSource`), continuously feeding the 15-state MEKF estimator and `TwinResidualMonitor`.

---

## 7. Real vs. Sim State Authority Model

The system operates under three explicit, mutually exclusive authority modes:

```
MODE A: PURE SIMULATION (Default)
DronePy 6-DOF Solver ──> StateStore ──> Virtual Drone / Plots

MODE B: REAL DRONE MIRROR (Hardware Authoritative)
Physical Drone Sensors ──> MAVLink ──> MEKF Estimator ──> StateStore
                                                              │
                                   Twin Prediction (Parallel) ┴──> Residual Monitor

MODE C: HARDWARE-IN-THE-LOOP / COMMAND STATION (Laptop GCS)
Laptop GCS Input ──> Safety Gateway ──> Flight Controller ──> MAVLink / Motors
                                                                 │
                                                    Real Drone + Digital Twin
```

---

## 8. DronePy Package Structure

```
sdk/drone_sdk/dronepy/
├── __init__.py           # Top-level exports: Drone, Environment, Flight, MonteCarlo, etc.
├── drone.py              # Main Drone class with factory constructors (.quadcopter, .hexacopter)
├── environment.py        # Environment, Wind, Turbulence models
├── motors.py             # Motor, MotorGroup, BrushlessMotor models
├── propellers.py         # Propeller, BEMTPropeller, UIUCPropeller models
├── aerodynamics.py       # Modular aerodynamics (AnalyticalDrag, OpenFOAMAeroDatabase, PINNAeroSurrogate)
├── dynamics.py           # 6-DOF equations of motion solver wrapper
├── flight.py             # Flight simulation engine and event manager
├── mission.py            # High-level mission wrapper around mission_planner
├── events.py             # Discrete events (payload drop, CG shift, parameter change)
├── failures.py           # Failure injector (motor loss, sensor bias, GPS spoofing)
├── controllers.py        # Cascaded PID, position, velocity, and custom user controllers
├── monte_carlo.py        # Stochastic Monte Carlo analysis engine with parallel execution
├── dispersion.py         # Landing and trajectory dispersion analyzer & heatmaps
├── sweep.py              # Multi-variable parameter sweep engine
├── results.py            # FlightResult time-series container (DataFrame, CSV, HDF5)
├── visualization.py      # Plotting suite (trajectory, attitude, power, motor loads, heatmaps)
├── twin.py               # Live Digital Twin bridge and TwinComparison
├── calibration.py        # Online parameter identification from flight logs
├── safety.py             # Safety interlocks, arming state machine, geofence
└── inputs.py             # LaptopController (keyboard, gamepad, mouse)
```

This specification establishes the foundation for building the DronePy engineering layer.
