# UAV Digital Twin & DronePy Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" />
  <img src="https://img.shields.io/badge/Tests-1119%20passing-brightgreen" />
  <img src="https://img.shields.io/badge/Modules-32-orange" />
  <img src="https://img.shields.io/badge/DronePy-RocketPy--Style%20UAV%20Sim-purple" />
  <img src="https://img.shields.io/badge/Status-Production%20Ready%20Simulation-brightgreen" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

<p align="center">
  <a href="https://blaze505050.github.io/drone-digital-twin/"><img src="https://img.shields.io/badge/🎮%20Live%203D%20Simulator-Fly%20in%20Browser-blueviolet?style=for-the-badge" alt="Live 3D Simulator" /></a>
  <a href="https://colab.research.google.com/github/blaze505050/drone-digital-twin/blob/main/examples/DronePy_Complete_Tutorial.ipynb"><img src="https://img.shields.io/badge/⚡%20Google%20Colab-1--Click%20Tutorial-orange?style=for-the-badge" alt="Open In Colab" /></a>
</p>

A comprehensive **multi-physics, closed-loop UAV Digital Twin ecosystem** and **DronePy** — a RocketPy-style multirotor UAV engineering simulation, stochastic analysis, and sim-to-real digital twin synchronization framework:

- **DronePy Engineering Layer**: Declarative notebook-first flight dynamics API (`from dronepy import Drone, Environment, Flight, MonteCarlo`), parallel Monte Carlo dispersion with CEP50/CEP95 circles, in-flight discrete event scheduling (payload drops, motor burnout), and multi-timeline digital twin synchronization.
- **Closed-Loop Digital Twin**: 15-state Multiplicative EKF (MEKF) fusing IMU, GPS, Baro, and Mag with parallel 6-DOF forward dynamic simulation, real-time divergence monitoring, and bounded parameter recalibration.
- **NASA F' (F Prime) Inspired SITL Bridge**: Clean-room SITL/HIL protocol bridge inspired by NASA JPL's component-based flight software architecture with binary wire framing (`0x5A5A5A5A`, CRC32) and telemetry channelization.
- **Glob3R-Inspired 3D Vision Perception**: Classical geometric 3D vision, volumetric voxel grids, and pseudo-LiDAR mapping inspired by the Global SfM / 3D vision paradigm, reconstructing 3D environments without physical LiDAR.
- **Aerodynamics & Propulsion**: Blade Element Momentum Theory (BEMT) propeller solver with Prandtl loss corrections, OpenFOAM CFD pipeline, and PINN neural aerodynamic surrogates.
- **Prognostics & Health**: NASA battery degradation dataset (B0005) calibration, Thevenin ECM, finite element structural modal analysis, and autoencoder predictive maintenance.
- **Flight Autonomy & Sim-to-Real**: Pluggable Command Sinks (Gazebo SITL and MAVLink hardware), YOLO vision perception bridge for GPS-denied relative navigation, and Gymnasium RL flight controller.
- **Defense & Swarm**: Multi-UAV swarm formation control under electronic warfare (GPS jamming and RF degradation).

### 🎓 Interactive Showcase & Learning Resources
- 🎮 **[Live 3D WebGL Flight Simulator](https://blaze505050.github.io/drone-digital-twin/)**: Fly directly in any browser with keyboard (WASD/Shift/Space) or USB gamepad without installing Python. Features procedural Web Audio motor acoustics and 8-gate aerobatic racing.
- 📓 **[1-Click Colab Master Tutorial](examples/DronePy_Complete_Tutorial.ipynb)**: Full 18-step engineering tutorial runnable on the cloud with zero setup.
- 📚 **5-Part Student Aerospace Curriculum (`examples/labs/`)**:
  - [Lab 1: Vehicle Design, BEMT & Propulsion Sizing](examples/labs/Lab1_Vehicle_Design_and_BEMT.ipynb)
  - [Lab 2: Atmospheric Physics, ISA 1976 & Dryden Turbulence](examples/labs/Lab2_Atmospheric_Physics_and_Turbulence.ipynb)
  - [Lab 3: Dynamic Payload Drops, CG Shifts & Parallel-Axis Inertia](examples/labs/Lab3_Dynamic_Payload_Release_and_Inertia.ipynb)
  - [Lab 4: Stochastic Monte Carlo & Landing Dispersion Analysis](examples/labs/Lab4_Stochastic_Monte_Carlo_and_Dispersion.ipynb)
  - [Lab 5: Digital Twin Synchronization, Residuals & Diagnostics](examples/labs/Lab5_Digital_Twin_Sync_and_Diagnostics.ipynb)
- 📄 **[Technical Whitepaper & Architecture Specification](docs/WHITE_PAPER.md)**: Rigorous mathematical derivations, UIUC wind tunnel benchmarks, and cyber-physical twin equations.
- 💼 **[Portfolio & Resume Interview Guide](docs/PORTFOLIO_RESUME_GUIDE.md)**: STAR resume bullets for 4 career tracks and top 10 technical interview defense answers.

---

## 🚁 DronePy Quickstart: "RocketPy, but for Multirotors"

Simulate 6-DOF dynamics, inject failures, and run stochastic dispersions with clean declarative code:

```python
import numpy as np
from dronepy import Drone, Environment, Wind, Flight, Mission, MonteCarlo, Distribution

# 1. Define Vehicle & Atmosphere
drone = Drone.quadcopter(mass=1.5, arm_length=0.25)
wind = Wind.gust(base_speed=3.0, magnitude=5.0, duration=2.0, start_time=5.0)
env = Environment.standard_atmosphere(altitude=100.0, wind=wind)

# 2. Plan Mission
mission = Mission("Delivery")
mission.takeoff(altitude=10.0).goto(north=20.0, east=15.0, down=-10.0).land()

# 3. Simulate 6-DOF Dynamics
flight = Flight(drone=drone, environment=env, mission=mission, duration=25.0)
res = flight.result

print(f"Final Position NED: {res.pos_ned[-1]}")
print(f"Energy Consumed: {res.battery_energy_wh[-1]:.2f} Wh")

# 4. Stochastic Monte Carlo Dispersion
mc = MonteCarlo(drone=drone, num_simulations=100, seed=42)
mc.add_parameter("mass", Distribution.normal(1.5, 0.05))
mc.add_parameter("cd", Distribution.uniform(0.75, 0.95))
mc.add_parameter("wind_speed", Distribution.triangular(0.0, 3.0, 8.0))

mc_res = mc.run(duration=15.0, parallel=True)
summary = mc_res.summary()
print(f"Landing Dispersion: CEP50 = {summary['cep50_m']:.2f} m | CEP95 = {summary['cep95_m']:.2f} m")
```

---

## Closed-Loop Architecture

```
                               ┌───────────────────────────────────────────┐
                               │       PHYSICAL VEHICLE / SENSORS          │
                               │   (IMU, GPS, Barometer, Magnetometer)     │
                               └─────────────────────┬─────────────────────┘
                                                     │ Live Sensor Stream
                                                     ▼
┌───────────────────────┐             ┌────────────────────────────────────┐
│   Actuator Commands   │────────────►│  Multiplicative EKF (15 States)    │
│  (Thrusts / Setpoints)│             │  delta_x = [pos, vel, att, bg, ba] │
└──────────┬────────────┘             └──────────────────┬─────────────────┘
           │                                             │ Authoritative
           │ Commanded                                   │ Estimated State
           │ Inputs                                      ▼
           │                          ┌────────────────────────────────────┐
           │                          │     Twin / Reality Residual        │
           │                          │             Monitor                │
           ▼                          │ (Tracks Pos/Vel/Attitude Errors)   │
┌───────────────────────┐             └──────────────▲─────────────────────┘
│  Dynamic Twin Model   │                            │
│ (Parallel 6-DOF Rigid │────────────────────────────┘ Predicted State
│  Body + BEMT + Drag)  │  (Tagged as DataSource.TWIN)
└──────────▲────────────┘
           │
           │ Dynamic Parameter Adaptation (m, Cd, Kt)
┌──────────┴────────────┐
│  Online Recalibrator  │
│  (System-ID Loop /    │
│   PINN Parameter Opt) │
└───────────────────────┘
```

---

## Installation

### 1. Clone and install in editable mode

```bash
git clone https://github.com/blaze505050/drone-digital-twin.git
cd drone-digital-twin
pip install -e .
```

### 2. Verify platform with end-to-end integration demo

```bash
python examples/full_platform_demo.py
```

Expected output:
```text
=================================================================
  ✓ All 28 platform modules verified successfully.
  ✓ StateStore: data flow confirmed end-to-end.
  ✓ Closed-Loop Digital Twin: bidirectional synchronization active.
  ✓ NASA F' SITL Bridge: wire framing & channelization verified.
  ✓ Glob3R 3D Engine: vision-to-pseudo-LiDAR mapping confirmed.
  ✓ Zero import errors. Zero runtime exceptions.
=================================================================

Platform Status: ALL 28 SUBSYSTEMS VERIFIED IN SIMULATION (SIM-TO-REAL READY)
=================================================================
```

### 3. Run full test suite

```bash
python -m pytest tests/ -v
# 1010+ tests, 0 failures
```

### 4. Optional dependencies

```bash
# MAVLink / Hardware connection
pip install pymavlink mavsdk

# Real-time WebSocket Dashboard
pip install aiohttp

# Deep Learning / Autograd PINN
pip install torch deepxde

# RL Training
pip install gymnasium stable-baselines3

# Telemetry recording
pip install h5py

# CAD & Mesh post-processing
pip install trimesh meshio
```

---

## Complete Module Reference

| # | Module | Category | Description |
|---|--------|----------|-------------|
| 1 | `state_manager` | Core | Thread-safe singleton `StateStore`, `DroneStateVector` schema, and subscription bus. |
| 2 | `telemetry_engine` | Core | 100 Hz multi-source ingestion engine, priority `SourceArbiter`, and HDF5 recorder. |
| 3 | `mavlink_bridge` | Core | Real-time PX4 MAVLink parser supporting 11 standard messages and `MAVLinkSource`. |
| 4 | `ros2_bridge` | Core | Zero-dependency ROS2 Humble message encoders/decoders with NED↔ENU coordinate mapping. |
| 5 | `dashboard` | Core | Real-time WebSocket serializer with delta-compression for web telemetry dashboards. |
| 6 | `sitl` | Simulation | PX4 SITL Docker/native process supervisor and multi-drone fleet configuration. |
| 7 | `gazebo_bridge` | Simulation | Dynamic Gazebo Harmonic SDF world/model builder and transport bridge. |
| 8 | `mission_planner` | Autonomy | MAVLink-aligned mission executor, geofencing, and QGC plan export. |
| 9 | `sensor_models` | Physics | Synthetic IMU, GPS, barometer, and magnetometer with calibrated Allan variance noise. |
| 10/11 | `cad_engine` | Aerospace | 3D mesh geometry analysis, cross-sectional area calculation, and mass properties. |
| 12 | `openfoam_bridge` | Aerospace | Automated OpenFOAM case generation, Reynolds/Mach flow solvers, and `AeroDatabase`. |
| 13 | `validation_framework` | Aerospace | Statistical trajectory validation (RMSE, MAE, R²) aligned with ASME V&V 10 standards. |
| 14 | `pinn_engine` | AI Research | Physics-Informed Neural Network aerodynamic surrogate predicting $C_L$ and $C_D$. |
| 15 | `battery_twin` | AI Research | 2-RC Thevenin Equivalent Circuit Model, thermal balance, and SEI capacity fade. |
| 16 | `structural_twin` | AI Research | 3D Euler-Bernoulli beam FEM, modal vibration analysis, and Palmgren-Miner fatigue. |
| 17 | `predictive_maintenance`| AI Research | Real-time autoencoder anomaly detection, health index scoring, and RUL scheduling. |
| 18 | `hil_interface` | Flight Stack | Hardware-In-The-Loop interface connecting Pixhawk flight controllers via USB/UART. |
| 19 | `rl_controller` | Autonomy | Full 6-DOF Gymnasium environment with motor lag, drag, rotor gyro, and curriculum. |
| 20 | `ai_copilot` | AI Research | Local engineering LLM copilot adapter (Ollama / Llama-3 integration). |
| 21 | `swarm_engine` | Swarm | Consensus formation flight with electronic warfare / GPS-jamming simulation. |
| 22 | `cad_engine.bemt` | Aerospace | Blade Element Momentum Theory (BEMT) propeller analysis with Prandtl tip/hub loss. |
| 23 | `battery_twin.nasa_adapter`| AI Research | NASA 18650 battery dataset (B0005) loader and degradation model parameter calibration. |
| 24 | `validation_framework.benchmarks`| Aerospace | ASL EuRoC MAV and Zurich Urban UAV benchmark loaders for state estimation validation. |
| 25 | `digital_twin_core` | Core Twin | 15-state MEKF state estimator, parallel 6-DOF physics twin, residual tracking, and online recalibration. |
| 26 | `mission_planner.sinks` & `perception_bridge` | Sim-to-Real | Sim-to-Real command sinks (`GazeboCommandSink`, `MAVLinkCommandSink`) and YOLO vision bridge. |
| 27 | `fprime_bridge` | Flight Stack | NASA F Prime (F') inspired clean-room SITL/HIL protocol bridge with framed telemetry and command channelization. |
| 28 | `perception_bridge.glob3r` | Perception | Glob3R-inspired 3D vision perception engine: pseudo-LiDAR point clouds, 3D voxel grids, and landing zone DEMs. |

---

## Quick-Start Examples

### 1. Running the Closed-Loop Digital Twin

```python
import numpy as np
from drone_sdk.digital_twin_core import ClosedLoopDigitalTwin

# Initialize closed-loop twin
twin = ClosedLoopDigitalTwin(vehicle_id="drone_alpha")

# 1. Ingest physical sensor measurements (100 Hz)
est_state = twin.update_sensors(
    accel_b=np.array([0.0, 0.0, -9.81]),
    gyro_b=np.array([0.0, 0.0, 0.0]),
    dt=0.01,
    gps_pos=np.array([10.0, 5.0, -20.0]),
    gps_vel=np.array([1.5, 0.0, 0.0]),
    baro_alt=20.0,
)

# 2. Step parallel physics prediction twin under commanded thrusts
pred_state, report = twin.step(
    commanded_action=np.full(4, 0.55),
    dt=0.01,
)

print(f"Tracking Status: {report.status.name}")
print(f"Position Residual: {report.pos_error_m * 100:.1f} cm")
print(f"Health Score: {report.health_score * 100:.1f}%")
```

### 2. Solving Propeller Aerodynamics via BEMT

```python
from drone_sdk.cad_engine import BladeElementMomentumSolver

solver = BladeElementMomentumSolver()
result = solver.solve(rpm=5200.0, v_inf=0.0)

print(f"Thrust: {result.thrust_n:.2f} N")
print(f"Torque: {result.torque_nm * 1000:.1f} mN·m")
print(f"Power:  {result.power_w:.1f} W")
print(f"Figure of Merit: {result.figure_of_merit:.2f}")
```

### 3. Calibrating Battery Aging with NASA Datasets

```python
from drone_sdk.battery_twin import NASABatteryDatasetAdapter, DegradationModel

adapter = NASABatteryDatasetAdapter(cell_id="B0005")
model = DegradationModel()

# Fit aging model to 168 NASA discharge cycles
alpha = adapter.calibrate_degradation_model(model, temperature_c=24.0)
soh_168 = model.capacity_fade(n_cycles=168, temperature_c=24.0)

print(f"Fitted SEI Growth Rate: {alpha:.6f}")
print(f"Predicted SOH at Cycle 168: {soh_168 * 100:.1f}%")
```

### 4. Sim-to-Real Mission Dispatch

```python
import numpy as np
from drone_sdk.mission_planner import Mission, MissionItem, GeoPoint, MissionExecutor, GazeboCommandSink, MAVLinkCommandSink

home = GeoPoint(lat=47.3977, lon=8.5455, alt=500.0)
mission = Mission("patrol", home=home)
mission.add(MissionItem.waypoint(0, GeoPoint(47.3985, 8.5460, 520.0), altitude_agl=20.0, speed_ms=8.0))

# Switch seamlessly between Gazebo simulation and Pixhawk hardware
sink = MAVLinkCommandSink(target_system=1)
executor = MissionExecutor(mission=mission, vehicle_id="drone_0", sink=sink)

# Advance waypoint navigation and dispatch MAVLink SET_POSITION_TARGET_LOCAL_NED
executor.tick()
print(f"Dispatched MAVLink packet: {len(sink.last_packet)} bytes")
```

### 5. Bridging NASA F' (F Prime) SITL Flight Computer

```python
from drone_sdk.fprime_bridge import FPrimeSITLBridge, FPrimeCommandOpcode
from drone_sdk.state_manager import StateFactory

bridge = FPrimeSITLBridge(vehicle_id="drone_fprime", use_loopback=True)
bridge.start()

# Transmit framed telemetry stream (0x5A5A5A5A sync + channelization + CRC32)
state = StateFactory.create_initial("drone_fprime").copy_with(x=10.0, y=5.0, z=-20.0)
bridge.send_telemetry(state)

# Ingest and decode flight computer commands from NASA F'
bridge.send_simulated_fprime_command(FPrimeCommandOpcode.SET_POSITION_NED, 25.0, 10.0, -30.0, 0.0)
commands = bridge.poll_commands()
print(f"Decoded F' Target: {commands[0].pos_target_ned}")
bridge.stop()
```

### 6. Vision-Based Pseudo-LiDAR & 3D Mapping with Glob3R

```python
from drone_sdk.perception_bridge import Glob3RPerceptionEngine
from drone_sdk.state_manager import StateFactory

glob3r = Glob3RPerceptionEngine()
drone_pose = StateFactory.create_initial("drone_0").copy_with(z=-10.0)

# Generate 3D maps and pseudo-LiDAR purely from vision
results = glob3r.process_perception_tick(drone_pose)
cloud = results["point_cloud"]
lidar = results["pseudo_lidar"]
voxel = results["voxel_grid"]
landing_zones = results["landing_sites"]

print(f"Dense 3D Points: {len(cloud)} vertices")
print(f"Pseudo-LiDAR: {len(lidar.ranges_m)} rays, nearest obstacle={lidar.min_distance_m:.1f} m")
print(f"Safe Landing Sites Identified: {len(landing_zones)}")
```

---

## Research & Defense Applications

| Focus Area | Relevant Modules | Impact & Applications |
|---|---|---|
| **Autonomous Defense Systems** | `swarm_engine`, `perception_bridge`, `digital_twin_core` | GPS-denied navigation during electronic warfare, RF jamming resilience, and automated target tracking. |
| **Aerospace Propulsion & CFD** | `cad_engine.bemt`, `openfoam_bridge`, `pinn_engine` | Sub-millisecond neural surrogates for aerodynamic coefficients and blade-level multirotor efficiency optimization. |
| **Fleet Prognostics & Airworthiness**| `battery_twin`, `structural_twin`, `predictive_maintenance` | Remaining Useful Life (RUL) estimation calibrated to NASA experimental data, structural modal strain, and predictive maintenance. |
| **Sim-to-Real Autonomy** | `rl_controller`, `mission_planner.sinks`, `validation_framework` | Zero-shot transfer from 6-DOF gym environments to Pixhawk autopilots, validated against ASL EuRoC MAV flight benchmarks. |

---

## Project Structure

```text
drone-digital-twin/
├── sdk/
│   └── drone_sdk/
│       ├── state_manager/          # Thread-safe StateStore & schema
│       ├── telemetry_engine/       # 100 Hz multi-source ingestion & arbiter
│       ├── digital_twin_core/      # MEKF, 6-DOF twin, residual monitor & recalibrator
│       ├── cad_engine/             # CAD analysis, BEMT propeller aerodynamics
│       ├── openfoam_bridge/        # OpenFOAM CFD pipeline & PINN data exporter
│       ├── pinn_engine/            # Neural aerodynamic surrogates
│       ├── battery_twin/           # Thevenin ECM & NASA battery aging calibration
│       ├── structural_twin/        # Finite element beam & modal analysis
│       ├── predictive_maintenance/ # Autoencoder anomaly detection & health index
│       ├── mission_planner/        # Mission executor & pluggable command sinks
│       ├── fprime_bridge/          # NASA F Prime (F') SITL/HIL flight computer bridge
│       ├── perception_bridge/      # YOLO detector & Glob3R 3D pseudo-LiDAR mapping
│       ├── rl_controller/          # 6-DOF Gym environment & motor dynamics
│       ├── swarm_engine/           # Swarm consensus & electronic warfare simulation
│       ├── mavlink_bridge/         # PX4/MAVLink parser
│       ├── ros2_bridge/            # ROS2 Humble converters
│       ├── sensor_models/          # Sensor noise models
│       ├── sitl/                   # SITL process launcher
│       ├── gazebo_bridge/          # Gazebo SDF builder
│       ├── dashboard/              # WebSocket telemetry streaming
│       └── validation_framework/   # ASME V&V benchmarks (EuRoC, Zurich UAV)
├── tests/
│   └── unit_tests/                 # 1000+ pytest unit tests
├── examples/
│   └── full_platform_demo.py       # Complete 28-module integration demo
├── docs/                           # Architecture documentation & API guides
├── pyproject.toml                  # Modern PEP 621 / setuptools configuration
├── setup.py                        # Backward-compatible build script
├── LICENSE                         # MIT License
└── README.md
```

## Verification & Evidence

The platform separates verification levels across all subsystems:

- **Empirically Validated**: Battery Equivalent Circuit Model (ECM) and SEI degradation curves calibrated directly to NASA Ames Li-ion 18650 experimental data (B0005, 168 discharge cycles).
- **Analytically Verified**: 6-DOF Euler dynamic equations with full 3×3 inertia tensor, NED gravity formulation, BEMT propeller aerodynamics with Prandtl losses, and Euler-Bernoulli structural modal analysis.
- **Simulation-Verified**: 15-state Multiplicative EKF (MEKF) with quaternion kinematics, divergence monitoring, RL 6-DOF flight control, Glob3R geometric 3D vision, and multi-agent swarm EW resilience.
- **Clean-Room Protocol Bridges**: NASA F' SITL/HIL wire protocol framing (`0x5A5A5A5A` sync, CRC32, channelized telemetry) and MAVLink v1/v2 packet deserialization.

For a full subsystem-by-subsystem evidence breakdown, refer to [docs/claims.md](docs/claims.md).

---

## Known Limitations & Operational Scope

1. **Hardware-In-The-Loop**: HIL execution requires a physically connected Pixhawk or compatible flight controller over serial UART/USB.
2. **CFD Solvers**: The OpenFOAM bridge generates complete case directories; running full 3D RANS/LES requires an installed `openfoam` environment.
3. **Magnetometer Reference**: The EKF magnetometer observation model assumes a calibrated local geomagnetic reference vector; soft/hard iron field distortion calibration is required for custom airframes.
4. **Flight Log Replay**: State estimation benchmarks fall back to synthetic trajectories when raw EuRoC or Zurich Urban flight logs are not downloaded locally.

---

## License

MIT License — free for academic, research, and commercial use.

---

## Citation

If you use this platform in academic or industrial research:

```bibtex
@software{uav_digital_twin_2026,
  title  = {UAV Digital Twin Platform: Closed-Loop State Estimation, NASA F' Flight Software, Glob3R 3D Vision Perception, and Sim-to-Real Autonomy},
  author = {blaze505050},
  year   = {2026},
  url    = {https://github.com/blaze505050/drone-digital-twin},
  note   = {28-module production UAV digital twin ecosystem}
}
```
