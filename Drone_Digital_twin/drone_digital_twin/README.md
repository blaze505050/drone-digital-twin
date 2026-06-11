# UAV Digital Twin Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" />
  <img src="https://img.shields.io/badge/Tests-930%2B%20passing-brightgreen" />
  <img src="https://img.shields.io/badge/Modules-21-orange" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

A **production-quality, open-source UAV Digital Twin ecosystem** implementing
21 integrated modules across five engineering phases:

- Real-time drone simulation and hardware synchronisation
- PX4 SITL / HIL / MAVLink / ROS2 integration
- OpenFOAM CFD surrogate modelling
- Physics-Informed Neural Networks (PINN)
- Battery / Structural / Predictive Maintenance digital twins
- Reinforcement Learning flight controllers
- Multi-drone swarm simulation with electronic warfare scenarios

---

## Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────────┐
│                    UAV Digital Twin Platform                      │
│                                                                  │
│  PX4 SITL / Hardware                                             │
│      │ MAVLink UDP                                               │
│      ▼                                                           │
│  MAVLinkSource ──► TelemetryEngine ──► StateStore ──► Dashboard  │
│                         │                  │                     │
│                    SourceArbiter      ROS2 Bridge                │
│                                            │                     │
│  AI Layer:   PINN │ BatteryTwin │ StructTwin │ PredMaint │ RL   │
│  Aerospace:  CFD Surrogate │ CAD Import │ OpenFOAM │ Validation  │
│  Swarm:      SwarmManager │ FormationCtrl │ DefenseSim           │
└──────────────────────────────────────────────────────────────────┘
```

---

## Installation

### 1. Clone and install (core — no optional deps)

```bash
git clone https://github.com/blaze505050/drone_digital_twin.git
cd drone_digital_twin
pip install -e .
```

### 2. Verify installation

```bash
python examples/full_platform_demo.py
```

Expected last line:
```
✓ All 21 platform modules verified successfully.
```

### 3. Run full test suite

```bash
python -m pytest tests/ -v
# 930+ tests, 0 failures
```

### 4. Optional dependencies

```bash
# MAVLink / PX4
pip install pymavlink mavsdk

# Dashboard server
pip install aiohttp

# RL training
pip install gymnasium stable-baselines3

# PINN with autograd
pip install torch deepxde

# Telemetry recording
pip install h5py

# CFD post-processing
pip install PyFoam

# CAD/mesh
pip install trimesh meshio
```

---

## Module Reference

| # | Module | Phase | Description |
|---|--------|-------|-------------|
| 1 | `state_manager` | Core | Thread-safe singleton StateStore, DroneStateVector schema |
| 2 | `telemetry_engine` | Core | 100 Hz multi-source telemetry ingestion, HDF5 recording |
| 3 | `mavlink_bridge` | Core | MAVLink/PX4 parser, 11 message types, MAVLinkSource |
| 4 | `ros2_bridge` | Core | ROS2 Humble topic publisher/subscriber, NED↔ENU |
| 5 | `dashboard` | Core | WebSocket + HTTP real-time dashboard, diff compression |
| 6 | `sitl` | Flight Stack | PX4 SITL Docker/native launcher, fleet config |
| 7 | `gazebo_bridge` | Flight Stack | SDF world/model builder, GazeboSource, WorldManager |
| 8 | `mission_planner` | Flight Stack | Waypoints, geofencing, path planning, QGC export |
| 9 | `sensor_models` | Flight Stack | IMU/GPS/baro/mag/LiDAR with Allan variance noise |
| 10/11 | `cad_engine` | Aerospace | STL/OBJ import, mesh analysis, mass estimation |
| 12 | `openfoam_bridge` | Aerospace | OpenFOAM case generation, results parsing, AeroDatabase |
| 13 | `validation_framework` | Aerospace | Statistical validation, benchmark suites, regression |
| 14 | `pinn_engine` | AI Research | PINN dynamics/aero surrogate, NumPy autoencoder |
| 15 | `battery_twin` | AI Research | Thevenin ECM, SEI degradation, SOH/RUL prediction |
| 16 | `structural_twin` | AI Research | FEM beam model, modal analysis, SHM, fatigue |
| 17 | `predictive_maintenance` | AI Research | Anomaly detection, health index, maintenance scheduler |
| 18 | HIL | Advanced | Hardware-In-the-Loop via MAVLinkSource (HARDWARE priority) |
| 19 | `rl_controller` | Advanced | Gymnasium env, reward shaping, domain randomisation, curriculum |
| 20 | AI Copilot | Advanced | Ollama local LLM adapter (runtime — no pip dep) |
| 21 | `swarm_engine` | Advanced | Formation control, communication model, DefenseSim |

---

## Quick-Start Examples

### Simulate a hover and read state

```python
from drone_sdk.state_manager import StateStore, StateFactory, VehicleConfig
from drone_sdk.telemetry_engine import TelemetryEngine, CallableSource, SourcePriority
from drone_sdk.state_manager import DataSource, DroneStateUpdate
import numpy as np, time

# Create state store
store = StateStore.create("drone_0", VehicleConfig("drone_0"))

# Simple simulation source
t = [0.0]
def poll():
    t[0] += 0.01
    upd = DroneStateUpdate("drone_0", DataSource.SITL)
    upd.position = np.array([t[0], 0.0, -5.0])   # NED hover at 5m
    upd.altitude_agl = 5.0
    return [upd]

source = CallableSource("sim", "drone_0", poll, SourcePriority.SIMULATION)
engine = TelemetryEngine()
engine.add_source(source, auto_connect=True)
engine.start()
time.sleep(1.0)
engine.stop()

state = store.get_latest()
print(f"Position: {state.position_ned()}")
print(f"Health:   {state.health_status.name}")
print(f"History:  {store.history_length} frames")
StateStore.destroy_all()
```

### Connect to PX4 SITL

```bash
# Terminal 1: Start PX4 SITL
docker run --rm --network host px4io/px4-dev-simulation-focal:latest \
  bash -c "cd /src/PX4-Autopilot && make px4_sitl_default none_iris"

# Terminal 2: Connect digital twin
```

```python
from drone_sdk.state_manager import StateStore
from drone_sdk.telemetry_engine import TelemetryEngine
from drone_sdk.mavlink_bridge import MAVLinkSource, SourcePriority

StateStore.create("drone_0")
source = MAVLinkSource("px4", "drone_0", "udp:127.0.0.1:14550",
                        priority=SourcePriority.SITL)
source.connect()
source.configure_default_streams(rate_hz=50)

engine = TelemetryEngine()
engine.add_source(source)
engine.start()

import time; time.sleep(5)
state = StateStore.get_instance("drone_0").get_latest()
print(f"Mode: {state.flight_mode.name}, Armed: {state.is_armed}")
engine.stop()
```

### Train a PINN aerodynamics surrogate

```python
from drone_sdk.pinn_engine import PINNFactory

# Generate synthetic training data (replace with real OpenFOAM data)
data    = PINNFactory.generate_synthetic_aero_data(n_samples=1000)
trainer = PINNFactory.aerodynamics_pinn(n_epochs=5000)
history = trainer.fit(data)

print(f"Best loss: {history.best_loss():.2e}")
trainer.save("models/aero_surrogate.json")

# Inference: replace OpenFOAM with instant prediction
import numpy as np
X    = np.array([[10.0, 0.0873, 1.67e5]])   # U=10m/s, AoA=5°, Re=167k
pred = trainer.predict(X)
print(f"CL={pred[0,0]:.4f}  CD={pred[0,1]:.4f}")
```

### Battery Digital Twin with health monitoring

```python
from drone_sdk.battery_twin import BatteryDigitalTwin, CellParameters

twin = BatteryDigitalTwin(CellParameters(), initial_soh=0.95)
twin.begin_flight(initial_soc=0.92)

# Simulate 10-minute flight
for step in range(600):
    # Update with MAVLink telemetry current (replace with real data)
    state = twin.update(current=12.5, dt=1.0)
    if step % 100 == 0:
        h = twin.get_health_summary()
        print(f"t={step}s  SOC={h['soc']:.2f}  RUL={h['rul_cycles']:.0f} cycles")

twin.end_flight()
```

### 5-drone formation flight

```python
from drone_sdk.swarm_engine import SwarmManager, FormationType

swarm = SwarmManager(n_drones=5, formation=FormationType.V_SHAPE, spacing=5.0)

# Simulate 100 steps
for _ in range(100):
    swarm.tick(dt=0.1)

D = swarm.inter_drone_distances()
print(f"Min separation: {D[D>0].min():.2f} m")
print(f"Swarm centroid: {swarm.get_swarm_centroid().round(2)}")
```

---

## Project Structure

```
drone_digital_twin/
├── sdk/
│   └── drone_sdk/
│       ├── state_manager/         # Module 1
│       ├── telemetry_engine/      # Module 2
│       ├── mavlink_bridge/        # Module 3
│       ├── ros2_bridge/           # Module 4
│       ├── dashboard/             # Module 5
│       ├── sitl/                  # Module 6
│       ├── gazebo_bridge/         # Module 7
│       ├── mission_planner/       # Module 8
│       ├── sensor_models/         # Module 9
│       ├── cad_engine/            # Modules 10+11
│       ├── openfoam_bridge/       # Module 12
│       ├── validation_framework/  # Module 13
│       ├── pinn_engine/           # Module 14
│       ├── battery_twin/          # Module 15
│       ├── structural_twin/       # Module 16
│       ├── predictive_maintenance/ # Module 17
│       ├── rl_controller/         # Module 19
│       └── swarm_engine/          # Module 21
├── tests/
│   └── unit_tests/
│       ├── state_manager/         # 186 tests
│       ├── telemetry_engine/      # 45 tests
│       ├── mavlink_bridge/        # 49 tests
│       ├── ros2_bridge/           # 38 tests
│       ├── dashboard/             # 33 tests
│       ├── sitl/                  # 50 tests
│       ├── gazebo_bridge/         # 41 tests
│       ├── mission_planner/       # 59 tests
│       ├── sensor_models/         # 48 tests
│       ├── cad_engine/            # 61 tests
│       ├── openfoam_bridge/       # 50 tests
│       ├── validation_framework/  # 40 tests
│       ├── pinn_engine/           # 45 tests
│       ├── battery_twin/          # 37 tests
│       ├── structural_twin/       # 45 tests
│       ├── predictive_maintenance/ # 50 tests
│       ├── rl_controller/         # 38 tests
│       └── swarm_engine/          # 59 tests
├── examples/
│   └── full_platform_demo.py      # End-to-end demo
├── docs/
│   └── state_manager/
│       ├── README.md
│       └── API.md
├── requirements.txt
├── setup.py
└── README.md
```

---

## Connecting to Real Hardware

### Pixhawk via USB

```python
from drone_sdk.mavlink_bridge import MAVLinkSource, ConnectionPreset
source = MAVLinkSource("hw", "drone_0",
                        ConnectionPreset.PIXHAWK_USB,   # /dev/ttyACM0:57600
                        priority=SourcePriority.HARDWARE)
```

### SiK Telemetry Radio

```python
source = MAVLinkSource("telemetry", "drone_0",
                        "/dev/ttyUSB0:57600",
                        priority=SourcePriority.HARDWARE)
```

### QGroundControl simultaneously

```python
# SITL exposes both ports; QGC uses 14550, API uses 14540
cfg = SITLConfig(instance=0)
print(cfg.gcs_port)       # 14550 — QGC connects here
print(cfg.api_port)       # 14540 — our MAVLinkSource connects here
```

---

## Research Applications

| Research Area | Relevant Modules | Key Output |
|--------------|-----------------|------------|
| Dynamics System ID | PINN (14), Validation (13) | Identified parameters |
| Battery Prognostics | Battery DT (15), Pred Maint (17) | RUL prediction accuracy |
| Aero CFD Surrogate | OpenFOAM (12), PINN (14) | CL/CD prediction speed |
| RL Sim-to-Real | RL Controller (19), Sensor Models (9) | Transfer success rate |
| Structural Health | Structural DT (16), Pred Maint (17) | Frequency shift sensitivity |
| Swarm Coordination | Swarm Engine (21), Mission (8) | Formation error, collision rate |
| GPS-Denied Nav | DefenseSim (21), EKF (state_manager) | Position error without GPS |

---

## Resume / Portfolio Highlights

**For DRDO / Defense Labs:**
- DefenseSim: GPS denial, RF jamming, adversarial drone — unique open-source
- SwarmManager: Multi-drone coordination under electronic warfare
- MAVLink integration with real PX4 flight controllers

**For ISRO / NAL / ADA:**
- OpenFOAM CFD pipeline with PINN surrogate
- BEMT-based propeller analysis integrated with 6-DOF dynamics
- Structural FEM + modal analysis for airframe qualification

**For Airbus / Boeing / Honeywell:**
- Battery Digital Twin with NASA-calibrated degradation model
- Predictive maintenance with LSTM autoencoder anomaly detection
- Validation Framework aligned with ASME V&V 10-2006

**For UAV Startups / Research Labs:**
- Full ROS2 Humble integration (MAVROS-compatible)
- Gymnasium RL environment with domain randomisation
- HDF5 flight log recording + replay for dataset generation

---

## License

MIT License — free for academic, research, and commercial use.

---

## Citation

If you use this platform in academic work:

```bibtex
@software{uav_digital_twin_2026,
  title  = {UAV Digital Twin Platform},
  author = {blaze505050},
  year   = {2026},
  url    = {https://github.com/blaze505050/drone_digital_twin},
  note   = {21-module open-source UAV digital twin ecosystem}
}
```
