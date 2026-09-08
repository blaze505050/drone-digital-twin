# Platform Claims & Evidence Matrix

This document substantiates all technical capabilities and claims made across the UAV Digital Twin platform. It distinguishes between **Empirically Validated** (benchmarked against experimental flight/test data), **Analytically Verified** (tested against closed-form mathematical solutions), and **Simulation-Verified** (verified end-to-end in automated test suites).

---

## 1. Subsystem Verification & Evidence Levels

| Subsystem | Module | Verification Level | Primary Evidence / Artifact | Constraints & Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **15-State MEKF State Estimator** | `digital_twin_core.estimator` | Analytically & Synthetically Verified | `tests/unit_tests/test_digital_twin_core.py` | Validated in 6-DOF simulation with synthetic sensor noise & bias; real-log ingestion available via `benchmarks.py`. |
| **6-DOF Dynamic Twin** | `digital_twin_core.twin_model` | Analytically Verified | `tests/unit_tests/test_digital_twin_core.py` | Full 3x3 inertia tensor, NED gravity (+g in +z), quaternion kinematics, and specific-force accelerometer formulation. |
| **Online Recalibrator** | `digital_twin_core.recalibrator` | Verified in Simulation | `tests/unit_tests/test_digital_twin_core.py` | Mass prior frozen (`mass_scale=1.0`) by default; requires `enable_auto_apply=True` and sufficient dynamic excitation to update parameters. |
| **Battery Thevenin ECM & Degradation** | `battery_twin` | Empirically Validated | NASA Ames Battery Aging Dataset B0005 (`nasa_adapter.py`), `test_nasa_ecm_validation` | Exact matrix exponential state transition for RC branches; capacity fade calibrated to 168 discharge cycles. |
| **BEMT Propeller Aerodynamics** | `cad_engine.bemt` | Analytically Verified | `tests/unit_tests/test_cad_engine.py` | Blade Element Momentum Theory with Prandtl tip/hub loss and Glauert heavy-loading correction. |
| **Structural FEM & Modal Analysis** | `structural_twin` | Analytically Verified | `tests/unit_tests/test_structural_twin.py` | 3D Euler-Bernoulli beam elements, dynamic stiffness matrix, natural frequencies compared against beam theory. |
| **Predictive Maintenance & Health** | `predictive_maintenance` | Verified in Simulation | `tests/unit_tests/test_predictive_maintenance.py` | Feature extraction (time/frequency domain), autoencoder anomaly reconstruction, Palmgren-Miner cumulative fatigue. |
| **F' Flight Software Bridge** | `fprime_bridge` | Protocol Modeled | `tests/unit_tests/test_fprime_bridge.py` | Clean-room implementation of NASA JPL F' binary wire protocol framing (`0x5A5A5A5A`, CRC32, channelized telemetry). |
| **Glob3R 3D Vision Perception** | `perception_bridge` | Simulation Verified | `tests/unit_tests/test_perception_bridge.py` | Deterministic geometric projection, synthetic point cloud reconstruction, volumetric voxel occupancy, pseudo-LiDAR rays. |
| **Reinforcement Learning Environment** | `rl_controller` | Verified in Simulation | `tests/unit_tests/test_rl_controller.py` | Gymnasium 6-DOF environment with motor first-order lag, rotor gyroscopic precession, and NED dynamics. |
| **Swarm Consensus & EW** | `swarm_engine` | Verified in Simulation | `tests/unit_tests/test_swarm_engine.py` | Reynolds flocking + Olfati-Saber consensus with simulated GPS jamming and RF link degradation. |
| **MAVLink / PX4 Bridge** | `mavlink_bridge` | Wire-Protocol Verified | `tests/unit_tests/test_mavlink_bridge.py` | Zero-dependency binary parser for 11 standard MAVLink v1/v2 message types; requires physical serial/UDP connection for live vehicle. |
| **Hardware-In-The-Loop (HIL)** | `hil_interface` | Interface Verified | `tests/unit_tests/test_hil_interface.py` | Configured for Pixhawk serial interface; requires physical flight controller connected via USB/UART. |
| **OpenFOAM CFD Bridge** | `openfoam_bridge` | Script Generation Verified | `tests/unit_tests/test_openfoam_bridge.py` | Generates blockMesh, snappyHexMesh, and simpleFoam case directories; executing CFD requires external OpenFOAM installation. |
| **PINN Aerodynamic Surrogate** | `pinn_engine` | Simulation Verified | `tests/unit_tests/test_pinn_engine.py` | Physics-informed neural network surrogate predicting CL and CD; fast NumPy inference fallback included. |

---

## 2. Coordinate System Standards

The platform enforces rigorous aerospace coordinate conventions:

- **Navigation Frame**: North-East-Down (NED), where:
  - $+X$: Geodetic North
  - $+Y$: Geodetic East
  - $+Z$: Downward (toward Earth center; gravity vector is $[0, 0, +9.80665]\,\text{m/s}^2$)
- **Body Frame**: Forward-Right-Down (FRD), where:
  - $+X_b$: Forward nose
  - $+Y_b$: Right wing / starboard arm
  - $+Z_b$: Bottom / belly
- **Attitude Representation**: Unit quaternion $q = [q_w, q_x, q_y, q_z]$ representing body-to-NED passive rotation:
  $$v_{\text{ned}} = R(q)\, v_b$$
- **IMU Specific Force**: Accelerometers measure specific force $f_b = R(q)^T (a_{\text{ned}} - g_{\text{ned}})$. At rest in level flight:
  $$a_{\text{ned}} = [0, 0, 0]^T \implies f_b = [0, 0, -g]^T$$

---

## 3. Data Provenance & Anti-Deception Policy

1. **Truth vs. Sensor Isolation**: Sensors (IMU, GPS, Magnetometer, Barometer) never copy ground-truth states directly without explicit `DataStatus.SYNTHETIC` markings.
2. **Benchmark Integrity**: Evaluation against public datasets (e.g. Zurich Urban, EuRoC) reports status as `REAL` only when original raw measurements and ground truth logs are loaded from disk. If data is absent, fallback runs are explicitly tagged as `SYNTHETIC_FALLBACK`.
3. **No Hidden Pass-Through**: Recalibration routines cannot adjust vehicle parameters without verified dynamic persistence and observability.
