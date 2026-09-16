# Platform Validation & Verification Standards

This document establishes the testing and verification standards across the 32 subsystems of the UAV Digital Twin platform and **DronePy**.

---

## 1. Test Suite Architecture

The repository maintains an automated unit test suite with 1,119 tests spanning every layer of the architecture:

```
tests/unit_tests/
├── aerodynamics/             # CFD & PINN surrogate models
├── battery_twin/             # ECM pack thermodynamics & SOC estimation
├── cad_engine/               # BEMT propeller & airframe structural models
├── configuration/            # Parametric airframe schemas & validation
├── digital_twin_core/        # Dynamic physics twin & residual monitors
├── dronepy/                  # High-level DronePy simulation & analysis tests (25 tests)
├── flight_simulator/         # 6-DOF numerical integration & ground contact
├── identification/           # Bounded parameter estimation & PE checks
├── math_models/              # Quaternion transformations & rigid-body equations
├── mavlink_bridge/           # Telemetry packet encoding/decoding
├── mission_planner/          # Geodetic waypoint navigation & spline smoothing
├── predictive_maintenance/   # Health index & RUL estimation
├── safety/                   # Flight envelopes & emergency failsafes
└── validation_framework/     # Telemetry residual scoring & statistical tests
```

---

## 2. Verification Protocol

All modules and changes are validated against four strict operational criteria:

### Criterion 1: Mass & Conservation Laws
- Mass additions/drops must strictly conserve physical volume and momentum.
- Payload release shifts the center of gravity and relaxes the inertia tensor according to the parallel-axis theorem:
  $$\mathbf{I}_{new} = \mathbf{I}_{old} + m (\mathbf{r}^T\mathbf{r} \mathbf{I}_3 - \mathbf{r} \mathbf{r}^T)$$

### Criterion 2: Coordinate Frame Integrity
- NED (North-East-Down) inertial coordinates: $+X$ North, $+Y$ East, $+Z$ Down.
- FRD (Forward-Right-Down) body frame: $+X$ Forward, $+Y$ Right, $+Z$ Down.
- Thrust acts in the $-Z_b$ direction (upwards).

### Criterion 3: Deterministic Reproducibility
- Given identical random seeds (`seed=42`), all stochastic Monte Carlo runs and Dryden turbulence histories reproduce bit-for-bit identical trajectories.

### Criterion 4: Safe Hardware Interlocking
- Default system state is strictly `SIMULATION_ONLY`.
- Flight commands cannot be dispatched to physical serial/network MAVLink bridges without explicit confirmation tokens (`CONFIRM_HARDWARE_FLIGHT_SAFETY_CHECKED`) and active two-step arming sequences.

---

## 3. Running Test Suites

### Run All Unit Tests
```bash
python -m pytest tests/unit_tests/ -v
```

### Run DronePy Engineering Layer Tests Specifically
```bash
python -m pytest tests/unit_tests/dronepy/ -v
```

### Run Full Platform Integration Demo
```bash
python examples/full_platform_demo.py
```
