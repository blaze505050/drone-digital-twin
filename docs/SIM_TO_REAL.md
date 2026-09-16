# Sim-to-Real Methodology and Calibration Architecture

Transferring controllers and mission policies from numerical simulation to physical flight requires robust bounds and verification against physical realities. This document describes the **Sim-to-Real** bridge in the UAV Digital Twin platform and DronePy.

---

## 1. The Sim-to-Real Gap

Discrepancies between simulation and flight arise from four primary sources:
1. **Unmodeled Aerodynamics**: Rotor-to-rotor aerodynamic wake interactions and ground effect variations.
2. **Actuator Non-Linearities**: Voltage sag under peak current draw and non-instantaneous motor spin-up lags ($\tau_m$).
3. **Mass and Inertia Variations**: Minor payload imbalances and shifts in the center of gravity ($CG$).
4. **Sensor Noise & Biases**: Gyro drift, accelerometer vibration harmonics, and magnetometer anomalies.

---

## 2. Bounded Parameter Identification (`DigitalTwinCalibrator`)

To eliminate the sim-to-real gap without risking divergence, DronePy wraps `BoundedParameterIdentifier` (`drone_sdk.identification.identifier`).

### Safety Principles of the Calibrator
1. **Persistent Excitation Gating**:
   A steady-state hover flight does not contain enough dynamic information to identify mass, drag, and thrust independently. The calibrator computes the regressor condition number $\kappa$:
   $$\kappa = \frac{\sigma_{max}}{\sigma_{min}} \le 100.0$$
   If $\kappa > 100.0$, the estimation is rejected as ill-conditioned to prevent model drift.

2. **Strict Physical Box Bounds**:
   Parameter updates are bounded to $\pm 15\%$ maximum shift relative to nominal CAD priors:
   $$\theta_{calibrated} \in [0.85 \cdot \theta_{nominal}, 1.15 \cdot \theta_{nominal}]$$

3. **Disjoint Holdout Window Evaluation**:
   Candidate parameters are evaluated on a holdout segment of flight telemetry. If validation RMSE increases, the update is rolled back automatically.

### Running Calibration in DronePy
```python
from dronepy import DigitalTwinCalibrator, Drone, FlightReplay

# Load real flight telemetry log
flight_log = FlightReplay.from_csv("logs/flight_telemetry_2026_09_16.csv")

# Baseline nominal drone model
nominal_drone = Drone.quadcopter(mass=1.50)

# Execute bounded calibration
calibrator = DigitalTwinCalibrator(max_condition_number=100.0, max_parameter_shift_pct=15.0)
calibrated_drone, report = calibrator.calibrate(nominal_drone, flight_log)

if report.success:
    print(f"Calibration successful! Validated RMSE = {report.validation_rmse:.4f}")
    print(f"Calibrated Mass: {calibrated_drone.mass:.3f} kg")
else:
    print(f"Calibration rejected: {report.message}")
```

---

## 3. Flight Replay & Benchmark Comparison

Using `FlightReplay`, real-world CSV logs from Pixhawk/PX4, ArduPilot, or custom autopilots can be directly imported into DronePy to replay trajectories and evaluate tracking fidelity:

```python
from dronepy import FlightReplay, TwinComparison

replay = FlightReplay.from_csv("real_flight.csv")
real_flight_result = replay.to_flight_result()

# Re-simulate identical mission in DronePy
sim_flight_result = drone.simulate(duration=replay.duration)

# Compute comparative residuals
comparison = TwinComparison.compare(real=real_flight_result, twin=sim_flight_result)
print(f"Model Accuracy (RMSE): {comparison.rmse_position * 100:.1f} cm")
```
