# Real-Time Digital Twin Architecture

This document details the real-time parallel physics prediction, state synchronization, residual tracking, and online recalibration pipeline within the UAV Digital Twin platform and **DronePy**.

---

## 1. Architectural Philosophy: The Shadow Twin

The digital twin operates as an active, parallel "shadow" running alongside the physical vehicle or SITL simulation.

```
                    ┌─────────────────────────┐
                    │   Physical UAV Flight   │
                    │   or MAVLink Telemetry  │
                    └───────────┬─────────────┘
                                │  State Telemetry (MEKF Filtered)
                                ▼
                    ┌─────────────────────────┐
 Actuator Commands  │  Parallel Physics Twin  │
 ──────────────────>│   (6-DOF Dynamic Twin)  │
                    └───────────┬─────────────┘
                                │  Predicted State Trajectory
                                ▼
                    ┌─────────────────────────┐
                    │    Residual Monitor     │
                    │  (Twin vs Reality Comp) │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
┌───────────────────┐                       ┌───────────────────┐
│ Fault Diagnostics │                       │ System ID Engine  │
│ (Motor/Aero Slip) │                       │ (Recalibration)   │
└───────────────────┘                       └───────────────────┘
```

### The Role of `DynamicTwinModel`
`DynamicTwinModel` (in `drone_sdk.digital_twin_core.twin_model`) solves the coupled 6-DOF nonlinear equations of motion using identical actuator setpoints received by the physical drone's autopilot.

If reality and model match perfectly, the residual between their position, velocity, and attitude approaches zero. When a divergence occurs, the residual signature pinpoints the underlying physical anomaly.

---

## 2. Residual Signature Diagnostics

The `TwinResidualMonitor` (in `drone_sdk.digital_twin_core.residual_monitor`) continuously evaluates tracking divergence across three key axes:

| Residual Signature | Primary Observable | Probable Physical Root Cause |
| :--- | :--- | :--- |
| **High Velocity Residual, Low Attitude Residual** | Drone flies slower than expected for commanded pitch/roll | Wind disturbance or unmodeled translational drag shift |
| **High Attitude Residual, Asymmetric Motor Effort** | Motor 1 running 2,000 RPM higher than Motor 2 in hover | Propeller chip, structural distortion, or motor bearing degradation |
| **Steady Altitude Residual, Symmetrical RPM Offset** | Higher collective thrust required to maintain altitude | Unregistered payload mass addition |
| **Rapid Divergent Residual** | Attitude diverges exponentially within $< 100$ ms | Structural failure, propeller detachment, or sensor loss |

---

## 3. Real-Time Synchronization Pipeline: `DigitalTwinSynchronizer`

To integrate live telemetry streaming with the parallel twin in DronePy:

```python
from dronepy import DigitalTwinSynchronizer
from drone_sdk.state_manager.schema import DroneStateVector

# Initialize synchronizer for vehicle
synchronizer = DigitalTwinSynchronizer(vehicle_id="drone_alpha")

# Process incoming telemetry frame (e.g., at 100 Hz)
for real_state in telemetry_stream:
    predicted_twin_state, residual_report = synchronizer.step(real_state, dt=0.01)
    
    if residual_report.health_score < 0.70:
        print(f"WARNING: Twin divergence detected! Probable cause: {residual_report.probable_cause}")
```

---

## 4. Multi-Timeline Comparison: `TwinComparison`

`TwinComparison` aligns the physical and virtual timelines to compute rigorous statistical metrics:
- **Root Mean Square Error (RMSE)**: Position and velocity tracking accuracy.
- **Maximum Position Divergence**: Peak divergence during extreme maneuvers.
- **Energy Residual**: Measured battery power vs modeled electrical draw.

```python
from dronepy import TwinComparison

comparison = TwinComparison.compare(real=real_flight_result, twin=twin_flight_result)
print(f"Position Tracking RMSE: {comparison.rmse_position * 100:.1f} cm")
comparison.plot()  # Generates synchronized 3-panel comparative telemetry plot
```
