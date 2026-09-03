"""
drone_sdk.digital_twin_core
===========================
Closed-Loop Digital Twin Core Engine.

Transforms the UAV platform from a passive telemetry mirror into a genuine,
closed-loop digital twin by uniting:
  1. Multiplicative Extended Kalman Filter (continuous state estimation).
  2. Parallel 6-DOF Physics Prediction Twin (forward trajectory simulation).
  3. Twin/Reality Residual Monitor (divergence tracking & anomaly classification).
  4. Online Recalibration Loop (system-identification parameter adaptation).

Architecture::

      Sensors (IMU/GPS/Baro) ──► MultiplicativeEKF ──► Estimated State
                                                             │
                                                     Residual Monitor ◄── Predictive Maint
                                                             ▲
                                                             │
      Actuator Commands      ──► DynamicTwinModel  ──► Predicted State
                                       ▲
                                       │ (Adapts m, Cd, Kt)
                              Online Recalibrator

Python version: 3.9+
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np

from .estimator import MultiplicativeEKF
from .twin_model import DynamicTwinModel, TwinPhysicsParameters
from .residual_monitor import TwinResidualMonitor, ResidualReport
from .recalibrator import OnlineRecalibrator
from drone_sdk.state_manager.schema import DroneStateVector


class ClosedLoopDigitalTwin:
    """Unified closed-loop digital twin combining estimation, simulation, residual tracking, and recalibration."""

    def __init__(
        self,
        vehicle_id: str = "drone_0",
        params: Optional[TwinPhysicsParameters] = None,
        init_pos_ned: Optional[np.ndarray] = None,
        init_yaw_rad: float = 0.0,
    ) -> None:
        self.vehicle_id = vehicle_id
        self.estimator = MultiplicativeEKF(vehicle_id=vehicle_id, init_pos_ned=init_pos_ned, init_yaw_rad=init_yaw_rad)
        self.twin_model = DynamicTwinModel(vehicle_id=vehicle_id, params=params)
        self.residual_monitor = TwinResidualMonitor()
        self.recalibrator = OnlineRecalibrator(twin_model=self.twin_model)

        self._latest_estimated: Optional[DroneStateVector] = None
        self._latest_predicted: Optional[DroneStateVector] = None
        self._latest_report: Optional[ResidualReport] = None

    def update_sensors(
        self,
        accel_b: np.ndarray,
        gyro_b:  np.ndarray,
        dt:      float,
        gps_pos: Optional[np.ndarray] = None,
        gps_vel: Optional[np.ndarray] = None,
        baro_alt: Optional[float] = None,
        mag_b:   Optional[np.ndarray] = None,
    ) -> DroneStateVector:
        """Update continuous state estimator with incoming sensor packet."""
        self.estimator.predict(accel_b, gyro_b, dt)

        if gps_pos is not None and gps_vel is not None:
            self.estimator.update_gps(gps_pos, gps_vel)
        if baro_alt is not None:
            self.estimator.update_barometer(baro_alt)
        if mag_b is not None:
            self.estimator.update_magnetometer(mag_b)

        est_state = self.estimator.get_state_vector()
        self._latest_estimated = est_state
        return est_state

    def step(
        self,
        commanded_action: np.ndarray,
        dt: float,
    ) -> Tuple[DroneStateVector, ResidualReport]:
        """Advance parallel twin physics and monitor divergence against estimated vehicle state."""
        # 1. Step parallel physics prediction
        pred_state = self.twin_model.step(commanded_action, dt)
        self._latest_predicted = pred_state

        # 2. Compare against estimated state
        actual = self._latest_estimated if self._latest_estimated is not None else pred_state
        report = self.residual_monitor.update(predicted=pred_state, actual=actual)
        self._latest_report = report

        # 3. Feed frame into online recalibrator
        self.recalibrator.record_frame(commanded_action, actual)

        return pred_state, report

    def seed(self, state: DroneStateVector) -> None:
        """Synchronize twin model state directly to a known state."""
        self.twin_model.seed_from_state(state)
        self.estimator.p = np.array([state.x, state.y, state.z], dtype=np.float64)
        self.estimator.v = np.array([state.vx, state.vy, state.vz], dtype=np.float64)
        self.estimator.q = np.array([state.q0, state.q1, state.q2, state.q3], dtype=np.float64)

    @property
    def latest_estimated_state(self) -> Optional[DroneStateVector]:
        return self._latest_estimated

    @property
    def latest_predicted_state(self) -> Optional[DroneStateVector]:
        return self._latest_predicted

    @property
    def latest_report(self) -> Optional[ResidualReport]:
        return self._latest_report


__all__ = [
    "MultiplicativeEKF",
    "DynamicTwinModel",
    "TwinPhysicsParameters",
    "TwinResidualMonitor",
    "ResidualReport",
    "OnlineRecalibrator",
    "ClosedLoopDigitalTwin",
]
