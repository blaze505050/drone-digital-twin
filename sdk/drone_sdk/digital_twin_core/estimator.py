"""
drone_sdk.digital_twin_core.estimator
=====================================
Multiplicative Extended Kalman Filter (MEKF) for UAV State Estimation.

Fuses high-rate IMU (3-axis accelerometer, 3-axis gyroscope), barometer,
magnetometer, and GPS into a continuous 6-DOF DroneStateVector.

Error-State MEKF formulation:
  - 15-state continuous-discrete Kalman filter:
    delta_x = [delta_p (3), delta_v (3), delta_theta (3), delta_b_g (3), delta_b_a (3)]
  - Quaternion attitude representation avoids Euler singularity / gimbal lock.
  - Multiplicative error quaternion preserves unit norm constraints:
    q_true = q_est (x) delta_q(delta_theta)

Literature:
  - Markley, F. L. (2003). "Attitude error representations for Kalman filtering."
    Journal of Guidance, Control, and Dynamics, 26(2), 311-317.
  - Farrell, J. A. (2008). Aided Navigation: GPS with High Rate Sensors. McGraw-Hill.

Python version: 3.9+
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from drone_sdk.state_manager.schema import (
    ArmingState,
    DataSource,
    DroneStateUpdate,
    DroneStateVector,
    FlightMode,
    HealthStatus,
)


class MultiplicativeEKF:
    """15-State Multiplicative Extended Kalman Filter for UAV State Estimation."""

    def __init__(
        self,
        vehicle_id: str = "drone_twin",
        init_pos_ned: Optional[np.ndarray] = None,
        init_yaw_rad: float = 0.0,
    ) -> None:
        self.vehicle_id = vehicle_id

        # Nominal States
        self.p = np.array(init_pos_ned if init_pos_ned is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)

        # Quaternion [q0, q1, q2, q3] (w, x, y, z)
        cy = math.cos(init_yaw_rad * 0.5)
        sy = math.sin(init_yaw_rad * 0.5)
        self.q = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)

        # Sensor biases
        self.b_gyro = np.zeros(3, dtype=np.float64)
        self.b_accel = np.zeros(3, dtype=np.float64)

        # 15x15 Error-state covariance matrix
        # States: [pos (3), vel (3), att (3), b_g (3), b_a (3)]
        self.P = np.diag([
            0.1, 0.1, 0.1,      # Position (m^2)
            0.05, 0.05, 0.05,   # Velocity (m/s)^2
            0.01, 0.01, 0.01,   # Attitude tilt (rad^2)
            1e-4, 1e-4, 1e-4,   # Gyro bias (rad/s)^2
            1e-3, 1e-3, 1e-3,   # Accel bias (m/s^2)^2
        ])

        # Process Noise Covariance Q
        self.sigma_acc = 0.25      # m/s^2
        self.sigma_gyro = 0.015    # rad/s
        self.sigma_ba = 1e-4       # accel bias random walk
        self.sigma_bg = 1e-5       # gyro bias random walk

        self.default_source = DataSource.SITL
        self.flight_mode = FlightMode.POSITION_HOLD
        self.arming_state = ArmingState.ARMED
        self.health_status = HealthStatus.NOMINAL
        self.gps_fix_type = 3
        self.gps_satellites = 14

        self._seq = 0
        self._last_time = time.monotonic()
        self._last_accel_b = np.array([0.0, 0.0, -9.81])
        self._last_gyro_b = np.zeros(3)

    # ── Kinematic helpers ─────────────────────────────────────────────────────

    def _quat_to_rot(self) -> np.ndarray:
        """Body to NED direction cosine matrix R_body_to_ned."""
        q0, q1, q2, q3 = self.q
        return np.array([
            [1.0 - 2.0*(q2*q2 + q3*q3), 2.0*(q1*q2 - q0*q3),       2.0*(q1*q3 + q0*q2)],
            [2.0*(q1*q2 + q0*q3),       1.0 - 2.0*(q1*q1 + q3*q3), 2.0*(q2*q3 - q0*q1)],
            [2.0*(q1*q3 - q0*q2),       2.0*(q2*q3 + q0*q1),       1.0 - 2.0*(q1*q1 + q2*q2)],
        ])

    @staticmethod
    def _skew_symmetric(v: np.ndarray) -> np.ndarray:
        return np.array([
            [0.0,   -v[2],  v[1]],
            [v[2],   0.0,  -v[0]],
            [-v[1],  v[0],  0.0],
        ])

    # ── Predict step (IMU Propagation) ────────────────────────────────────────

    def predict(self, accel_meas: np.ndarray, gyro_meas: np.ndarray, dt: float) -> None:
        """High-rate strapdown propagation with IMU measurements."""
        if dt <= 0.0:
            return
        dt = min(0.1, dt)

        self._last_accel_b = accel_meas.copy()
        self._last_gyro_b = gyro_meas.copy()

        # Correct for estimated sensor biases
        f_b = accel_meas - self.b_accel
        omega_b = gyro_meas - self.b_gyro

        R = self._quat_to_rot()

        # Specific force to inertial acceleration (subtract gravity: gravity in NED is [0, 0, g])
        acc_inertial = R @ f_b + np.array([0.0, 0.0, 9.81])

        # State propagation (Trapezoidal / Euler integration)
        self.p += self.v * dt + 0.5 * acc_inertial * (dt ** 2)
        self.v += acc_inertial * dt

        # Quaternion propagation (zeroth-order integration)
        omega_norm = np.linalg.norm(omega_b)
        if omega_norm > 1e-8:
            axis = omega_b / omega_norm
            angle = omega_norm * dt
            delta_q = np.array([math.cos(angle * 0.5), *(axis * math.sin(angle * 0.5))])
        else:
            delta_q = np.array([1.0, 0.5 * omega_b[0] * dt, 0.5 * omega_b[1] * dt, 0.5 * omega_b[2] * dt])

        # Quaternion multiplication: self.q = self.q (x) delta_q
        q0, q1, q2, q3 = self.q
        d0, d1, d2, d3 = delta_q
        self.q = np.array([
            q0*d0 - q1*d1 - q2*d2 - q3*d3,
            q0*d1 + q1*d0 + q2*d3 - q3*d2,
            q0*d2 - q1*d3 + q2*d0 + q3*d1,
            q0*d3 + q1*d2 - q2*d1 + q3*d0,
        ])
        norm_q = np.linalg.norm(self.q)
        if norm_q > 1e-10:
            self.q /= norm_q

        # Error-state Jacobian F (15x15)
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -R @ self._skew_symmetric(f_b) * dt
        F[3:6, 12:15] = -R * dt
        F[6:9, 6:9] = np.eye(3) - self._skew_symmetric(omega_b) * dt
        F[6:9, 9:12] = -np.eye(3) * dt

        # Process noise covariance Q (15x15)
        Q = np.zeros((15, 15))
        q_pos = 0.5 * (self.sigma_acc ** 2) * (dt ** 3)
        q_vel = (self.sigma_acc ** 2) * dt
        q_att = (self.sigma_gyro ** 2) * dt
        q_ba  = (self.sigma_ba ** 2) * dt
        q_bg  = (self.sigma_bg ** 2) * dt

        Q[0:3, 0:3] = np.eye(3) * q_pos
        Q[3:6, 3:6] = np.eye(3) * q_vel
        Q[6:9, 6:9] = np.eye(3) * q_att
        Q[9:12, 9:12] = np.eye(3) * q_bg
        Q[12:15, 12:15] = np.eye(3) * q_ba

        # Covariance propagation
        self.P = F @ self.P @ F.T + Q

    # ── Update Steps ──────────────────────────────────────────────────────────

    def update_gps(
        self,
        pos_ned_gps: np.ndarray,
        vel_ned_gps: np.ndarray,
        cov_pos: float = 0.25,
        cov_vel: float = 0.05,
        fix_type: int = 3,
        satellites: int = 14,
    ) -> None:
        """Update with GPS position and velocity fix."""
        self.gps_fix_type = fix_type
        self.gps_satellites = satellites

        # Measurement residual (6x1)
        z = np.concatenate([pos_ned_gps, vel_ned_gps])
        h = np.concatenate([self.p, self.v])
        y = z - h

        # Measurement Jacobian H (6x15)
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)

        R_cov = np.diag([cov_pos, cov_pos, cov_pos * 1.5, cov_vel, cov_vel, cov_vel * 1.2])

        self._apply_kalman_update(H, y, R_cov)

    def update_barometer(self, alt_baro_m: float, cov_alt: float = 0.1) -> None:
        """Update with Barometric altitude (NED z = -alt)."""
        z_meas = -alt_baro_m
        y = np.array([z_meas - self.p[2]])

        H = np.zeros((1, 15))
        H[0, 2] = 1.0

        R_cov = np.array([[cov_alt]])
        self._apply_kalman_update(H, y, R_cov)

    def update_magnetometer(self, mag_body: np.ndarray, declination_rad: float = 0.0, cov_mag: float = 0.08) -> None:
        """Correct heading with calibrated 3-axis magnetometer."""
        mag_norm = np.linalg.norm(mag_body)
        if mag_norm < 1e-4:
            return

        R = self._quat_to_rot()
        # Rotate magnetic field to world frame
        mag_ned = R @ (mag_body / mag_norm)
        # Measured yaw relative to magnetic North
        meas_yaw = math.atan2(mag_ned[1], mag_ned[0]) + declination_rad

        # Current estimated yaw
        q0, q1, q2, q3 = self.q
        est_yaw = math.atan2(2.0*(q1*q2 + q0*q3), 1.0 - 2.0*(q2*q2 + q3*q3))

        yaw_err = math.atan2(math.sin(meas_yaw - est_yaw), math.cos(meas_yaw - est_yaw))
        y = np.array([yaw_err])

        H = np.zeros((1, 15))
        H[0, 8] = 1.0  # yaw error state

        R_cov = np.array([[cov_mag]])
        self._apply_kalman_update(H, y, R_cov)

    def _apply_kalman_update(self, H: np.ndarray, y: np.ndarray, R_cov: np.ndarray) -> None:
        """Standard Kalman gain and error-state injection."""
        S = H @ self.P @ H.T + R_cov
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)

        dx = K @ y

        # Inject error states
        self.p += dx[0:3]
        self.v += dx[3:6]

        # Multiplicative attitude injection
        d_theta = dx[6:9]
        angle = np.linalg.norm(d_theta)
        if angle > 1e-8:
            axis = d_theta / angle
            dq = np.array([math.cos(angle * 0.5), *(axis * math.sin(angle * 0.5))])
        else:
            dq = np.array([1.0, 0.5 * d_theta[0], 0.5 * d_theta[1], 0.5 * d_theta[2]])

        q0, q1, q2, q3 = self.q
        d0, d1, d2, d3 = dq
        self.q = np.array([
            q0*d0 - q1*d1 - q2*d2 - q3*d3,
            q0*d1 + q1*d0 + q2*d3 - q3*d2,
            q0*d2 - q1*d3 + q2*d0 + q3*d1,
            q0*d3 + q1*d2 - q2*d1 + q3*d0,
        ])
        norm_q = np.linalg.norm(self.q)
        if norm_q > 1e-10:
            self.q /= norm_q

        # Bias updates
        self.b_gyro += dx[9:12]
        self.b_accel += dx[12:15]

        # Joseph form covariance update for numerical stability
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_cov @ K.T

    # ── State conversion ──────────────────────────────────────────────────────

    def get_state_vector(
        self,
        source: Optional[DataSource] = None,
        flight_mode: Optional[FlightMode] = None,
        arming_state: Optional[ArmingState] = None,
        health_status: Optional[HealthStatus] = None,
    ) -> DroneStateVector:
        """Construct authoritative DroneStateVector from current MEKF estimate."""
        self._seq += 1
        q0, q1, q2, q3 = self.q

        roll = math.atan2(2.0*(q0*q1 + q2*q3), 1.0 - 2.0*(q1*q1 + q2*q2))
        pitch = math.asin(max(-1.0, min(1.0, 2.0*(q0*q2 - q3*q1))))
        yaw = math.atan2(2.0*(q0*q3 + q1*q2), 1.0 - 2.0*(q2*q2 + q3*q3))

        omega_corrected = self._last_gyro_b - self.b_gyro
        f_corrected = self._last_accel_b - self.b_accel

        # Evaluate health status dynamically from covariance trace and GPS
        if health_status is not None:
            resolved_health = health_status
        else:
            cov_trace = float(np.trace(self.P[0:3, 0:3]))
            if cov_trace > 20.0 or not np.all(np.isfinite(self.P)):
                resolved_health = HealthStatus.CRITICAL
            elif cov_trace > 5.0 or (self.gps_fix_type > 0 and self.gps_fix_type < 3):
                resolved_health = HealthStatus.DEGRADED
            else:
                resolved_health = self.health_status

        resolved_source = source if source is not None else self.default_source
        resolved_mode = flight_mode if flight_mode is not None else self.flight_mode
        resolved_arm = arming_state if arming_state is not None else self.arming_state

        return DroneStateVector(
            vehicle_id=self.vehicle_id,
            sequence=self._seq,
            timestamp_wall=time.time(),
            timestamp_mono=time.monotonic(),
            timestamp_sim=0.0,
            source=resolved_source,
            flight_mode=resolved_mode,
            arming_state=resolved_arm,
            health_status=resolved_health,
            x=float(self.p[0]),
            y=float(self.p[1]),
            z=float(self.p[2]),
            vx=float(self.v[0]),
            vy=float(self.v[1]),
            vz=float(self.v[2]),
            ax=float(f_corrected[0]),
            ay=float(f_corrected[1]),
            az=float(f_corrected[2]),
            q0=float(q0),
            q1=float(q1),
            q2=float(q2),
            q3=float(q3),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            roll_rate=float(omega_corrected[0]),
            pitch_rate=float(omega_corrected[1]),
            yaw_rate=float(omega_corrected[2]),
            gps_fix_type=self.gps_fix_type,
            gps_satellites=self.gps_satellites,
            altitude_agl=float(max(0.0, -self.p[2])),
            is_valid=True,
        )
