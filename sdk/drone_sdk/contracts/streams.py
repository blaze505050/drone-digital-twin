"""
contracts.streams
=================
Typed Stream Contracts for Telemetry, State, Predictions, and Actuation.

Provides strictly typed dataclass schemas with provenance status tags, ensuring
clear separation between:
1. Raw Sensor Ingestion (SensorMeasurement)
2. Authoritative Estimated States (StateEstimate)
3. Dynamic Twin Model Predictions (TwinPrediction)
4. Truth References (ReferenceTruth)
5. Commanded Intent (CommandIntent)
6. Applied Actuation (AppliedActuation)

Python version: 3.9+
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from drone_sdk.contracts.data_status import DataStatus


@dataclass
class SensorMeasurement:
    """Raw sensor packet from physical hardware or simulated sensor model."""

    vehicle_id: str
    sensor_type: str  # "imu", "gps", "mag", "baro", "optical_flow", "lidar"
    timestamp_mono: float = field(default_factory=time.monotonic)
    timestamp_utc: float = field(default_factory=time.time)
    status: DataStatus = DataStatus.SYNTHETIC
    frame: str = "FRD"  # "FRD", "NED", "GEODETIC", "BODY"
    values: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    covariance: Optional[np.ndarray] = None
    metadata: Dict[str, float] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return self.status != DataStatus.INVALID and np.all(np.isfinite(self.values))


@dataclass
class StateEstimate:
    """Authoritative estimated UAV state produced by the MEKF estimator."""

    vehicle_id: str
    sequence: int
    timestamp_mono: float
    pos_ned: np.ndarray        # [x, y, z] in metres
    vel_ned: np.ndarray        # [vx, vy, vz] in m/s
    att_quat: np.ndarray       # [qw, qx, qy, qz] Hamilton unit quaternion
    rate_body: np.ndarray      # [p, q, r] in rad/s
    bias_gyro: np.ndarray      # [bg_x, bg_y, bg_z] in rad/s
    bias_accel: np.ndarray     # [ba_x, ba_y, ba_z] in m/s^2
    covariance_diag: np.ndarray  # 15-state covariance diagonals
    status: DataStatus = DataStatus.SYNTHETIC
    cov_trace: float = 0.0

    @property
    def is_healthy(self) -> bool:
        return self.status != DataStatus.INVALID and math.isfinite(self.cov_trace) and self.cov_trace < 25.0


@dataclass
class TwinPrediction:
    """Predicted vehicle state advanced by the parallel 6-DOF dynamic twin model."""

    vehicle_id: str
    sequence: int
    timestamp_sim: float
    pos_ned: np.ndarray
    vel_ned: np.ndarray
    att_quat: np.ndarray
    rate_body: np.ndarray
    rotor_speeds: np.ndarray
    applied_thrust_n: float
    status: DataStatus = DataStatus.SYNTHETIC
    dt_step: float = 0.02


@dataclass
class ReferenceTruth:
    """High-accuracy ground truth from external motion capture (Vicon/OptiTrack) or RTK-GPS."""

    vehicle_id: str
    timestamp_mono: float
    pos_ned: np.ndarray
    vel_ned: np.ndarray
    att_quat: np.ndarray
    rate_body: np.ndarray
    source_device: str = "vicon"
    status: DataStatus = DataStatus.REAL


@dataclass
class CommandIntent:
    """Commanded control intent, setpoints, or waypoint targets."""

    vehicle_id: str
    timestamp_mono: float = field(default_factory=time.monotonic)
    flight_mode: str = "POSITION_HOLD"
    target_pos_ned: Optional[np.ndarray] = None
    target_vel_ned: Optional[np.ndarray] = None
    target_yaw_rad: Optional[float] = None
    target_thrust: float = 0.0
    armed: bool = False


@dataclass
class AppliedActuation:
    """Effective actuator outputs dispatched to motor ESCs or physics simulation."""

    vehicle_id: str
    timestamp_mono: float = field(default_factory=time.monotonic)
    motor_signals: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))  # Normalized [0, 1]
    motor_rpms: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    total_thrust_n: float = 0.0
    torques_body_nm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
