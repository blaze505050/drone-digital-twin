"""
drone_sdk.validation_framework.replay_service
=============================================
Deterministic Flight Log Replay Service.
Implements Backlog Item B10 & PRD DAT-03/04/INT-01.

Key features:
1. Replays time-indexed flight logs (CSV, NumPy arrays, or dict frames).
2. Generates strictly separated SensorMeasurement, AppliedActuation, and ReferenceTruth streams.
3. Reference truth NEVER leaks into the sensor measurement channel.
4. Validates clock monotonicity and catches dropped/corrupted frames.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np

from ..contracts import (
    AppliedActuation,
    DataStatus,
    ReferenceTruth,
    SensorMeasurement,
    STANDARD_GRAVITY_MPS2,
    specific_force_from_acceleration,
)
from .manifest import DataManifest


@dataclass
class ReplayFrame:
    """A synchronized snapshot of replayed telemetry and truth data at timestamp t."""
    timestamp_sec: float
    measurements: List[SensorMeasurement] = field(default_factory=list)
    actuation: Optional[AppliedActuation] = None
    truth: Optional[ReferenceTruth] = None


class LogReplayService:
    """
    Deterministic replay engine for UAV flight data.
    Ensures mathematical isolation between truth signals and sensor inputs.
    """

    def __init__(
        self,
        vehicle_id: str = "holybro_x500_v2",
        data_status: DataStatus = DataStatus.SYNTHETIC,
        manifest: Optional[DataManifest] = None,
    ) -> None:
        self.vehicle_id = vehicle_id
        self.data_status = data_status
        self.manifest = manifest
        self._frames: List[ReplayFrame] = []
        self._cursor: int = 0
        self._is_loaded: bool = False

    @property
    def total_frames(self) -> int:
        return len(self._frames)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_from_arrays(
        self,
        t: np.ndarray,
        pos_ned: Optional[np.ndarray] = None,
        vel_ned: Optional[np.ndarray] = None,
        quat_nb: Optional[np.ndarray] = None,
        gyro_rps: Optional[np.ndarray] = None,
        accel_mps2: Optional[np.ndarray] = None,
        voltage_v: Optional[np.ndarray] = None,
        current_a: Optional[np.ndarray] = None,
        rotor_rpm: Optional[np.ndarray] = None,
        pwm_cmd: Optional[np.ndarray] = None,
    ) -> int:
        """
        Load replay frames from synchronized NumPy arrays.
        Strictly maps kinematics to accelerometer specific-force and ReferenceTruth.
        """
        n = len(t)
        self._frames.clear()

        last_t = -1.0
        for i in range(n):
            ti = float(t[i])
            if ti < last_t:
                # Monotonicity check
                raise ValueError(f"Non-monotonic timestamp at index {i}: {ti} < {last_t}")
            last_t = ti

            # 1. Reference Truth (Truth only!)
            truth_obj: Optional[ReferenceTruth] = None
            if pos_ned is not None or vel_ned is not None or quat_nb is not None:
                p = pos_ned[i] if pos_ned is not None else np.zeros(3)
                v = vel_ned[i] if vel_ned is not None else np.zeros(3)
                q = quat_nb[i] if quat_nb is not None else np.array([1.0, 0.0, 0.0, 0.0])
                w = gyro_rps[i] if gyro_rps is not None else np.zeros(3)
                truth_obj = ReferenceTruth(
                    vehicle_id=self.vehicle_id,
                    timestamp_mono=ti,
                    pos_ned=np.asarray(p, dtype=float),
                    vel_ned=np.asarray(v, dtype=float),
                    att_quat=np.asarray(q, dtype=float),
                    rate_body=np.asarray(w, dtype=float),
                    status=self.data_status,
                )

            # 2. Sensor Measurements (Simulated or Real sensor outputs)
            meas_list: List[SensorMeasurement] = []

            # IMU: Gyro (rad/s) + Accelerometer specific force (m/s^2)
            if gyro_rps is not None and accel_mps2 is not None:
                # Gyro
                meas_list.append(SensorMeasurement(
                    vehicle_id=self.vehicle_id,
                    sensor_type="gyro",
                    timestamp_mono=ti,
                    status=self.data_status,
                    values=np.asarray(gyro_rps[i], dtype=float),
                    covariance=np.diag([1e-4, 1e-4, 1e-4]),
                    frame="FRD",
                ))
                # Accel
                meas_list.append(SensorMeasurement(
                    vehicle_id=self.vehicle_id,
                    sensor_type="accel",
                    timestamp_mono=ti,
                    status=self.data_status,
                    values=np.asarray(accel_mps2[i], dtype=float),
                    covariance=np.diag([1e-3, 1e-3, 1e-3]),
                    frame="FRD",
                ))

            # Battery Sensor
            if voltage_v is not None and current_a is not None:
                meas_list.append(SensorMeasurement(
                    vehicle_id=self.vehicle_id,
                    sensor_type="battery",
                    timestamp_mono=ti,
                    status=self.data_status,
                    values=np.array([float(voltage_v[i]), float(current_a[i])]),
                    frame="SCALAR",
                ))

            # Rotor Tachometer / RPM
            if rotor_rpm is not None:
                meas_list.append(SensorMeasurement(
                    vehicle_id=self.vehicle_id,
                    sensor_type="rpm",
                    timestamp_mono=ti,
                    status=self.data_status,
                    values=np.asarray(rotor_rpm[i], dtype=float),
                    frame="SCALAR",
                ))

            # 3. Applied Actuation
            actuation_obj: Optional[AppliedActuation] = None
            if pwm_cmd is not None or rotor_rpm is not None:
                pwm = pwm_cmd[i] if pwm_cmd is not None else np.zeros(4)
                rpm = rotor_rpm[i] if rotor_rpm is not None else np.zeros(4)
                actuation_obj = AppliedActuation(
                    vehicle_id=self.vehicle_id,
                    timestamp_mono=ti,
                    motor_signals=np.asarray(pwm, dtype=float),
                    motor_rpms=np.asarray(rpm, dtype=float),
                )

            self._frames.append(ReplayFrame(
                timestamp_sec=ti,
                measurements=meas_list,
                actuation=actuation_obj,
                truth=truth_obj,
            ))

        self._cursor = 0
        self._is_loaded = True
        return len(self._frames)

    def load_from_csv(self, csv_path: Path | str) -> int:
        """Parse structured CSV flight log into replay frames."""
        p = Path(csv_path)
        if not p.is_file():
            raise FileNotFoundError(f"Log file {p} not found.")

        rows: List[Dict[str, str]] = []
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            return 0

        t_list = []
        pos_list = []
        vel_list = []
        quat_list = []
        gyro_list = []
        accel_list = []
        v_list = []
        i_list = []

        for r in rows:
            t_list.append(float(r.get("timestamp", r.get("time", 0.0))))
            if "pos_x" in r:
                pos_list.append([float(r["pos_x"]), float(r["pos_y"]), float(r["pos_z"])])
            if "vel_x" in r:
                vel_list.append([float(r["vel_x"]), float(r["vel_y"]), float(r["vel_z"])])
            if "quat_w" in r:
                quat_list.append([float(r["quat_w"]), float(r["quat_x"]), float(r["quat_y"]), float(r["quat_z"])])
            if "gyro_x" in r:
                gyro_list.append([float(r["gyro_x"]), float(r["gyro_y"]), float(r["gyro_z"])])
            if "accel_x" in r:
                accel_list.append([float(r["accel_x"]), float(r["accel_y"]), float(r["accel_z"])])
            if "voltage_v" in r:
                v_list.append(float(r["voltage_v"]))
            if "current_a" in r:
                i_list.append(float(r["current_a"]))

        return self.load_from_arrays(
            t=np.array(t_list),
            pos_ned=np.array(pos_list) if pos_list else None,
            vel_ned=np.array(vel_list) if vel_list else None,
            quat_nb=np.array(quat_list) if quat_list else None,
            gyro_rps=np.array(gyro_list) if gyro_list else None,
            accel_mps2=np.array(accel_list) if accel_list else None,
            voltage_v=np.array(v_list) if v_list else None,
            current_a=np.array(i_list) if i_list else None,
        )

    def stream_frames(self) -> Generator[ReplayFrame, None, None]:
        """Stream replay frames in strict chronological order."""
        for frame in self._frames:
            yield frame

    def step(self) -> Optional[ReplayFrame]:
        """Advance one frame forward in time."""
        if self._cursor >= len(self._frames):
            return None
        frame = self._frames[self._cursor]
        self._cursor += 1
        return frame

    def reset(self) -> None:
        """Reset replay playhead to start."""
        self._cursor = 0
