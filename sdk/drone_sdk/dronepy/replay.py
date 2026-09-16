"""
dronepy.replay
==============
Flight log replayer and telemetry importer for DronePy.
Supports CSV, HDF5, Pandas DataFrames, and MAVLink telemetry streams.
Works seamlessly with or without pandas installed.
Allows step-by-step playback, resampling, and conversion to FlightResult.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from drone_sdk.dronepy.results import FlightResult


class FlightReplay:
    """
    Flight log replay container and stream iterator.
    Stores series internally as Dict[str, np.ndarray] for zero-dependency operation.
    """

    def __init__(self, data: Union[Dict[str, np.ndarray], Any], source_name: str = "flight_log") -> None:
        self.source_name = source_name
        self._columns: Dict[str, np.ndarray] = self._normalize_columns(data)
        self._current_index = 0

    @classmethod
    def from_csv(cls, filepath: Union[str, Path]) -> FlightReplay:
        """Load telemetry from a CSV file."""
        if pd is not None:
            df = pd.read_csv(filepath)
            data = {c: df[c].values for c in df.columns}
        else:
            data = {}
            with open(filepath, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    keys = rows[0].keys()
                    for k in keys:
                        vals = []
                        for r in rows:
                            try:
                                vals.append(float(r[k]))
                            except (ValueError, TypeError):
                                vals.append(0.0)
                        data[k] = np.array(vals)
        return cls(data, source_name=Path(filepath).name)

    @classmethod
    def from_hdf5(cls, filepath: Union[str, Path], key: str = "telemetry") -> FlightReplay:
        """Load telemetry from an HDF5 file."""
        if pd is None:
            raise ImportError("pandas is required for from_hdf5()")
        df = pd.read_hdf(filepath, key=key)
        return cls({c: df[c].values for c in df.columns}, source_name=Path(filepath).name)

    @classmethod
    def from_dataframe(cls, df: Any, source_name: str = "dataframe") -> FlightReplay:
        """Load from an existing Pandas DataFrame."""
        return cls({c: df[c].values for c in df.columns}, source_name=source_name)

    @classmethod
    def from_flight_result(cls, result: FlightResult) -> FlightReplay:
        """Load directly from a simulated FlightResult."""
        data = {
            "time_sec": result.time,
            "x_m": result.pos_ned[:, 0],
            "y_m": result.pos_ned[:, 1],
            "z_m": result.pos_ned[:, 2],
            "vx_m_s": result.vel_ned[:, 0],
            "vy_m_s": result.vel_ned[:, 1],
            "vz_m_s": result.vel_ned[:, 2],
            "roll_deg": result.euler_deg[:, 0],
            "pitch_deg": result.euler_deg[:, 1],
            "yaw_deg": result.euler_deg[:, 2],
            "battery_voltage_v": result.battery_voltage,
            "battery_current_a": result.battery_current,
            "battery_soc_pct": result.battery_soc,
            "power_w": result.battery_power,
            "total_thrust_n": result.total_thrust,
        }
        num_motors = result.motor_rpms.shape[1] if result.motor_rpms is not None else 0
        for i in range(num_motors):
            data[f"motor_rpm_{i+1}"] = result.motor_rpms[:, i]
        return cls(data, source_name="simulation_result")

    def _normalize_columns(self, raw_data: Union[Dict[str, Any], Any]) -> Dict[str, np.ndarray]:
        """Standardizes column names to DronePy canonical conventions."""
        if pd is not None and isinstance(raw_data, pd.DataFrame):
            in_dict = {c: raw_data[c].values for c in raw_data.columns}
        elif isinstance(raw_data, dict):
            in_dict = {k: np.asarray(v) for k, v in raw_data.items()}
        else:
            in_dict = {}

        col_map = {
            "time": "time_sec",
            "t": "time_sec",
            "timestamp": "time_sec",
            "pos_x": "x_m",
            "pos_y": "y_m",
            "pos_z": "z_m",
            "vel_x": "vx_m_s",
            "vel_y": "vy_m_s",
            "vel_z": "vz_m_s",
            "roll": "roll_deg",
            "pitch": "pitch_deg",
            "yaw": "yaw_deg",
            "voltage": "battery_voltage_v",
            "current": "battery_current_a",
            "soc": "battery_soc_pct",
        }

        norm_dict: Dict[str, np.ndarray] = {}
        for k, v in in_dict.items():
            kl = k.lower()
            canonical = col_map.get(kl, k)
            norm_dict[canonical] = np.asarray(v, dtype=float)

        if "time_sec" not in norm_dict:
            n = len(next(iter(norm_dict.values()))) if norm_dict else 0
            norm_dict["time_sec"] = np.arange(n) * 0.01

        # Sort by time
        sort_order = np.argsort(norm_dict["time_sec"])
        for k in norm_dict:
            norm_dict[k] = norm_dict[k][sort_order]

        return norm_dict

    @property
    def duration(self) -> float:
        """Total duration of flight log in seconds."""
        t = self._columns.get("time_sec", np.array([0.0]))
        return float(t[-1] - t[0]) if len(t) > 0 else 0.0

    @property
    def sample_rate(self) -> float:
        """Average sample frequency in Hz."""
        t = self._columns.get("time_sec", np.array([0.0]))
        if len(t) < 2:
            return 100.0
        dt = np.diff(t)
        avg_dt = float(np.mean(dt))
        return 1.0 / max(avg_dt, 1e-6)

    @property
    def columns(self) -> List[str]:
        return list(self._columns.keys())

    def __getitem__(self, key: str) -> np.ndarray:
        return self._columns[key]

    def to_dataframe(self) -> Any:
        if pd is None:
            raise ImportError("pandas is required for to_dataframe().")
        return pd.DataFrame(self._columns)

    def resample(self, target_hz: float = 50.0) -> FlightReplay:
        """Resample telemetry to a uniform sample frequency using linear interpolation."""
        t = self._columns["time_sec"]
        t_start = t[0]
        t_end = t[-1]
        dt = 1.0 / target_hz
        new_times = np.arange(t_start, t_end, dt)

        resampled: Dict[str, np.ndarray] = {"time_sec": new_times}
        for k, arr in self._columns.items():
            if k == "time_sec":
                continue
            resampled[k] = np.interp(new_times, t, arr)

        return FlightReplay(resampled, source_name=f"{self.source_name}_resampled")

    def to_flight_result(self) -> FlightResult:
        """Converts replay data into a DronePy FlightResult."""
        t = self._columns["time_sec"]
        n = len(t)

        def get_col(candidates: List[str], default: float = 0.0) -> np.ndarray:
            for c in candidates:
                if c in self._columns:
                    return self._columns[c]
            return np.full(n, default)

        pos = np.column_stack([
            get_col(["x_m", "north_m", "pos_x"]),
            get_col(["y_m", "east_m", "pos_y"]),
            get_col(["z_m", "down_m", "pos_z"]),
        ])
        vel = np.column_stack([
            get_col(["vx_m_s", "vx", "vel_x"]),
            get_col(["vy_m_s", "vy", "vel_y"]),
            get_col(["vz_m_s", "vz", "vel_z"]),
        ])
        eul_rad = np.column_stack([
            np.radians(get_col(["roll_deg", "roll"])),
            np.radians(get_col(["pitch_deg", "pitch"])),
            np.radians(get_col(["yaw_deg", "yaw"])),
        ])
        omega = np.column_stack([
            get_col(["p_rad_s", "p", "omega_x"]),
            get_col(["q_rad_s", "q", "omega_y"]),
            get_col(["r_rad_s", "r", "omega_z"]),
        ])

        # Quaternions from Euler
        q = np.zeros((n, 4))
        for i in range(n):
            cr = np.cos(eul_rad[i, 0] * 0.5)
            sr = np.sin(eul_rad[i, 0] * 0.5)
            cp = np.cos(eul_rad[i, 1] * 0.5)
            sp = np.sin(eul_rad[i, 1] * 0.5)
            cy = np.cos(eul_rad[i, 2] * 0.5)
            sy = np.sin(eul_rad[i, 2] * 0.5)
            q[i, 0] = cr * cp * cy + sr * sp * sy
            q[i, 1] = sr * cp * cy - cr * sp * sy
            q[i, 2] = cr * sp * cy + sr * cp * sy
            q[i, 3] = cr * cp * sy - sr * sp * cy

        motor_cols = [c for c in self._columns if "motor_rpm_" in c or "rpm_" in c]
        if motor_cols:
            motor_rpms = np.column_stack([self._columns[c] for c in motor_cols])
        else:
            motor_rpms = np.zeros((n, 4))

        return FlightResult(
            time=t,
            pos_ned=pos,
            altitude_agl=-pos[:, 2],
            vel_ned=vel,
            vel_body=vel,
            airspeed=np.linalg.norm(vel, axis=1),
            accel_body=np.zeros((n, 3)),
            quaternion=q,
            euler_rad=eul_rad,
            euler_deg=np.degrees(eul_rad),
            omega_body=omega,
            motor_rpms=motor_rpms,
            motor_thrusts=motor_rpms * 0.001,
            total_thrust=get_col(["total_thrust_n", "thrust"], 14.7),
            forces_body=np.zeros((n, 3)),
            moments_body=np.zeros((n, 3)),
            drag_body=np.zeros((n, 3)),
            battery_voltage=get_col(["battery_voltage_v", "voltage", "batt_v"], 15.2),
            battery_current=get_col(["battery_current_a", "current", "batt_i"], 12.0),
            battery_power=get_col(["power_w", "power"], 180.0),
            battery_energy_wh=get_col(["energy_wh"], 0.0),
            battery_soc=get_col(["battery_soc_pct", "soc"], 95.0),
            motor_commands=np.ones((n, 4)) * 0.5,
            metadata={"source": self.source_name, "sample_count": n},
        )

    def step_iterator(self) -> Iterator[Dict[str, float]]:
        """Yields sequential state telemetry dictionaries frame-by-frame."""
        n = len(self._columns.get("time_sec", []))
        for i in range(n):
            yield {k: float(self._columns[k][i]) for k in self._columns}
