"""
dronepy.results
===============
Structured Flight Result Container for DronePy Simulations.
Provides seamless Pandas DataFrame conversion, CSV/HDF5 export, and plotting delegation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class FlightResult:
    """Comprehensive time-series container of multirotor 6-DOF simulation results."""

    time: np.ndarray
    # Position in NED (metres)
    pos_ned: np.ndarray                 # (N, 3)
    altitude_agl: np.ndarray             # (N,)
    # Velocity (m/s)
    vel_ned: np.ndarray                 # (N, 3)
    vel_body: np.ndarray                # (N, 3)
    airspeed: np.ndarray                # (N,)
    # Acceleration (m/s^2)
    accel_body: np.ndarray              # (N, 3)
    # Attitude
    quaternion: np.ndarray              # (N, 4) [qw, qx, qy, qz]
    euler_rad: np.ndarray               # (N, 3) [roll, pitch, yaw]
    euler_deg: np.ndarray               # (N, 3) [roll, pitch, yaw]
    # Angular velocity (rad/s)
    omega_body: np.ndarray              # (N, 3) [p, q, r]
    # Propulsion
    motor_rpms: np.ndarray              # (N, num_motors)
    motor_thrusts: np.ndarray           # (N, num_motors)
    total_thrust: np.ndarray            # (N,)
    # Aerodynamic Forces & Moments
    forces_body: np.ndarray             # (N, 3)
    moments_body: np.ndarray            # (N, 3)
    drag_body: np.ndarray               # (N, 3)
    # Electrical & Battery
    battery_voltage: np.ndarray         # (N,)
    battery_current: np.ndarray         # (N,)
    battery_power: np.ndarray           # (N,)
    battery_energy_wh: np.ndarray       # (N,)
    battery_soc: np.ndarray             # (N,)
    # Controls
    motor_commands: np.ndarray          # (N, num_motors)
    # Event Log
    events_log: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Pandas & Tabular Export ───────────────────────────────────────────────

    def to_dataframe(self) -> Any:
        """Convert time-series flight telemetry into a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas")

        data: Dict[str, np.ndarray] = {
            "time_s": self.time,
            "x_north_m": self.pos_ned[:, 0],
            "y_east_m": self.pos_ned[:, 1],
            "z_down_m": self.pos_ned[:, 2],
            "altitude_agl_m": self.altitude_agl,
            "vx_mps": self.vel_ned[:, 0],
            "vy_mps": self.vel_ned[:, 1],
            "vz_mps": self.vel_ned[:, 2],
            "airspeed_mps": self.airspeed,
            "ax_body_mps2": self.accel_body[:, 0],
            "ay_body_mps2": self.accel_body[:, 1],
            "az_body_mps2": self.accel_body[:, 2],
            "roll_deg": self.euler_deg[:, 0],
            "pitch_deg": self.euler_deg[:, 1],
            "yaw_deg": self.euler_deg[:, 2],
            "p_rad_s": self.omega_body[:, 0],
            "q_rad_s": self.omega_body[:, 1],
            "r_rad_s": self.omega_body[:, 2],
            "total_thrust_n": self.total_thrust,
            "battery_v": self.battery_voltage,
            "battery_a": self.battery_current,
            "battery_power_w": self.battery_power,
            "energy_wh": self.battery_energy_wh,
            "battery_soc": self.battery_soc,
        }

        # Add motor columns dynamically
        num_motors = self.motor_rpms.shape[1]
        for i in range(num_motors):
            data[f"motor_{i+1}_rpm"] = self.motor_rpms[:, i]
            data[f"motor_{i+1}_thrust_n"] = self.motor_thrusts[:, i]
            data[f"motor_{i+1}_cmd"] = self.motor_commands[:, i]

        return pd.DataFrame(data)

    def to_csv(self, filepath: Union[str, Path]) -> None:
        """Export flight results to a CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

    def to_hdf5(self, filepath: Union[str, Path]) -> None:
        """Export telemetry to an HDF5 binary archive."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for to_hdf5(). Install with: pip install h5py")

        with h5py.File(filepath, "w") as hf:
            hf.create_dataset("time", data=self.time)
            hf.create_dataset("pos_ned", data=self.pos_ned)
            hf.create_dataset("vel_ned", data=self.vel_ned)
            hf.create_dataset("accel_body", data=self.accel_body)
            hf.create_dataset("quaternion", data=self.quaternion)
            hf.create_dataset("euler_deg", data=self.euler_deg)
            hf.create_dataset("omega_body", data=self.omega_body)
            hf.create_dataset("motor_rpms", data=self.motor_rpms)
            hf.create_dataset("motor_thrusts", data=self.motor_thrusts)
            hf.create_dataset("total_thrust", data=self.total_thrust)
            hf.create_dataset("battery_voltage", data=self.battery_voltage)
            hf.create_dataset("battery_power", data=self.battery_power)
            hf.create_dataset("battery_energy_wh", data=self.battery_energy_wh)
            hf.create_dataset("battery_soc", data=self.battery_soc)
            hf.attrs["metadata"] = json.dumps(self.metadata)

    # ── High-Level Plotting Delegations ───────────────────────────────────────

    def plot_trajectory(self, **kwargs: Any) -> Any:
        from .visualization import plot_trajectory
        return plot_trajectory(self, **kwargs)

    def plot_attitude(self, **kwargs: Any) -> Any:
        from .visualization import plot_attitude
        return plot_attitude(self, **kwargs)

    def plot_velocity(self, **kwargs: Any) -> Any:
        from .visualization import plot_velocity
        return plot_velocity(self, **kwargs)

    def plot_motor_rpm(self, **kwargs: Any) -> Any:
        from .visualization import plot_motor_rpm
        return plot_motor_rpm(self, **kwargs)

    def plot_power(self, **kwargs: Any) -> Any:
        from .visualization import plot_power
        return plot_power(self, **kwargs)

    def plot_energy(self, **kwargs: Any) -> Any:
        from .visualization import plot_energy
        return plot_energy(self, **kwargs)

    def plot_forces(self, **kwargs: Any) -> Any:
        from .visualization import plot_forces
        return plot_forces(self, **kwargs)

    def plot_moments(self, **kwargs: Any) -> Any:
        from .visualization import plot_moments
        return plot_moments(self, **kwargs)

    def plot_all(self, **kwargs: Any) -> Any:
        from .visualization import plot_dashboard
        return plot_dashboard(self, **kwargs)
