"""
dronepy.twin
============
Digital Twin Synchronization, Live MEKF Prediction, and Twin vs Reality Diagnostics.
Computes tracking residuals, health index, and multi-timeline comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from drone_sdk.digital_twin_core.residual_monitor import TwinResidualMonitor
from drone_sdk.digital_twin_core.twin_model import DynamicTwinModel
from drone_sdk.state_manager.schema import DataSource, DroneStateUpdate, DroneStateVector, HealthStatus


@dataclass
class TwinComparison:
    """Rigorous statistical and time-series comparison between Physical Reality and Digital Twin."""
    time: np.ndarray
    real_pos_ned: np.ndarray      # (N, 3)
    twin_pos_ned: np.ndarray      # (N, 3)
    real_vel_ned: np.ndarray      # (N, 3)
    twin_vel_ned: np.ndarray      # (N, 3)
    real_euler_deg: np.ndarray    # (N, 3)
    twin_euler_deg: np.ndarray    # (N, 3)
    real_energy_wh: Optional[np.ndarray] = None
    twin_energy_wh: Optional[np.ndarray] = None

    @classmethod
    def compare(cls, real: Any, twin: Any) -> "TwinComparison":
        """Construct a TwinComparison directly from two FlightResult objects."""
        n = min(len(real.time), len(twin.time))
        return cls(
            time=real.time[:n],
            real_pos_ned=real.pos_ned[:n],
            twin_pos_ned=twin.pos_ned[:n],
            real_vel_ned=real.vel_ned[:n],
            twin_vel_ned=twin.vel_ned[:n],
            real_euler_deg=real.euler_deg[:n],
            twin_euler_deg=twin.euler_deg[:n],
            real_energy_wh=getattr(real, "battery_energy_wh", None)[:n] if getattr(real, "battery_energy_wh", None) is not None else None,
            twin_energy_wh=getattr(twin, "battery_energy_wh", None)[:n] if getattr(twin, "battery_energy_wh", None) is not None else None,
        )

    @property
    def position_residuals(self) -> np.ndarray:
        """Euclidean 3D position error over time (metres)."""
        return np.linalg.norm(self.real_pos_ned - self.twin_pos_ned, axis=1)

    @property
    def velocity_residuals(self) -> np.ndarray:
        """Velocity error over time (m/s)."""
        return np.linalg.norm(self.real_vel_ned - self.twin_vel_ned, axis=1)

    @property
    def attitude_residuals(self) -> np.ndarray:
        """Angular attitude divergence over time (degrees)."""
        diff = np.abs(self.real_euler_deg - self.twin_euler_deg)
        # Wrap around 360
        diff = np.where(diff > 180.0, 360.0 - diff, diff)
        return np.linalg.norm(diff, axis=1)

    @property
    def rmse_position(self) -> float:
        """Root Mean Square Error for 3D position (m)."""
        return float(np.sqrt(np.mean(self.position_residuals ** 2)))

    @property
    def rmse_velocity(self) -> float:
        """Root Mean Square Error for velocity (m/s)."""
        return float(np.sqrt(np.mean(self.velocity_residuals ** 2)))

    @property
    def max_position_error(self) -> float:
        return float(np.max(self.position_residuals))

    def plot(self, show: bool = True) -> Any:
        """Plot REAL vs TWIN trajectories and RESIDUAL on the same timeline."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for TwinComparison.plot().")

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

        # 1. Altitude (Z) Comparison
        ax1.plot(self.time, -self.real_pos_ned[:, 2], label="REAL Flight", color="#00ff88", linewidth=1.8)
        ax1.plot(self.time, -self.twin_pos_ned[:, 2], label="TWIN Prediction", color="#00d4ff", linestyle="--", linewidth=1.8)
        ax1.set_ylabel("Altitude AGL (m)")
        ax1.set_title("Reality vs Digital Twin Comparison")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")

        # 2. Velocity Comparison
        real_speed = np.linalg.norm(self.real_vel_ned, axis=1)
        twin_speed = np.linalg.norm(self.twin_vel_ned, axis=1)
        ax2.plot(self.time, real_speed, label="REAL Speed", color="#00ff88", linewidth=1.6)
        ax2.plot(self.time, twin_speed, label="TWIN Speed", color="#00d4ff", linestyle="--", linewidth=1.6)
        ax2.set_ylabel("Airspeed (m/s)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")

        # 3. Position Residual
        ax3.plot(self.time, self.position_residuals, label=f"Tracking Residual (RMSE={self.rmse_position*100:.1f} cm)", color="#ff4757", linewidth=2.0)
        ax3.axhline(0.60, color="#ff8a3d", linestyle=":", label="Degraded Threshold (60cm)")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Position Error (m)")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper right")

        fig.tight_layout()
        if show:
            plt.show()
        return fig


class DigitalTwinSynchronizer:
    """Real-time parallel physics predictor tracking incoming physical drone states."""

    def __init__(self, vehicle_id: str = "drone_0") -> None:
        self.vehicle_id = vehicle_id
        self.twin_model = DynamicTwinModel(vehicle_id=vehicle_id)
        self.residual_monitor = TwinResidualMonitor()

    def step(self, real_state: DroneStateVector, dt: float = 0.01) -> Tuple[DroneStateVector, Any]:
        """Advance digital twin prediction and evaluate residual against physical state.

        Returns
        -------
        Tuple[predicted_state, residual_report]
        """
        max_omega = 1200.0
        cmd_action = np.array([
            min(1.0, real_state.omega1 / max_omega),
            min(1.0, real_state.omega2 / max_omega),
            min(1.0, real_state.omega3 / max_omega),
            min(1.0, real_state.omega4 / max_omega),
        ], dtype=np.float64)
        if np.max(cmd_action) <= 0.01:
            cmd_action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

        # Step parallel model
        predicted_state = self.twin_model.step(cmd_action=cmd_action, dt=dt)
        # Evaluate tracking discrepancy
        report = self.residual_monitor.update(real_state=real_state, twin_state=predicted_state)
        return predicted_state, report
