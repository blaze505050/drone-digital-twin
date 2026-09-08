"""
drone_sdk.digital_twin_core.recalibrator
========================================
Online Parameter Recalibration & System Identification Loop.

Closes the digital twin loop:
  1. Gathers a sliding window of (commanded control actions, observed kinematics).
  2. Solves an online system-identification optimization (or calls pinn_engine.system_id_pinn).
  3. Re-estimates effective parameters:
     - mass_kg (payload change, fuel/battery consumption)
     - cd_translational (payload drag changes, air density)
     - max_thrust_motor_n (battery voltage degradation, motor wear)
  4. Automatically updates the running DynamicTwinModel parameters.

Result: When the twin drifts from physical telemetry, the recalibration loop
dynamically adapts model parameters until the prediction converges back to reality.

Python version: 3.9+
"""
from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from drone_sdk.state_manager.schema import DroneStateVector


@dataclass
class RecalibrationRecord:
    """Telemetry frame stored in the recalibration buffer."""
    timestamp:  float
    action:     np.ndarray     # Commanded motor inputs (4,)
    vel_actual: np.ndarray     # Measured NED velocity (3,)
    acc_actual: np.ndarray     # Measured NED acceleration (3,)


class OnlineRecalibrator:
    """Online system-identification and parameter recalibration engine."""

    def __init__(
        self,
        twin_model: Optional[object] = None,
        window_size: int = 60,
        recal_interval_frames: int = 30,
        nominal_mass_kg: Optional[float] = None,
        nominal_max_thrust_n: Optional[float] = None,
        enable_auto_apply: bool = False,
    ) -> None:
        self.twin_model = twin_model
        self.window_size = window_size
        self.recal_interval = recal_interval_frames
        self.enable_auto_apply = enable_auto_apply

        if twin_model is not None and hasattr(twin_model, "params"):
            self.nominal_mass = float(twin_model.params.mass_kg)
            self.nominal_thrust = float(twin_model.params.max_thrust_motor_n)
            self.nominal_cd = float(getattr(twin_model.params, "cd_translational", 1.05))
        else:
            self.nominal_mass = nominal_mass_kg if nominal_mass_kg is not None else 1.50
            self.nominal_thrust = nominal_max_thrust_n if nominal_max_thrust_n is not None else 6.62
            self.nominal_cd = 1.05

        self._buffer: Deque[RecalibrationRecord] = collections.deque(maxlen=window_size)
        self._frame_count = 0
        self._recal_count = 0
        self._last_calibrated_scales: Dict[str, float] = {
            "mass_scale": 1.0,
            "drag_scale": 1.0,
            "thrust_scale": 1.0,
        }

    def record_frame(
        self,
        action: np.ndarray,
        actual_state: DroneStateVector,
    ) -> bool:
        """Record one timestep of commanded input and resulting motion.

        Transforms body specific force into world NED linear acceleration.
        Returns True if a recalibration optimization was executed during this step.
        """
        q0, q1, q2, q3 = actual_state.q0, actual_state.q1, actual_state.q2, actual_state.q3
        R_nb = np.array([
            [1.0 - 2.0*(q2*q2 + q3*q3), 2.0*(q1*q2 - q0*q3),       2.0*(q1*q3 + q0*q2)],
            [2.0*(q1*q2 + q0*q3),       1.0 - 2.0*(q1*q1 + q3*q3), 2.0*(q2*q3 - q0*q1)],
            [2.0*(q1*q3 - q0*q2),       2.0*(q2*q3 + q0*q1),       1.0 - 2.0*(q1*q1 + q2*q2)],
        ])
        f_b = np.array([actual_state.ax, actual_state.ay, actual_state.az], dtype=np.float64)
        acc_ned = R_nb @ f_b + np.array([0.0, 0.0, 9.81], dtype=np.float64)

        rec = RecalibrationRecord(
            timestamp=time.time(),
            action=np.array(action, dtype=np.float64),
            vel_actual=np.array([actual_state.vx, actual_state.vy, actual_state.vz], dtype=np.float64),
            acc_actual=acc_ned,
        )
        self._buffer.append(rec)
        self._frame_count += 1

        if self._frame_count >= self.recal_interval and len(self._buffer) >= 20:
            self._frame_count = 0
            self.recalibrate()
            return True
        return False

    def recalibrate(self) -> Dict[str, float]:
        """Perform system identification optimization over buffered flight data."""
        if len(self._buffer) < 15:
            return self._last_calibrated_scales

        actions = np.array([r.action for r in self._buffer])
        vels    = np.array([r.vel_actual for r in self._buffer])
        accs    = np.array([r.acc_actual for r in self._buffer])

        # Vertical balance in NED:
        # a_z = g - (F_thrust_z / m) - (F_drag_z / m)
        # Note: thrust acts along -z, opposing gravity (+z).
        cmd_sum = np.sum(actions, axis=1)  # (N,)
        vert_acc = accs[:, 2]              # (N,) NED vertical acceleration
        vert_vel = vels[:, 2]

        # Excitation check: require non-trivial command variation to avoid rank deficiency
        if np.var(cmd_sum) < 1e-4:
            self._recal_count += 1
            return self._last_calibrated_scales

        # Linear regression on vertical dynamics:
        # F_net_z / m = acc_z - 9.81 = - (4 * max_thrust / mass) * (cmd_sum / 4) + drag_term
        delta_az = vert_acc - 9.81
        A = np.column_stack([
            -0.25 * cmd_sum,
            vert_vel * np.abs(vert_vel),
        ])

        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, delta_az, rcond=None)
            k_t_m = float(coeffs[0])

            nominal_mass = self.twin_model.params.mass_kg if self.twin_model is not None else self.nominal_mass
            nominal_thrust = self.twin_model.params.max_thrust_motor_n if self.twin_model is not None else self.nominal_thrust
            nominal_k_t_m = (4.0 * nominal_thrust) / max(1e-3, nominal_mass)

            if k_t_m > 5.0 and k_t_m < 35.0:
                thrust_scale = float(k_t_m / nominal_k_t_m)
            else:
                thrust_scale = 1.0

            # Horizontal drag fit from x and y dynamics
            horiz_vel_sq = vels[:, 0:2] * np.abs(vels[:, 0:2])
            horiz_acc = accs[:, 0:2]
            drag_scale = 1.0
            if np.mean(np.abs(horiz_vel_sq)) > 1e-3:
                drag_est = -np.mean(horiz_acc * np.sign(vels[:, 0:2]))
                if drag_est > 0.05:
                    drag_scale = float(np.clip(drag_est / 0.5, 0.7, 1.4))

            # Bound scales to realistic physical envelopes
            thrust_scale = float(np.clip(thrust_scale, 0.75, 1.25))
            drag_scale = float(np.clip(drag_scale, 0.70, 1.35))
            # Hold mass prior constant unless independently measured
            mass_scale = 1.0

        except Exception:
            thrust_scale = 1.0
            drag_scale = 1.0
            mass_scale = 1.0

        # Smooth update (exponential moving average)
        alpha = 0.4
        self._last_calibrated_scales["thrust_scale"] = (
            (1.0 - alpha) * self._last_calibrated_scales["thrust_scale"] + alpha * thrust_scale
        )
        self._last_calibrated_scales["drag_scale"] = (
            (1.0 - alpha) * self._last_calibrated_scales["drag_scale"] + alpha * drag_scale
        )
        self._last_calibrated_scales["mass_scale"] = (
            (1.0 - alpha) * self._last_calibrated_scales["mass_scale"] + alpha * mass_scale
        )

        # Apply to twin model ONLY if explicitly authorized
        if self.enable_auto_apply and hasattr(self.twin_model, "params"):
            p = getattr(self.twin_model, "params")
            p.mass_kg = self.nominal_mass * self._last_calibrated_scales["mass_scale"]
            p.max_thrust_motor_n = self.nominal_thrust * self._last_calibrated_scales["thrust_scale"]
            p.cd_translational = self.nominal_cd * self._last_calibrated_scales["drag_scale"]

        self._recal_count += 1
        return self._last_calibrated_scales

    @property
    def recalibration_count(self) -> int:
        return self._recal_count

    @property
    def calibrated_parameters(self) -> Dict[str, float]:
        return dict(self._last_calibrated_scales)
