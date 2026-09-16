"""
dronepy.controllers
===================
Flight Controllers for Multirotor UAVs in DronePy.
Provides cascaded PID position/velocity/attitude controllers and custom controller wrappers.
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from drone_sdk.contracts.coordinates import STANDARD_GRAVITY_MPS2
from drone_sdk.math_models.frames import quat_to_dcm, quat_to_euler
from drone_sdk.math_models.rigid_body import RigidBodyState


class Controller(abc.ABC):
    """Abstract base class for all DronePy flight controllers."""

    @abc.abstractmethod
    def compute(
        self,
        state: RigidBodyState,
        dt: float,
        target_pos_ned: Optional[np.ndarray] = None,
        target_yaw_rad: float = 0.0,
    ) -> np.ndarray:
        """Compute normalized motor commands [0.0, 1.0] for the multirotor.

        Parameters
        ----------
        state : RigidBodyState
            Current 6-DOF rigid body state.
        dt : float
            Timestep in seconds.
        target_pos_ned : np.ndarray, optional
            Target position in NED coordinates [x, y, z].
        target_yaw_rad : float
            Desired yaw angle in radians.

        Returns
        -------
        np.ndarray (num_motors,)
            Normalized motor throttle commands in range [0.0, 1.0].
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal integrator states."""
        pass


class CustomController(Controller):
    """Wraps an arbitrary user-defined Python function or callable.

    Example
    -------
    >>> def my_law(state, target_pos, target_yaw):
    ...     return np.array([0.55, 0.55, 0.55, 0.55])
    >>> ctrl = CustomController(my_law)
    """

    def __init__(self, func: Callable[..., np.ndarray], num_motors: int = 4) -> None:
        self.func = func
        self.num_motors = num_motors

    def compute(
        self,
        state: RigidBodyState,
        dt: float,
        target_pos_ned: Optional[np.ndarray] = None,
        target_yaw_rad: float = 0.0,
    ) -> np.ndarray:
        try:
            res = self.func(state, target_pos_ned, target_yaw_rad)
        except TypeError:
            res = self.func(state)
        return np.clip(np.asarray(res, dtype=np.float64), 0.0, 1.0)


@dataclass
class PIDGains:
    """PID gain coefficients with integral clamping."""
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    i_max: float = 5.0
    out_min: float = -100.0
    out_max: float = 100.0


class PIDAxis:
    """1D PID controller axis with anti-windup."""

    def __init__(self, gains: PIDGains) -> None:
        self.gains = gains
        self.integral: float = 0.0
        self.prev_error: float = 0.0
        self.has_prev: bool = False

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.has_prev = False

    def update(self, error: float, dt: float, d_error: Optional[float] = None) -> float:
        if dt <= 1e-6:
            return 0.0

        # Integral with anti-windup
        self.integral += error * dt
        self.integral = max(-self.gains.i_max, min(self.gains.i_max, self.integral))

        # Derivative
        if d_error is not None:
            deriv = d_error
        elif self.has_prev:
            deriv = (error - self.prev_error) / dt
        else:
            deriv = 0.0
            self.has_prev = True

        self.prev_error = error

        out = (self.gains.kp * error) + (self.gains.ki * self.integral) + (self.gains.kd * deriv)
        return max(self.gains.out_min, min(self.gains.out_max, out))


class PIDPositionController(Controller):
    """Full Cascaded Position $\\to$ Velocity $\\to$ Attitude $\\to$ Rate PID Controller.

    Architecture:
    1. Position error in NED -> Desired NED velocity
    2. Velocity error in NED -> Desired thrust vector -> Desired Roll, Pitch, Thrust
    3. Attitude error (Roll, Pitch, Yaw) -> Desired body angular rates (p, q, r)
    4. Rate error -> Body control moments (tau_x, tau_y, tau_z)
    5. Motor allocation via mixing matrix -> Individual motor throttle [0, 1]
    """

    def __init__(
        self,
        mass_kg: float = 1.5,
        num_motors: int = 4,
        max_tilt_deg: float = 35.0,
        max_vel_xy_mps: float = 12.0,
        max_vel_z_mps: float = 4.0,
    ) -> None:
        self.mass = mass_kg
        self.num_motors = num_motors
        self.max_tilt_rad = math.radians(max_tilt_deg)
        self.max_vel_xy = max_vel_xy_mps
        self.max_vel_z = max_vel_z_mps

        # Axis PID controllers
        self.pid_x = PIDAxis(PIDGains(kp=1.5, ki=0.05, kd=0.8, i_max=2.0))
        self.pid_y = PIDAxis(PIDGains(kp=1.5, ki=0.05, kd=0.8, i_max=2.0))
        self.pid_z = PIDAxis(PIDGains(kp=2.5, ki=0.20, kd=1.2, i_max=5.0))

        self.pid_roll = PIDAxis(PIDGains(kp=6.5, ki=0.1, kd=0.45, i_max=1.0))
        self.pid_pitch = PIDAxis(PIDGains(kp=6.5, ki=0.1, kd=0.45, i_max=1.0))
        self.pid_yaw = PIDAxis(PIDGains(kp=4.0, ki=0.05, kd=0.3, i_max=1.0))

        self.pid_p = PIDAxis(PIDGains(kp=0.18, ki=0.0, kd=0.005))
        self.pid_q = PIDAxis(PIDGains(kp=0.18, ki=0.0, kd=0.005))
        self.pid_r = PIDAxis(PIDGains(kp=0.25, ki=0.0, kd=0.008))

        # Default allocation weights (X-configuration quadcopter)
        self.base_hover_throttle = (mass_kg * STANDARD_GRAVITY_MPS2) / (num_motors * 7.5)

    def reset(self) -> None:
        for p in [self.pid_x, self.pid_y, self.pid_z, self.pid_roll, self.pid_pitch, self.pid_yaw, self.pid_p, self.pid_q, self.pid_r]:
            p.reset()

    def compute(
        self,
        state: RigidBodyState,
        dt: float,
        target_pos_ned: Optional[np.ndarray] = None,
        target_yaw_rad: float = 0.0,
    ) -> np.ndarray:
        target = target_pos_ned if target_pos_ned is not None else np.array([0.0, 0.0, -5.0])

        pos_curr = state.pos_ned
        vel_curr = state.vel_ned

        # 1. Position Error -> Desired Acceleration in NED
        err_x = target[0] - pos_curr[0]
        err_y = target[1] - pos_curr[1]
        err_z = target[2] - pos_curr[2]  # Negative is up

        ax_cmd = self.pid_x.update(err_x, dt, d_error=-vel_curr[0])
        ay_cmd = self.pid_y.update(err_y, dt, d_error=-vel_curr[1])
        az_cmd = self.pid_z.update(err_z, dt, d_error=-vel_curr[2])

        # Clamp horizontal accelerations to tilt limit
        max_a_xy = STANDARD_GRAVITY_MPS2 * math.tan(self.max_tilt_rad)
        ax_cmd = max(-max_a_xy, min(max_a_xy, ax_cmd))
        ay_cmd = max(-max_a_xy, min(max_a_xy, ay_cmd))

        # Total vertical acceleration: must oppose gravity
        # In NED, gravity is +g (downward). Total upward force needed: m * (g - az_cmd)
        thrust_total_n = self.mass * max(2.0, (STANDARD_GRAVITY_MPS2 - az_cmd))

        # 2. Desired Roll and Pitch from horizontal accelerations and current Yaw
        roll, pitch, yaw = state.euler

        # Rotate horizontal commands to body yaw direction
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        a_forward = ax_cmd * cos_yaw + ay_cmd * sin_yaw
        a_right   = -ax_cmd * sin_yaw + ay_cmd * cos_yaw

        target_pitch = math.atan2(a_forward, STANDARD_GRAVITY_MPS2)
        target_roll  = math.atan2(-a_right, STANDARD_GRAVITY_MPS2)

        target_pitch = max(-self.max_tilt_rad, min(self.max_tilt_rad, target_pitch))
        target_roll  = max(-self.max_tilt_rad, min(self.max_tilt_rad, target_roll))

        # 3. Attitude Error -> Desired Body Angular Rates
        err_roll = target_roll - roll
        err_pitch = target_pitch - pitch

        # Wrap yaw error to [-pi, pi]
        err_yaw = (target_yaw_rad - yaw + math.pi) % (2.0 * math.pi) - math.pi

        p_cmd = self.pid_roll.update(err_roll, dt)
        q_cmd = self.pid_pitch.update(err_pitch, dt)
        r_cmd = self.pid_yaw.update(err_yaw, dt)

        # 4. Rate Error -> Control Torques
        omega = state.omega_body
        tau_x = self.pid_p.update(p_cmd - omega[0], dt)
        tau_y = self.pid_q.update(q_cmd - omega[1], dt)
        tau_z = self.pid_r.update(r_cmd - omega[2], dt)

        # 5. Multirotor Throttle Allocation
        # Base throttle per motor
        t_base = thrust_total_n / (self.num_motors * 7.5)

        if self.num_motors == 4:
            # Standard X-quad motor mixing
            # M1: Front-Right (+pitch, -roll, -yaw)
            # M2: Front-Left  (+pitch, +roll, +yaw)
            # M3: Rear-Left   (-pitch, +roll, -yaw)
            # M4: Rear-Right  (-pitch, -roll, +yaw)
            u1 = t_base - tau_x + tau_y - tau_z
            u2 = t_base + tau_x + tau_y + tau_z
            u3 = t_base + tau_x - tau_y - tau_z
            u4 = t_base - tau_x - tau_y + tau_z
            cmds = np.array([u1, u2, u3, u4], dtype=np.float64)
        else:
            cmds = np.full(self.num_motors, t_base, dtype=np.float64)

        return np.clip(cmds, 0.0, 1.0)
