"""
drone_sdk.flight_simulator.controller
======================================
PID Cascade Flight Controller for UAV Simulation.

Implements a PX4-style cascaded PID controller:

    Position loop  → Velocity loop  → Attitude loop  → Rate loop  → Motor mixing
    (outer)           (mid-outer)      (mid-inner)      (inner)       (output)

Each loop runs at different rates:
  - Position/Velocity: ~50 Hz
  - Attitude/Rate: ~250 Hz

Supports:
  - Multirotor mode (direct motor commands)
  - Fixed-wing mode (surface deflections + throttle)
  - VTOL transition blend

Python version: 3.9+
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ..math_models.frames import quat_to_euler, euler_to_quat, quat_normalise
from ..math_models.rigid_body import RigidBodyState, GRAVITY_MPS2


@dataclass
class PIDGains:
    """PID controller gains."""
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    i_max: float = 1.0     # Integral windup limit
    output_max: float = 1.0  # Output saturation


class PIDController:
    """Single-axis PID controller with anti-windup."""

    def __init__(self, gains: PIDGains) -> None:
        self.gains = gains
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._initialised: bool = False

    def update(self, error: float, dt: float) -> float:
        """Compute PID output."""
        g = self.gains
        dt = max(1e-4, dt)

        # Proportional
        p_term = g.kp * error

        # Integral with anti-windup
        self._integral += error * dt
        self._integral = max(-g.i_max, min(g.i_max, self._integral))
        i_term = g.ki * self._integral

        # Derivative (on error)
        if not self._initialised:
            d_term = 0.0
            self._initialised = True
        else:
            d_term = g.kd * (error - self._prev_error) / dt

        self._prev_error = error
        output = p_term + i_term + d_term
        return max(-g.output_max, min(g.output_max, output))

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialised = False


@dataclass
class MultirotorControllerConfig:
    """Configuration for multirotor cascaded PID controller."""
    # Position controller gains
    pos_xy: PIDGains = field(default_factory=lambda: PIDGains(kp=1.5, ki=0.1, kd=0.3, i_max=2.0, output_max=5.0))
    pos_z: PIDGains = field(default_factory=lambda: PIDGains(kp=2.0, ki=0.3, kd=0.5, i_max=3.0, output_max=8.0))

    # Velocity controller gains
    vel_xy: PIDGains = field(default_factory=lambda: PIDGains(kp=1.8, ki=0.2, kd=0.1, i_max=3.0, output_max=10.0))
    vel_z: PIDGains = field(default_factory=lambda: PIDGains(kp=3.0, ki=0.5, kd=0.2, i_max=5.0, output_max=15.0))

    # Attitude controller gains
    att_roll: PIDGains = field(default_factory=lambda: PIDGains(kp=6.0, ki=0.0, kd=0.0, output_max=3.0))
    att_pitch: PIDGains = field(default_factory=lambda: PIDGains(kp=6.0, ki=0.0, kd=0.0, output_max=3.0))
    att_yaw: PIDGains = field(default_factory=lambda: PIDGains(kp=3.0, ki=0.0, kd=0.0, output_max=2.0))

    # Rate controller gains
    rate_roll: PIDGains = field(default_factory=lambda: PIDGains(kp=0.15, ki=0.05, kd=0.002, i_max=0.3, output_max=0.4))
    rate_pitch: PIDGains = field(default_factory=lambda: PIDGains(kp=0.15, ki=0.05, kd=0.002, i_max=0.3, output_max=0.4))
    rate_yaw: PIDGains = field(default_factory=lambda: PIDGains(kp=0.10, ki=0.02, kd=0.0, i_max=0.2, output_max=0.3))

    max_tilt_rad: float = math.radians(35)


class MultirotorController:
    """Cascaded PID controller for multirotor UAVs.

    Outer loop: Position → desired velocity
    Mid loop: Velocity → desired attitude + thrust
    Inner loop: Attitude → desired body rates
    Innermost: Rate → motor differential commands
    """

    def __init__(self, config: Optional[MultirotorControllerConfig] = None, mass_kg: float = 1.5) -> None:
        self.config = config or MultirotorControllerConfig()
        self.mass_kg = mass_kg
        c = self.config

        # Position controllers
        self.pid_pos_x = PIDController(c.pos_xy)
        self.pid_pos_y = PIDController(c.pos_xy)
        self.pid_pos_z = PIDController(c.pos_z)

        # Velocity controllers
        self.pid_vel_x = PIDController(c.vel_xy)
        self.pid_vel_y = PIDController(c.vel_xy)
        self.pid_vel_z = PIDController(c.vel_z)

        # Attitude controllers
        self.pid_att_roll = PIDController(c.att_roll)
        self.pid_att_pitch = PIDController(c.att_pitch)
        self.pid_att_yaw = PIDController(c.att_yaw)

        # Rate controllers
        self.pid_rate_roll = PIDController(c.rate_roll)
        self.pid_rate_pitch = PIDController(c.rate_pitch)
        self.pid_rate_yaw = PIDController(c.rate_yaw)

    def position_control(
        self,
        state: RigidBodyState,
        target_pos_ned: np.ndarray,
        target_yaw: float,
        dt: float,
    ) -> np.ndarray:
        """Full cascade: position setpoint → 4 motor commands [0, 1]^4.

        Args:
            state: Current rigid body state.
            target_pos_ned: Desired position in NED frame [x, y, z].
            target_yaw: Desired yaw angle (rad).
            dt: Control timestep.

        Returns:
            Motor commands [0, 1]^4 for quadrotor.
        """
        vel_ned = state.vel_ned
        roll, pitch, yaw = state.euler
        omega = state.omega_body

        # ── Position → Velocity ──
        pos_err = target_pos_ned - state.pos_ned
        vel_des_x = self.pid_pos_x.update(pos_err[0], dt)
        vel_des_y = self.pid_pos_y.update(pos_err[1], dt)
        vel_des_z = self.pid_pos_z.update(pos_err[2], dt)

        # ── Velocity → Desired attitude + thrust ──
        vel_err = np.array([vel_des_x, vel_des_y, vel_des_z]) - vel_ned
        acc_des_x = self.pid_vel_x.update(vel_err[0], dt)
        acc_des_y = self.pid_vel_y.update(vel_err[1], dt)
        acc_des_z = self.pid_vel_z.update(vel_err[2], dt)

        # Desired thrust (NED: z-positive down, so thrust compensates gravity)
        thrust_des = self.mass_kg * (GRAVITY_MPS2 - acc_des_z)  # Positive = up
        thrust_normalised = max(0.0, thrust_des) / (self.mass_kg * GRAVITY_MPS2 * 2.0)

        # Desired roll/pitch from horizontal accelerations (small angle)
        cy, sy = math.cos(yaw), math.sin(yaw)
        max_tilt = self.config.max_tilt_rad
        roll_des = max(-max_tilt, min(max_tilt,
            (acc_des_x * sy - acc_des_y * cy) / max(GRAVITY_MPS2, 1.0)))
        pitch_des = max(-max_tilt, min(max_tilt,
            (acc_des_x * cy + acc_des_y * sy) / max(GRAVITY_MPS2, 1.0)))

        # ── Attitude → Rate ──
        roll_err = roll_des - roll
        pitch_err = pitch_des - pitch
        yaw_err = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))

        rate_des_roll = self.pid_att_roll.update(roll_err, dt)
        rate_des_pitch = self.pid_att_pitch.update(pitch_err, dt)
        rate_des_yaw = self.pid_att_yaw.update(yaw_err, dt)

        # ── Rate → Motor differentials ──
        rate_err_roll = rate_des_roll - omega[0]
        rate_err_pitch = rate_des_pitch - omega[1]
        rate_err_yaw = rate_des_yaw - omega[2]

        d_roll = self.pid_rate_roll.update(rate_err_roll, dt)
        d_pitch = self.pid_rate_pitch.update(rate_err_pitch, dt)
        d_yaw = self.pid_rate_yaw.update(rate_err_yaw, dt)

        # ── Motor mixing (X-config) ──
        m1 = thrust_normalised - d_roll + d_pitch - d_yaw
        m2 = thrust_normalised + d_roll + d_pitch + d_yaw
        m3 = thrust_normalised - d_roll - d_pitch + d_yaw
        m4 = thrust_normalised + d_roll - d_pitch - d_yaw

        return np.clip([m1, m2, m3, m4], 0.0, 1.0)

    def manual_control(
        self,
        state: RigidBodyState,
        roll_cmd: float,
        pitch_cmd: float,
        yaw_rate_cmd: float,
        throttle: float,
        dt: float,
    ) -> np.ndarray:
        """Rate/attitude mode: direct stick → motor commands.

        Args:
            roll_cmd: Desired roll angle (rad), clamped to ±max_tilt.
            pitch_cmd: Desired pitch angle (rad), clamped to ±max_tilt.
            yaw_rate_cmd: Desired yaw rate (rad/s).
            throttle: Normalised throttle [0, 1].
            dt: Control timestep.
        """
        roll, pitch, yaw = state.euler
        omega = state.omega_body
        max_tilt = self.config.max_tilt_rad

        roll_des = max(-max_tilt, min(max_tilt, roll_cmd))
        pitch_des = max(-max_tilt, min(max_tilt, pitch_cmd))

        # Attitude → rate
        rate_des_roll = self.pid_att_roll.update(roll_des - roll, dt)
        rate_des_pitch = self.pid_att_pitch.update(pitch_des - pitch, dt)
        rate_des_yaw = yaw_rate_cmd

        # Rate → motor
        d_roll = self.pid_rate_roll.update(rate_des_roll - omega[0], dt)
        d_pitch = self.pid_rate_pitch.update(rate_des_pitch - omega[1], dt)
        d_yaw = self.pid_rate_yaw.update(rate_des_yaw - omega[2], dt)

        m1 = throttle - d_roll + d_pitch - d_yaw
        m2 = throttle + d_roll + d_pitch + d_yaw
        m3 = throttle - d_roll - d_pitch + d_yaw
        m4 = throttle + d_roll - d_pitch - d_yaw

        return np.clip([m1, m2, m3, m4], 0.0, 1.0)

    def reset(self) -> None:
        """Reset all PID integrators."""
        for attr in dir(self):
            obj = getattr(self, attr)
            if isinstance(obj, PIDController):
                obj.reset()
