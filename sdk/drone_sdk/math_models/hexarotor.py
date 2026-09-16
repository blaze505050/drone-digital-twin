"""
drone_sdk.math_models.hexarotor
================================
Hexarotor Mathematical Model — Full Nonlinear 6-DOF Dynamics.

Supports three hexarotor layouts:
  - Flat X (6 motors in a plane, 60° apart)
  - Flat Star/Y (alternating CW/CCW at 60° intervals)
  - Coaxial Y6 (3 arms, 2 coaxial rotors per arm)

Key feature: motor failure reconfiguration — a hexarotor can sustain
controlled flight with one motor lost by reallocating remaining motors
via the pseudo-inverse of the reduced mixing matrix.

Motor numbering (Flat X, viewed from above)::

          1 (CW)
       ╱        ╲
    6 (CW)    2 (CCW)     → y (Right)
       │          │
    5 (CCW)   3 (CW)
       ╲        ╱
          4 (CCW)
          ↓
          x (Forward)

References:
  - Du, Schulz, Zhu, "Single Motor Loss Tolerance for Hexacopters",
    IROS 2015.
  - Michieletto, Ryll, Franchi, "Fundamental Actuation Properties of
    Multirotors", IEEE T-RO 2018.

Python version: 3.9+
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .frames import quat_to_dcm
from .rigid_body import (
    ExternalWrench,
    InertiaParams,
    RigidBodyState,
    GRAVITY_MPS2,
)


class HexConfig(str, Enum):
    FLAT_X = "flat_x"
    FLAT_STAR = "flat_star"
    COAXIAL_Y6 = "coaxial_y6"


@dataclass
class HexarotorParams:
    """Physical parameters for a hexarotor UAV."""
    mass_kg: float = 2.80
    arm_length_m: float = 0.30

    ixx: float = 0.065
    iyy: float = 0.065
    izz: float = 0.120
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0

    max_thrust_per_motor_n: float = 7.50
    motor_torque_coeff: float = 0.015
    motor_time_constant_s: float = 0.035
    rotor_polar_inertia: float = 1.8e-5
    prop_radius_m: float = 0.127

    cd_body: float = 1.10
    frontal_area_m2: float = 0.035

    # Coaxial interference factor (lower rotor receives ~85% effective thrust)
    coaxial_interference: float = 0.85

    config: HexConfig = HexConfig.FLAT_X

    @property
    def inertia_tensor(self) -> np.ndarray:
        return np.array([
            [self.ixx, self.ixy, self.ixz],
            [self.ixy, self.iyy, self.iyz],
            [self.ixz, self.iyz, self.izz],
        ], dtype=np.float64)

    @property
    def to_inertia_params(self) -> InertiaParams:
        return InertiaParams(mass_kg=self.mass_kg, inertia_tensor=self.inertia_tensor)


def _hex_flat_rotor_positions(arm: float, n_motors: int = 6) -> np.ndarray:
    """Compute rotor positions for flat hexarotor (60° spacing).

    Returns (6, 3) array of rotor positions in body frame [x, y, z].
    """
    positions = np.zeros((n_motors, 3), dtype=np.float64)
    for i in range(n_motors):
        angle = math.pi / 2.0 - i * (2.0 * math.pi / n_motors)  # Start from front
        positions[i, 0] = arm * math.cos(angle)
        positions[i, 1] = arm * math.sin(angle)
    return positions


def mixing_matrix_hex_flat(arm: float, c_tau: float) -> np.ndarray:
    """Mixing matrix for flat hexarotor (6 motors, alternating CW/CCW).

    Returns 4×6 matrix: [T, τ_φ, τ_θ, τ_ψ] = M @ [F1..F6].
    """
    spin_dirs = np.array([1, -1, 1, -1, 1, -1], dtype=np.float64)
    positions = _hex_flat_rotor_positions(arm)

    M = np.zeros((4, 6), dtype=np.float64)
    M[0, :] = 1.0  # Total thrust
    for i in range(6):
        M[1, i] = -positions[i, 1]              # Roll: -y_i * F_i
        M[2, i] = positions[i, 0]               # Pitch: x_i * F_i
        M[3, i] = spin_dirs[i] * c_tau           # Yaw: spin_dir * c_τ * F_i

    return M


def mixing_matrix_coaxial_y6(arm: float, c_tau: float, coax_factor: float = 0.85) -> np.ndarray:
    """Mixing matrix for Y6 coaxial hexarotor (3 arms × 2 coaxial rotors).

    Motors 1,3,5 = upper (full thrust); motors 2,4,6 = lower (reduced by coax_factor).
    """
    spin_dirs = np.array([1, -1, 1, -1, 1, -1], dtype=np.float64)
    M = np.zeros((4, 6), dtype=np.float64)

    for i in range(3):
        angle = math.pi / 2.0 - i * (2.0 * math.pi / 3.0)
        x_i = arm * math.cos(angle)
        y_i = arm * math.sin(angle)

        upper = 2 * i
        lower = 2 * i + 1

        # Upper rotor (full thrust)
        M[0, upper] = 1.0
        M[1, upper] = -y_i
        M[2, upper] = x_i
        M[3, upper] = spin_dirs[upper] * c_tau

        # Lower rotor (reduced thrust due to coaxial interference)
        M[0, lower] = coax_factor
        M[1, lower] = -y_i * coax_factor
        M[2, lower] = x_i * coax_factor
        M[3, lower] = spin_dirs[lower] * c_tau * coax_factor

    return M


class HexarotorModel:
    """Full nonlinear hexarotor dynamics model."""

    def __init__(self, params: Optional[HexarotorParams] = None) -> None:
        self.params = params or HexarotorParams()
        p = self.params

        if p.config == HexConfig.COAXIAL_Y6:
            self.mixing = mixing_matrix_coaxial_y6(p.arm_length_m, p.motor_torque_coeff, p.coaxial_interference)
        else:
            self.mixing = mixing_matrix_hex_flat(p.arm_length_m, p.motor_torque_coeff)

        self.mixing_inv = np.linalg.pinv(self.mixing)
        self.motor_thrusts = np.zeros(6, dtype=np.float64)
        self.spin_dirs = np.array([1, -1, 1, -1, 1, -1], dtype=np.float64)
        self._failed_motors: set = set()

    def compute_wrench(
        self,
        state: RigidBodyState,
        motor_commands: np.ndarray,
        dt: float = 0.005,
        air_density: float = 1.225,
        wind_ned: Optional[np.ndarray] = None,
    ) -> ExternalWrench:
        """Compute body forces and torques from motor commands."""
        p = self.params
        if len(motor_commands) < 6:
            cmds = np.zeros(6, dtype=np.float64)
            if len(motor_commands) == 4:
                cmds[[0, 1]] = motor_commands[0]
                cmds[[2]]    = motor_commands[1]
                cmds[[3, 4]] = motor_commands[2]
                cmds[[5]]    = motor_commands[3]
            elif len(motor_commands) > 0:
                cmds[:len(motor_commands)] = motor_commands
                cmds[len(motor_commands):] = np.mean(motor_commands)
        else:
            cmds = np.array(motor_commands[:6], dtype=np.float64)
        cmds = np.clip(cmds, 0.0, 1.0)

        # Zero out failed motors
        for m_idx in self._failed_motors:
            cmds[m_idx] = 0.0

        target_thrusts = cmds * p.max_thrust_per_motor_n

        # Motor lag
        alpha_m = min(1.0, dt / max(1e-4, p.motor_time_constant_s))
        self.motor_thrusts += (target_thrusts - self.motor_thrusts) * alpha_m
        thrusts = self.motor_thrusts.copy()

        # Mixing → body wrench
        wrench_vec = self.mixing @ thrusts
        F_thrust_body = np.array([0.0, 0.0, -wrench_vec[0]], dtype=np.float64)
        tau_thrust = wrench_vec[1:4]

        # Gyroscopic torque
        omega_rotors = np.sqrt(np.maximum(thrusts, 0.0) / max(1e-9, p.max_thrust_per_motor_n)) * 1000.0
        net_h = p.rotor_polar_inertia * np.sum(omega_rotors * self.spin_dirs)
        tau_gyro = np.array([
            -state.omega_body[1] * net_h,
             state.omega_body[0] * net_h,
             0.0,
        ], dtype=np.float64)

        # Parasitic drag
        v_air = state.vel_body
        if wind_ned is not None:
            R_bn = quat_to_dcm(state.quat).T
            v_air = v_air - R_bn @ wind_ned
        drag_body = -0.5 * air_density * p.cd_body * p.frontal_area_m2 * v_air * np.abs(v_air)

        return ExternalWrench(
            force_body=F_thrust_body + drag_body,
            torque_body=tau_thrust + tau_gyro,
        )

    def fail_motor(self, motor_index: int) -> None:
        """Simulate motor failure. Reconfigures control allocation automatically."""
        if 0 <= motor_index < 6:
            self._failed_motors.add(motor_index)
            self._recompute_allocation()

    def recover_motor(self, motor_index: int) -> None:
        """Restore a failed motor."""
        self._failed_motors.discard(motor_index)
        self._recompute_allocation()

    def _recompute_allocation(self) -> None:
        """Recompute pseudo-inverse mixing with failed motors zeroed out."""
        M_reduced = self.mixing.copy()
        for m_idx in self._failed_motors:
            M_reduced[:, m_idx] = 0.0
        self.mixing_inv = np.linalg.pinv(M_reduced)

    def allocate_motors(self, desired_thrust: float, desired_torques: np.ndarray) -> np.ndarray:
        """Control allocation with failure reconfiguration."""
        desired = np.array([desired_thrust, *desired_torques], dtype=np.float64)
        raw = self.mixing_inv @ desired
        normalised = raw / max(1e-6, self.params.max_thrust_per_motor_n)
        return np.clip(normalised, 0.0, 1.0)

    def hover_command(self) -> np.ndarray:
        """Motor commands for hover."""
        t_per = self.params.mass_kg * GRAVITY_MPS2 / 6.0
        return np.full(6, t_per / self.params.max_thrust_per_motor_n, dtype=np.float64)

    def can_sustain_hover_with_failure(self) -> bool:
        """Check if hover is achievable with current motor failures."""
        total_available = (6 - len(self._failed_motors)) * self.params.max_thrust_per_motor_n
        weight = self.params.mass_kg * GRAVITY_MPS2
        return total_available > weight * 1.1  # 10% margin

    @property
    def uav_type(self) -> str:
        return "hexarotor"

    @property
    def num_motors(self) -> int:
        return 6
