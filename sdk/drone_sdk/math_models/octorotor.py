"""
drone_sdk.math_models.octorotor
================================
Octorotor Mathematical Model — Full Nonlinear 6-DOF Dynamics.

Supports two octorotor layouts:
  - Flat X8 (8 motors in a plane, 45° apart)
  - Coaxial X8 (4 arms, 2 coaxial rotors per arm)

Key feature: dual motor failure tolerance — an octorotor can sustain
controlled flight with up to 2 motors lost (non-adjacent) through
pseudo-inverse reallocation of the remaining motors.

Motor numbering (Flat X8, viewed from above)::

        1     2
      ╱    ╲╱    ╲
    8    ╳    3    → y
      ╲    ╱╲    ╱
        7     4
      ╲    ╱╲    ╱
    6    ╳    5
      ╱    ╲╱    ╱
          ↓
          x (Forward)

References:
  - Marks, Whidborne, Yamamoto, "Control Allocation for Fault-Tolerant
    Multirotor UAVs", IFAC 2012.
  - Russell, "Longitudinal Stability of Octocopters", IEEE ICUAS 2018.

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


class OctoConfig(str, Enum):
    FLAT_X8 = "flat_x8"
    COAXIAL_X8 = "coaxial_x8"


@dataclass
class OctorotorParams:
    """Physical parameters for an octorotor UAV."""
    mass_kg: float = 4.20
    arm_length_m: float = 0.35

    ixx: float = 0.120
    iyy: float = 0.120
    izz: float = 0.220
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0

    max_thrust_per_motor_n: float = 8.50
    motor_torque_coeff: float = 0.015
    motor_time_constant_s: float = 0.035
    rotor_polar_inertia: float = 2.0e-5
    prop_radius_m: float = 0.127

    cd_body: float = 1.15
    frontal_area_m2: float = 0.045

    coaxial_interference: float = 0.85

    config: OctoConfig = OctoConfig.FLAT_X8

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


def mixing_matrix_flat_x8(arm: float, c_tau: float) -> np.ndarray:
    """Mixing matrix for flat X8 octorotor (8 motors, 45° apart).

    Returns 4×8 matrix: [T, τ_φ, τ_θ, τ_ψ] = M @ [F1..F8].
    """
    spin_dirs = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)
    M = np.zeros((4, 8), dtype=np.float64)
    M[0, :] = 1.0

    for i in range(8):
        angle = math.pi / 2.0 - i * (2.0 * math.pi / 8.0)
        x_i = arm * math.cos(angle)
        y_i = arm * math.sin(angle)
        M[1, i] = -y_i                    # Roll
        M[2, i] = x_i                     # Pitch
        M[3, i] = spin_dirs[i] * c_tau     # Yaw

    return M


def mixing_matrix_coaxial_x8(arm: float, c_tau: float, coax_factor: float = 0.85) -> np.ndarray:
    """Mixing matrix for coaxial X8 (4 arms × 2 coaxial rotors).

    Motors 0,2,4,6 = upper; motors 1,3,5,7 = lower.
    """
    spin_dirs = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)
    M = np.zeros((4, 8), dtype=np.float64)

    for i in range(4):
        angle = math.pi / 4.0 + i * (math.pi / 2.0)  # 45° offset for X-config
        x_i = arm * math.cos(angle)
        y_i = arm * math.sin(angle)

        upper = 2 * i
        lower = 2 * i + 1

        M[0, upper] = 1.0
        M[1, upper] = -y_i
        M[2, upper] = x_i
        M[3, upper] = spin_dirs[upper] * c_tau

        M[0, lower] = coax_factor
        M[1, lower] = -y_i * coax_factor
        M[2, lower] = x_i * coax_factor
        M[3, lower] = spin_dirs[lower] * c_tau * coax_factor

    return M


class OctorotorModel:
    """Full nonlinear octorotor dynamics model with dual-failure tolerance."""

    def __init__(self, params: Optional[OctorotorParams] = None) -> None:
        self.params = params or OctorotorParams()
        p = self.params

        if p.config == OctoConfig.COAXIAL_X8:
            self.mixing = mixing_matrix_coaxial_x8(p.arm_length_m, p.motor_torque_coeff, p.coaxial_interference)
        else:
            self.mixing = mixing_matrix_flat_x8(p.arm_length_m, p.motor_torque_coeff)

        self.mixing_inv = np.linalg.pinv(self.mixing)
        self.motor_thrusts = np.zeros(8, dtype=np.float64)
        self.spin_dirs = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)
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
        if len(motor_commands) < 8:
            cmds = np.zeros(8, dtype=np.float64)
            if len(motor_commands) == 4:
                cmds[[0, 1]] = motor_commands[0]
                cmds[[2, 3]] = motor_commands[1]
                cmds[[4, 5]] = motor_commands[2]
                cmds[[6, 7]] = motor_commands[3]
            elif len(motor_commands) > 0:
                cmds[:len(motor_commands)] = motor_commands
                cmds[len(motor_commands):] = np.mean(motor_commands)
        else:
            cmds = np.array(motor_commands[:8], dtype=np.float64)
        cmds = np.clip(cmds, 0.0, 1.0)

        for m_idx in self._failed_motors:
            cmds[m_idx] = 0.0

        target_thrusts = cmds * p.max_thrust_per_motor_n

        alpha_m = min(1.0, dt / max(1e-4, p.motor_time_constant_s))
        self.motor_thrusts += (target_thrusts - self.motor_thrusts) * alpha_m
        thrusts = self.motor_thrusts.copy()

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
        if 0 <= motor_index < 8:
            self._failed_motors.add(motor_index)
            self._recompute_allocation()

    def recover_motor(self, motor_index: int) -> None:
        self._failed_motors.discard(motor_index)
        self._recompute_allocation()

    def _recompute_allocation(self) -> None:
        M_reduced = self.mixing.copy()
        for m_idx in self._failed_motors:
            M_reduced[:, m_idx] = 0.0
        self.mixing_inv = np.linalg.pinv(M_reduced)

    def allocate_motors(self, desired_thrust: float, desired_torques: np.ndarray) -> np.ndarray:
        desired = np.array([desired_thrust, *desired_torques], dtype=np.float64)
        raw = self.mixing_inv @ desired
        normalised = raw / max(1e-6, self.params.max_thrust_per_motor_n)
        return np.clip(normalised, 0.0, 1.0)

    def hover_command(self) -> np.ndarray:
        t_per = self.params.mass_kg * GRAVITY_MPS2 / 8.0
        return np.full(8, t_per / self.params.max_thrust_per_motor_n, dtype=np.float64)

    def max_failures_tolerable(self) -> int:
        """Maximum number of motor failures that still allow hover."""
        weight = self.params.mass_kg * GRAVITY_MPS2
        for n_fail in range(8):
            if (8 - n_fail) * self.params.max_thrust_per_motor_n < weight * 1.1:
                return max(0, n_fail - 1)
        return 7

    @property
    def uav_type(self) -> str:
        return "octorotor"

    @property
    def num_motors(self) -> int:
        return 8
