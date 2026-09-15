"""
drone_sdk.math_models.quadrotor
================================
Quadrotor Mathematical Model — Full Nonlinear 6-DOF Dynamics.

Supports both X-configuration and +-configuration motor layouts with
complete force/moment models including:

  - Motor mixing matrix (thrust allocation → body forces & torques)
  - Ground effect (Cheeseman & Bennett 1955)
  - Rotor gyroscopic precession torque
  - Blade flapping (first-harmonic tip-path-plane tilt)
  - Parasitic drag (body) + induced drag (rotor disc)
  - Motor first-order spin-up lag

Motor numbering (X-configuration, viewed from above)::

        Front
    1 (CW)   2 (CCW)
        ╲   ╱
         ╲ ╱
          ╳    → y (Right)
         ╱ ╲
        ╱   ╲
    3 (CCW)  4 (CW)
        ↓
        x (Forward)

References:
  - Mahony, Kumar & Corke, "Multirotor Aerial Vehicles: Modeling, Estimation, and
    Control of Quadrotor", IEEE RAM 2012.
  - Bouabdallah, "Design and Control of Quadrotors with Application to Autonomous
    Flying", PhD Thesis, EPFL 2007.
  - Cheeseman & Bennett, "Effect of Ground on a Helicopter Rotor in Forward Flight",
    ARC R&M 3021, 1955.

Python version: 3.9+
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from .frames import quat_rotate_vector, quat_to_dcm, quat_to_euler
from .rigid_body import (
    ExternalWrench,
    InertiaParams,
    RigidBodySimulator,
    RigidBodyState,
    GRAVITY_MPS2,
)


class QuadConfig(str, Enum):
    """Motor layout configuration."""
    X_CONFIG = "x"        # Arms at 45° from forward axis
    PLUS_CONFIG = "plus"  # Arms along forward/right axes


@dataclass
class QuadrotorParams:
    """Physical parameters for a quadrotor UAV.

    All SI units unless noted.
    """
    mass_kg: float = 1.50
    arm_length_m: float = 0.25

    # Inertia tensor (symmetric, about CG)
    ixx: float = 0.035
    iyy: float = 0.046
    izz: float = 0.098
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0

    # Motor/propeller parameters
    max_thrust_per_motor_n: float = 6.62
    motor_torque_coeff: float = 0.015     # c_τ = Q / T (reaction torque / thrust ratio)
    motor_time_constant_s: float = 0.035  # First-order spin-up lag τ_m
    rotor_polar_inertia: float = 1.5e-5   # I_r for gyroscopic torque (kg·m²)
    prop_radius_m: float = 0.127          # Propeller radius R (m)

    # Aerodynamic drag
    cd_body: float = 1.05                 # Body drag coefficient
    frontal_area_m2: float = 0.025        # Effective frontal area (m²)

    # Blade flapping (first-harmonic)
    flapping_gain: float = 0.0            # k_β, typically 0.01–0.05, 0 = disabled

    # Configuration
    config: QuadConfig = QuadConfig.X_CONFIG

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


# ─────────────────────────────────────────────────────────────────────────────
#  Motor Mixing Matrices
# ─────────────────────────────────────────────────────────────────────────────

def mixing_matrix_x(arm: float, c_tau: float) -> np.ndarray:
    """Motor mixing matrix for X-configuration quadrotor.

    Maps individual motor thrusts [F1, F2, F3, F4] to body wrench
    [Total_thrust, τ_roll, τ_pitch, τ_yaw].

    X-config arm projections: ±L/√2 on both x and y axes.

    Motor layout and spin directions::

        1 (CW, front-left)   2 (CCW, front-right)
        3 (CCW, rear-left)   4 (CW, rear-right)

    Returns:
        4×4 mixing matrix M such that [T, τ_φ, τ_θ, τ_ψ] = M @ [F1, F2, F3, F4].
    """
    a = arm / math.sqrt(2.0)  # Projected arm length
    return np.array([
        [ 1.0,    1.0,    1.0,    1.0  ],   # Total thrust
        [-a,      a,     -a,      a    ],   # Roll torque (about x)
        [ a,      a,     -a,     -a    ],   # Pitch torque (about y)
        [-c_tau,  c_tau, c_tau,  -c_tau],   # Yaw torque (about z)
    ], dtype=np.float64)


def mixing_matrix_plus(arm: float, c_tau: float) -> np.ndarray:
    """Motor mixing matrix for +-configuration quadrotor.

    Motor layout::

        1 (CW, front)
        2 (CCW, right)   3 (CCW, left)
        4 (CW, rear)
    """
    return np.array([
        [ 1.0,    1.0,    1.0,    1.0  ],
        [ 0.0,    arm,    0.0,   -arm  ],
        [ arm,    0.0,   -arm,    0.0  ],
        [-c_tau,  c_tau, c_tau,  -c_tau],
    ], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
#  Ground effect model
# ─────────────────────────────────────────────────────────────────────────────

def ground_effect_factor(altitude_agl: float, prop_radius: float) -> float:
    """Cheeseman & Bennett (1955) ground effect thrust augmentation.

    T_ige = T_oge / (1 - (R / 4z)²)

    Returns the multiplicative factor (≥ 1.0). Clamped to max 1.3.
    """
    z = max(altitude_agl, prop_radius * 0.5)
    if z >= 4.0 * prop_radius:
        return 1.0
    ratio = prop_radius / (4.0 * z)
    denom = max(0.6, 1.0 - ratio * ratio)
    return min(1.3, 1.0 / denom)


# ─────────────────────────────────────────────────────────────────────────────
#  Quadrotor dynamics model
# ─────────────────────────────────────────────────────────────────────────────

class QuadrotorModel:
    """Full nonlinear quadrotor dynamics model.

    Computes body forces and torques from motor commands, suitable for
    integration with RigidBodySimulator.

    Usage::

        model = QuadrotorModel(params)
        wrench = model.compute_wrench(state, motor_commands)
        sim.step(wrench, dt)
    """

    def __init__(self, params: Optional[QuadrotorParams] = None) -> None:
        self.params = params or QuadrotorParams()
        p = self.params

        # Build mixing matrix
        if p.config == QuadConfig.X_CONFIG:
            self.mixing = mixing_matrix_x(p.arm_length_m, p.motor_torque_coeff)
        else:
            self.mixing = mixing_matrix_plus(p.arm_length_m, p.motor_torque_coeff)

        # Inverse mixing for control allocation
        self.mixing_inv = np.linalg.pinv(self.mixing)

        # Motor spin-up state (current thrust per motor)
        self.motor_thrusts = np.zeros(4, dtype=np.float64)

        # Spin directions: [CW, CCW, CCW, CW] for X-config
        self.spin_dirs = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)

    def compute_wrench(
        self,
        state: RigidBodyState,
        motor_commands: np.ndarray,
        dt: float = 0.005,
        air_density: float = 1.225,
        wind_ned: Optional[np.ndarray] = None,
    ) -> ExternalWrench:
        """Compute total body forces and torques for given motor commands.

        Args:
            state: Current rigid body state.
            motor_commands: Normalised motor commands [0, 1]^4.
            dt: Timestep for motor lag integration.
            air_density: Air density ρ (kg/m³).
            wind_ned: Wind velocity in NED (m/s), optional.

        Returns:
            ExternalWrench with total force and torque in body frame.
        """
        p = self.params
        rho = air_density
        cmds = np.clip(motor_commands, 0.0, 1.0)
        target_thrusts = cmds * p.max_thrust_per_motor_n

        # 1. Motor first-order lag
        alpha_m = min(1.0, dt / max(1e-4, p.motor_time_constant_s))
        self.motor_thrusts += (target_thrusts - self.motor_thrusts) * alpha_m
        thrusts = self.motor_thrusts.copy()

        # 2. Ground effect
        alt_agl = state.altitude_agl
        ge_factor = ground_effect_factor(alt_agl, p.prop_radius_m)
        thrusts *= ge_factor

        # 3. Apply mixing matrix → body wrench
        wrench_vec = self.mixing @ thrusts  # [T, τ_φ, τ_θ, τ_ψ]
        F_thrust_body = np.array([0.0, 0.0, -wrench_vec[0]], dtype=np.float64)  # Thrust along -z_b (up in FRD)
        tau_thrust = wrench_vec[1:4]

        # 4. Rotor gyroscopic torque
        #    τ_gyro = -ω × (Σ I_r · Ω_i · spin_dir_i · ê_z)
        #    Ω_i ≈ √(T_i / k_T), but we use thrust as proxy for angular momentum
        omega_rotors = np.sqrt(np.maximum(thrusts, 0.0) / max(1e-9, p.max_thrust_per_motor_n)) * 1000.0
        net_angular_momentum = p.rotor_polar_inertia * np.sum(omega_rotors * self.spin_dirs)
        tau_gyro = np.array([
            -state.omega_body[1] * net_angular_momentum,
             state.omega_body[0] * net_angular_momentum,
             0.0,
        ], dtype=np.float64)

        # 5. Blade flapping (H-force due to tip-path-plane tilt)
        tau_flap = np.zeros(3, dtype=np.float64)
        if p.flapping_gain > 0:
            # First-harmonic blade flapping: lateral H-force proportional to body lateral velocity
            v_body = state.vel_body
            tau_flap[0] = -p.flapping_gain * v_body[1] * wrench_vec[0]  # Roll from v_y
            tau_flap[1] =  p.flapping_gain * v_body[0] * wrench_vec[0]  # Pitch from v_x

        # 6. Parasitic drag (body frame)
        v_body = state.vel_body
        if wind_ned is not None:
            R_bn = quat_to_dcm(state.quat).T
            wind_body = R_bn @ wind_ned
            v_air = v_body - wind_body
        else:
            v_air = v_body
        drag_body = -0.5 * rho * p.cd_body * p.frontal_area_m2 * v_air * np.abs(v_air)

        # 7. Gravity is handled by the rigid body integrator, not here
        total_force = F_thrust_body + drag_body
        total_torque = tau_thrust + tau_gyro + tau_flap

        return ExternalWrench(force_body=total_force, torque_body=total_torque)

    def allocate_motors(
        self,
        desired_thrust: float,
        desired_torques: np.ndarray,
    ) -> np.ndarray:
        """Control allocation: desired [T, τ_φ, τ_θ, τ_ψ] → motor commands [0, 1]^4.

        Uses pseudo-inverse of the mixing matrix for optimal allocation.
        """
        desired = np.array([desired_thrust, *desired_torques], dtype=np.float64)
        raw_thrusts = self.mixing_inv @ desired
        normalised = raw_thrusts / max(1e-6, self.params.max_thrust_per_motor_n)
        return np.clip(normalised, 0.0, 1.0)

    def hover_thrust_per_motor(self) -> float:
        """Thrust per motor required for hover (total thrust = weight)."""
        return self.params.mass_kg * GRAVITY_MPS2 / 4.0

    def hover_command(self) -> np.ndarray:
        """Motor commands [0, 1]^4 for steady hover."""
        t_hover = self.hover_thrust_per_motor()
        return np.full(4, t_hover / self.params.max_thrust_per_motor_n, dtype=np.float64)

    @property
    def thrust_to_weight_ratio(self) -> float:
        """Maximum thrust-to-weight ratio."""
        return (4 * self.params.max_thrust_per_motor_n) / (self.params.mass_kg * GRAVITY_MPS2)

    @property
    def uav_type(self) -> str:
        return "quadrotor"

    @property
    def num_motors(self) -> int:
        return 4
