"""
drone_sdk.math_models.vtol_tiltrotor
=====================================
VTOL Tilt-Rotor Hybrid Mathematical Model — Full Nonlinear 6-DOF Dynamics.

Implements a unified dynamics model for a quadplane (quad-tiltrotor + wing)
that transitions between:
  - Hover mode (γ = 0°): pure multirotor thrust, rotors vertical
  - Cruise mode (γ = 90°): rotors tilted forward, wing provides lift
  - Transition corridor (0° < γ < 90°): blended rotor + wing forces

The nacelle tilt angle γ(t) drives the transition. At intermediate γ,
both rotor thrust vectoring AND aerodynamic surfaces contribute to flight.

Key equations:

  Force blending:
    F_body = F_rotor(γ) + F_aero(V_a, α) · σ(V_a)
    where σ(V_a) = smooth sigmoid scaling aerodynamic contribution

  Tilt geometry:
    F_rotor_body = R_tilt(γ) · [0, 0, -T]
    R_tilt = Ry(γ)  (rotation about body y-axis)

References:
  - Ducard, "Autonomous Quadrotor Flight Using a Vision System", ETH PhD 2009.
  - Ryll, Bülthoff, Giordano, "A Novel Overactuated Quadrotor Unmanned Aerial
    Vehicle", IEEE T-CST 2015.

Python version: 3.9+
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from .frames import quat_to_dcm, compute_wind_triangle
from .rigid_body import (
    ExternalWrench,
    InertiaParams,
    RigidBodyState,
    GRAVITY_MPS2,
)


class VTOLFlightPhase(str, Enum):
    HOVER = "hover"
    TRANSITION = "transition"
    CRUISE = "cruise"


@dataclass
class VTOLParams:
    """Physical parameters for a VTOL tilt-rotor quadplane.

    Default values: medium quadplane (~4 kg, 2m span).
    """
    mass_kg: float = 4.00

    # Inertia
    ixx: float = 0.180
    iyy: float = 0.300
    izz: float = 0.400
    ixz: float = 0.020

    # ── Rotor parameters (4 tilt-rotors) ──
    num_rotors: int = 4
    arm_length_m: float = 0.30
    max_thrust_per_motor_n: float = 12.0
    motor_torque_coeff: float = 0.015
    motor_time_constant_s: float = 0.035
    rotor_polar_inertia: float = 2.0e-5
    prop_radius_m: float = 0.152

    # ── Wing aerodynamics ──
    wing_span_m: float = 2.00
    wing_area_m2: float = 0.50
    mean_chord_m: float = 0.25

    CL0: float = 0.25
    CLa: float = 4.80
    CD0: float = 0.032
    K_drag: float = 0.044
    Cma: float = -0.55
    Cmde: float = -1.00
    CLde: float = 0.30

    # ── Tilt mechanism ──
    tilt_rate_max_dps: float = 30.0   # Maximum tilt rate (°/s)
    tilt_angle_min_deg: float = 0.0   # Full hover (vertical)
    tilt_angle_max_deg: float = 90.0  # Full cruise (horizontal)

    # ── Transition corridor ──
    transition_airspeed_min_mps: float = 8.0   # Below this → hover authority only
    transition_airspeed_max_mps: float = 18.0  # Above this → full wing authority

    # ── Control surfaces (active in cruise) ──
    max_elevator_rad: float = math.radians(25)
    max_aileron_rad: float = math.radians(25)

    @property
    def inertia_tensor(self) -> np.ndarray:
        return np.array([
            [self.ixx,  0,       self.ixz],
            [0,         self.iyy, 0],
            [self.ixz,  0,       self.izz],
        ], dtype=np.float64)

    @property
    def to_inertia_params(self) -> InertiaParams:
        return InertiaParams(mass_kg=self.mass_kg, inertia_tensor=self.inertia_tensor)


class VTOLTiltrotorModel:
    """Full nonlinear VTOL tilt-rotor dynamics model.

    Manages the tilt angle and blends rotor thrust with aerodynamic forces
    throughout the transition envelope.
    """

    def __init__(self, params: Optional[VTOLParams] = None) -> None:
        self.params = params or VTOLParams()
        self.tilt_angle_rad: float = 0.0  # Current nacelle tilt (0 = hover, π/2 = cruise)
        self._target_tilt_rad: float = 0.0
        self.motor_thrusts = np.zeros(self.params.num_rotors, dtype=np.float64)
        self.spin_dirs = np.array([1, -1, -1, 1], dtype=np.float64)  # X-config spin pattern

        # Control surface state
        self.elevator_rad: float = 0.0
        self.aileron_rad: float = 0.0

    @property
    def flight_phase(self) -> VTOLFlightPhase:
        gamma_deg = math.degrees(self.tilt_angle_rad)
        if gamma_deg < 5.0:
            return VTOLFlightPhase.HOVER
        elif gamma_deg > 85.0:
            return VTOLFlightPhase.CRUISE
        return VTOLFlightPhase.TRANSITION

    def set_tilt_target(self, angle_deg: float) -> None:
        """Set desired tilt angle (degrees). 0 = hover, 90 = cruise."""
        p = self.params
        clamped = max(p.tilt_angle_min_deg, min(p.tilt_angle_max_deg, angle_deg))
        self._target_tilt_rad = math.radians(clamped)

    def _update_tilt(self, dt: float) -> None:
        """Rate-limited tilt mechanism dynamics."""
        max_rate = math.radians(self.params.tilt_rate_max_dps)
        error = self._target_tilt_rad - self.tilt_angle_rad
        delta = max(-max_rate * dt, min(max_rate * dt, error))
        self.tilt_angle_rad += delta

    def _aero_blend_factor(self, airspeed: float) -> float:
        """Sigmoid blending factor for aerodynamic forces based on airspeed.

        0.0 at low airspeed (hover), 1.0 at high airspeed (cruise).
        """
        p = self.params
        v_min = p.transition_airspeed_min_mps
        v_max = p.transition_airspeed_max_mps
        if airspeed <= v_min:
            return 0.0
        if airspeed >= v_max:
            return 1.0
        # Smooth cosine blend
        t = (airspeed - v_min) / (v_max - v_min)
        return 0.5 * (1.0 - math.cos(math.pi * t))

    def _tilt_rotation_matrix(self) -> np.ndarray:
        """Rotation matrix for nacelle tilt about body y-axis: Ry(γ).

        Transforms rotor thrust from vertical [0, 0, -T] through tilt.
        """
        cg = math.cos(self.tilt_angle_rad)
        sg = math.sin(self.tilt_angle_rad)
        return np.array([
            [ cg,  0, -sg],
            [  0,  1,   0],
            [ sg,  0,  cg],
        ], dtype=np.float64)

    def compute_wrench(
        self,
        state: RigidBodyState,
        motor_commands: np.ndarray,
        elevator_cmd: float = 0.0,
        aileron_cmd: float = 0.0,
        dt: float = 0.005,
        air_density: float = 1.225,
        wind_ned: Optional[np.ndarray] = None,
    ) -> ExternalWrench:
        """Compute blended rotor + aerodynamic forces/torques."""
        p = self.params
        rho = air_density
        n_rot = p.num_rotors

        # 1. Update tilt angle
        self._update_tilt(dt)

        # 2. Motor lag
        cmds = np.clip(motor_commands[:n_rot], 0.0, 1.0)
        target_thrusts = cmds * p.max_thrust_per_motor_n
        alpha_m = min(1.0, dt / max(1e-4, p.motor_time_constant_s))
        self.motor_thrusts += (target_thrusts - self.motor_thrusts) * alpha_m
        thrusts = self.motor_thrusts.copy()

        # 3. Tilt rotation: transform rotor thrust into body frame
        R_tilt = self._tilt_rotation_matrix()
        total_thrust = float(np.sum(thrusts))
        # In rotor frame, thrust is along -z_rotor (up)
        thrust_rotor_frame = np.array([0.0, 0.0, -total_thrust], dtype=np.float64)
        F_rotor_body = R_tilt @ thrust_rotor_frame

        # 4. Rotor torques (mixing in body frame, affected by tilt for yaw authority)
        arm = p.arm_length_m / math.sqrt(2.0)
        tau_roll  = (thrusts[0] + thrusts[2] - thrusts[1] - thrusts[3]) * arm
        tau_pitch = (thrusts[0] + thrusts[1] - thrusts[2] - thrusts[3]) * arm
        tau_yaw   = (thrusts[0] + thrusts[3] - thrusts[1] - thrusts[2]) * p.motor_torque_coeff

        # Yaw authority reduces with tilt (rotor reaction torque axis tilts)
        tau_yaw *= math.cos(self.tilt_angle_rad)

        tau_rotor = np.array([tau_roll, tau_pitch, tau_yaw], dtype=np.float64)

        # 5. Gyroscopic torque
        omega_rotors = np.sqrt(np.maximum(thrusts, 0.0) / max(1e-9, p.max_thrust_per_motor_n)) * 1000.0
        net_h = p.rotor_polar_inertia * np.sum(omega_rotors * self.spin_dirs)
        tau_gyro = np.array([
            -state.omega_body[1] * net_h,
             state.omega_body[0] * net_h,
             0.0,
        ], dtype=np.float64)

        # 6. Aerodynamic forces (wing, blended by airspeed)
        wind = wind_ned if wind_ned is not None else np.zeros(3)
        wt = compute_wind_triangle(state.vel_body, wind, state.quat)
        Va = wt.airspeed_mps
        alpha = wt.alpha_rad

        sigma = self._aero_blend_factor(Va)  # 0 in hover, 1 in cruise

        F_aero_body = np.zeros(3, dtype=np.float64)
        tau_aero = np.zeros(3, dtype=np.float64)

        if sigma > 0.01 and Va > 1.0:
            q_bar = 0.5 * rho * Va ** 2
            S = p.wing_area_m2
            b = p.wing_span_m
            c = p.mean_chord_m

            # Update control surfaces
            self.elevator_rad += (elevator_cmd * p.max_elevator_rad - self.elevator_rad) * min(1.0, dt / 0.05)
            self.aileron_rad += (aileron_cmd * p.max_aileron_rad - self.aileron_rad) * min(1.0, dt / 0.05)

            CL = p.CL0 + p.CLa * alpha + p.CLde * self.elevator_rad
            CD = p.CD0 + p.K_drag * CL ** 2
            Cm = p.Cma * alpha + p.Cmde * self.elevator_rad

            F_lift = q_bar * S * CL * sigma
            F_drag = q_bar * S * CD * sigma

            ca, sa = math.cos(alpha), math.sin(alpha)
            F_aero_body[0] = -F_drag * ca + F_lift * sa
            F_aero_body[2] = -F_drag * sa - F_lift * ca

            tau_aero[1] = q_bar * S * c * Cm * sigma  # Pitch from wing
            # Aileron roll authority
            Cl_ail = 0.15 * self.aileron_rad
            tau_aero[0] = q_bar * S * b * Cl_ail * sigma

        # 7. Body drag
        v_air = state.vel_body
        drag_body = -0.5 * rho * 1.0 * 0.03 * v_air * np.abs(v_air)

        # 8. Total
        total_force = F_rotor_body + F_aero_body + drag_body
        total_torque = tau_rotor + tau_gyro + tau_aero

        return ExternalWrench(force_body=total_force, torque_body=total_torque)

    def hover_command(self) -> np.ndarray:
        """Motor commands for hover (tilt = 0°)."""
        t_per = self.params.mass_kg * GRAVITY_MPS2 / self.params.num_rotors
        return np.full(self.params.num_rotors, t_per / self.params.max_thrust_per_motor_n, dtype=np.float64)

    def begin_transition(self) -> None:
        """Initiate hover-to-cruise transition."""
        self.set_tilt_target(90.0)

    def begin_back_transition(self) -> None:
        """Initiate cruise-to-hover back-transition."""
        self.set_tilt_target(0.0)

    @property
    def uav_type(self) -> str:
        return "vtol_tiltrotor"

    @property
    def num_motors(self) -> int:
        return self.params.num_rotors

    @property
    def transition_progress(self) -> float:
        """Transition progress: 0.0 = full hover, 1.0 = full cruise."""
        return self.tilt_angle_rad / (math.pi / 2.0)
