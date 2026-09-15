"""
drone_sdk.math_models.fixed_wing
=================================
Fixed-Wing Aircraft Mathematical Model — Full Nonlinear 6-DOF Dynamics.

Implements the complete aerodynamic force and moment model for a
conventional fixed-wing aircraft with:

  - Lift, drag, sideforce coefficients as nonlinear functions of
    angle of attack α, sideslip β, and control surface deflections.
  - Stability and control derivatives (dimensional and non-dimensional).
  - Propeller thrust model (C_T based).
  - Control surface actuator dynamics (first-order lag).
  - Trim solver for steady level flight.
  - Linearisation around trim → state-space (A, B) matrices.

The aerodynamic model follows the standard coefficient buildup::

    C_L = C_L0 + C_Lα·α + C_Lδe·δe + C_Lq·(q̄c/2V)
    C_D = C_D0 + K·C_L²  (parabolic drag polar)
    C_m = C_m0 + C_mα·α + C_mδe·δe + C_mq·(q̄c/2V)
    ...

Axes: body-fixed FRD (Forward-Right-Down).

References:
  - Nelson, "Flight Stability and Automatic Control", 2nd Ed, 1998.
  - Stevens, Lewis & Johnson, "Aircraft Control and Simulation", 3rd Ed, 2016.
  - Stengel, "Flight Dynamics", Princeton University Press, 2004.

Python version: 3.9+
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .frames import quat_to_dcm, compute_wind_triangle, WindTriangle
from .rigid_body import (
    ExternalWrench,
    InertiaParams,
    RigidBodyState,
    GRAVITY_MPS2,
)


@dataclass
class FixedWingParams:
    """Physical and aerodynamic parameters for a fixed-wing UAV.

    Default values approximate a small trainer-class UAV (wingspan ~2m).
    """
    mass_kg: float = 3.50
    wing_span_m: float = 2.00
    wing_area_m2: float = 0.45
    mean_chord_m: float = 0.225  # MAC = S / b

    # Inertia (about CG)
    ixx: float = 0.120
    iyy: float = 0.200
    izz: float = 0.280
    ixz: float = 0.015

    # ── Longitudinal aerodynamic coefficients ──
    CL0: float = 0.28              # Zero-α lift coefficient
    CLa: float = 4.44              # Lift curve slope ∂C_L/∂α (1/rad)
    CLq: float = 7.95              # Pitch rate lift derivative
    CLde: float = 0.355            # Elevator lift effectiveness

    CD0: float = 0.030             # Parasite drag coefficient
    K_drag: float = 0.0468         # Induced drag factor: C_D = C_D0 + K·C_L²
    CDde: float = 0.0              # Elevator drag (usually negligible)

    Cm0: float = 0.045             # Zero-α pitching moment
    Cma: float = -0.680            # Pitch moment slope (negative = stable)
    Cmq: float = -12.40            # Pitch damping derivative
    Cmde: float = -1.13            # Elevator pitch effectiveness

    # ── Lateral-directional aerodynamic coefficients ──
    CYb: float = -0.310            # Sideforce due to sideslip
    CYdr: float = 0.187            # Sideforce due to rudder
    CYp: float = 0.0               # Sideforce due to roll rate (usually small)

    Clb: float = -0.090            # Roll moment due to sideslip (dihedral effect)
    Clp: float = -0.470            # Roll damping
    Clr: float = 0.096             # Roll due to yaw rate
    Clda: float = 0.178            # Aileron roll effectiveness
    Cldr: float = 0.0024           # Rudder roll coupling

    Cnb: float = 0.065             # Yaw moment due to sideslip (weathervane stability)
    Cnp: float = -0.030            # Yaw due to roll rate (adverse yaw)
    Cnr: float = -0.099            # Yaw damping
    Cnda: float = -0.053           # Aileron yaw coupling (adverse yaw)
    Cndr: float = -0.069           # Rudder yaw effectiveness

    # ── Propulsion ──
    prop_diameter_m: float = 0.356  # 14" propeller
    prop_ct0: float = 0.110         # Static thrust coefficient at J=0
    max_rpm: float = 9000.0         # Maximum engine RPM

    # ── Control surface limits ──
    max_aileron_rad: float = math.radians(25)
    max_elevator_rad: float = math.radians(25)
    max_rudder_rad: float = math.radians(25)
    actuator_tau_s: float = 0.05    # Control surface time constant

    # ── Stall ──
    alpha_stall_rad: float = math.radians(15)
    CL_max: float = 1.40

    @property
    def aspect_ratio(self) -> float:
        return self.wing_span_m ** 2 / self.wing_area_m2

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


@dataclass
class ControlSurfaces:
    """Current control surface deflections (rad)."""
    aileron: float = 0.0    # δa, positive = right wing down (right roll)
    elevator: float = 0.0   # δe, positive = trailing edge down (pitch up)
    rudder: float = 0.0     # δr, positive = trailing edge left (nose left)
    throttle: float = 0.0   # δt, normalised [0, 1]


class FixedWingModel:
    """Full nonlinear fixed-wing aerodynamic dynamics model.

    Computes body forces and torques from aerodynamic state and control
    surface deflections using the coefficient buildup method.
    """

    def __init__(self, params: Optional[FixedWingParams] = None) -> None:
        self.params = params or FixedWingParams()
        self.surfaces = ControlSurfaces()
        self._target_surfaces = ControlSurfaces()

    def compute_aero_coefficients(
        self,
        alpha: float,
        beta: float,
        p_hat: float,
        q_hat: float,
        r_hat: float,
        da: float,
        de: float,
        dr: float,
    ) -> Tuple[float, float, float, float, float, float]:
        """Compute all 6 aerodynamic coefficients.

        Args:
            alpha: Angle of attack (rad).
            beta: Sideslip angle (rad).
            p_hat, q_hat, r_hat: Non-dimensionalised angular rates.
            da, de, dr: Control surface deflections (rad).

        Returns:
            (CL, CD, CY, Cl, Cm, Cn)
        """
        par = self.params

        # Lift (with post-stall model)
        CL_linear = par.CL0 + par.CLa * alpha + par.CLq * q_hat + par.CLde * de
        if abs(alpha) > par.alpha_stall_rad:
            # Viterna flat-plate post-stall model
            CL = par.CL_max * math.sin(2 * alpha) * math.copysign(1, alpha)
        else:
            CL = CL_linear

        # Drag (parabolic polar)
        CD = par.CD0 + par.K_drag * CL_linear ** 2 + par.CDde * abs(de)

        # Sideforce
        CY = par.CYb * beta + par.CYdr * dr + par.CYp * p_hat

        # Rolling moment
        Cl = par.Clb * beta + par.Clp * p_hat + par.Clr * r_hat + par.Clda * da + par.Cldr * dr

        # Pitching moment
        Cm = par.Cm0 + par.Cma * alpha + par.Cmq * q_hat + par.Cmde * de

        # Yawing moment
        Cn = par.Cnb * beta + par.Cnp * p_hat + par.Cnr * r_hat + par.Cnda * da + par.Cndr * dr

        return CL, CD, CY, Cl, Cm, Cn

    def compute_wrench(
        self,
        state: RigidBodyState,
        controls: Optional[ControlSurfaces] = None,
        dt: float = 0.005,
        air_density: float = 1.225,
        wind_ned: Optional[np.ndarray] = None,
    ) -> ExternalWrench:
        """Compute body forces and torques from aerodynamic state.

        Args:
            state: Current rigid body state.
            controls: Target control surface commands (applies lag).
            dt: Timestep for actuator lag.
            air_density: ρ (kg/m³).
            wind_ned: Wind vector in NED frame (m/s).
        """
        par = self.params
        rho = air_density

        # Update control surfaces with first-order lag
        if controls is not None:
            self._target_surfaces = controls
        alpha_act = min(1.0, dt / max(1e-4, par.actuator_tau_s))
        self.surfaces.aileron += (self._target_surfaces.aileron - self.surfaces.aileron) * alpha_act
        self.surfaces.elevator += (self._target_surfaces.elevator - self.surfaces.elevator) * alpha_act
        self.surfaces.rudder += (self._target_surfaces.rudder - self.surfaces.rudder) * alpha_act
        self.surfaces.throttle += (self._target_surfaces.throttle - self.surfaces.throttle) * alpha_act

        # Clamp control surfaces
        self.surfaces.aileron = max(-par.max_aileron_rad, min(par.max_aileron_rad, self.surfaces.aileron))
        self.surfaces.elevator = max(-par.max_elevator_rad, min(par.max_elevator_rad, self.surfaces.elevator))
        self.surfaces.rudder = max(-par.max_rudder_rad, min(par.max_rudder_rad, self.surfaces.rudder))
        self.surfaces.throttle = max(0.0, min(1.0, self.surfaces.throttle))

        # Wind triangle
        wind = wind_ned if wind_ned is not None else np.zeros(3)
        wt = compute_wind_triangle(state.vel_body, wind, state.quat)
        Va = wt.airspeed_mps
        alpha = wt.alpha_rad
        beta = wt.beta_rad

        if Va < 0.5:
            # Below stall speed — only gravity and thrust apply
            thrust = self._compute_thrust(self.surfaces.throttle, 0.0, rho)
            F_thrust = np.array([thrust, 0.0, 0.0], dtype=np.float64)
            return ExternalWrench(force_body=F_thrust, torque_body=np.zeros(3))

        # Dynamic pressure
        q_bar = 0.5 * rho * Va ** 2
        S = par.wing_area_m2
        b = par.wing_span_m
        c = par.mean_chord_m

        # Non-dimensionalised angular rates
        p, q, r = state.omega_body
        p_hat = p * b / (2 * Va)
        q_hat = q * c / (2 * Va)
        r_hat = r * b / (2 * Va)

        # Aerodynamic coefficients
        CL, CD, CY, Cl, Cm, Cn = self.compute_aero_coefficients(
            alpha, beta, p_hat, q_hat, r_hat,
            self.surfaces.aileron, self.surfaces.elevator, self.surfaces.rudder,
        )

        # Forces in wind frame → body frame
        # Wind frame: x_w along airspeed, z_w perpendicular in plane of symmetry
        ca, sa = math.cos(alpha), math.sin(alpha)
        cb, sb = math.cos(beta), math.sin(beta)

        # Aerodynamic forces in stability frame → body frame
        F_lift = q_bar * S * CL
        F_drag = q_bar * S * CD
        F_side = q_bar * S * CY

        # Stability → body rotation (about y by α)
        Fx_aero = -F_drag * ca + F_lift * sa
        Fy_aero = F_side
        Fz_aero = -F_drag * sa - F_lift * ca

        # Propeller thrust
        thrust = self._compute_thrust(self.surfaces.throttle, Va, rho)
        Fx_total = Fx_aero + thrust

        # Moments
        L_moment = q_bar * S * b * Cl   # Rolling moment
        M_moment = q_bar * S * c * Cm   # Pitching moment
        N_moment = q_bar * S * b * Cn   # Yawing moment

        return ExternalWrench(
            force_body=np.array([Fx_total, Fy_aero, Fz_aero], dtype=np.float64),
            torque_body=np.array([L_moment, M_moment, N_moment], dtype=np.float64),
        )

    def _compute_thrust(self, throttle: float, airspeed: float, rho: float) -> float:
        """Propeller thrust model: T = C_T · ρ · n² · D⁴."""
        par = self.params
        n = throttle * par.max_rpm / 60.0  # rev/s
        D = par.prop_diameter_m

        if n < 1.0:
            return 0.0

        J = airspeed / (n * D) if n > 1.0 else 0.0
        # Simple quadratic thrust coefficient model
        ct = par.prop_ct0 * max(0.0, 1.0 - (J / 0.7) ** 2)
        return ct * rho * n ** 2 * D ** 4

    def trim_solver(
        self,
        airspeed_target: float = 15.0,
        altitude_m: float = 100.0,
        air_density: float = 1.225,
        max_iter: int = 50,
    ) -> Tuple[float, float, float, bool]:
        """Find trim condition for steady level flight.

        Solves for (α₀, δe₀, δt₀) such that:
          - Lift = Weight (vertical equilibrium)
          - Cm = 0 (pitch moment balance)
          - Thrust = Drag (longitudinal equilibrium)

        Returns:
            (alpha_trim, elevator_trim, throttle_trim, converged)
        """
        par = self.params
        W = par.mass_kg * GRAVITY_MPS2
        q_bar = 0.5 * air_density * airspeed_target ** 2
        S = par.wing_area_m2

        # Initial guesses
        CL_req = W / (q_bar * S)
        alpha = (CL_req - par.CL0) / par.CLa
        de = 0.0
        dt = 0.3

        converged = False
        for _ in range(max_iter):
            CL = par.CL0 + par.CLa * alpha + par.CLde * de
            CD = par.CD0 + par.K_drag * CL ** 2
            Cm = par.Cm0 + par.Cma * alpha + par.Cmde * de

            # Residuals
            lift_err = CL - CL_req
            moment_err = Cm

            # Newton step (2×2 Jacobian)
            dCL_da = par.CLa
            dCL_dde = par.CLde
            dCm_da = par.Cma
            dCm_dde = par.Cmde

            det = dCL_da * dCm_dde - dCL_dde * dCm_da
            if abs(det) < 1e-10:
                break

            d_alpha = (-dCm_dde * lift_err + dCL_dde * moment_err) / det
            d_de = (dCm_da * lift_err - dCL_da * moment_err) / det

            alpha += d_alpha
            de += d_de

            # Throttle from drag balance
            D = q_bar * S * CD
            T_max = par.prop_ct0 * air_density * (par.max_rpm / 60.0) ** 2 * par.prop_diameter_m ** 4
            dt = D / max(1.0, T_max)

            if abs(lift_err) < 1e-5 and abs(moment_err) < 1e-5:
                converged = True
                break

        return float(alpha), float(de), float(np.clip(dt, 0, 1)), converged

    def linearise_at_trim(
        self,
        alpha_trim: float,
        airspeed: float = 15.0,
        air_density: float = 1.225,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linearise longitudinal dynamics around trim to get (A_lon, B_lon).

        State: [Δu, Δw, Δq, Δθ]
        Control: [Δδe, Δδt]

        Returns:
            (A_lon, B_lon) state-space matrices for short-period and phugoid analysis.
        """
        par = self.params
        m = par.mass_kg
        Iyy = par.iyy
        q_bar = 0.5 * air_density * airspeed ** 2
        S = par.wing_area_m2
        c = par.mean_chord_m
        V = airspeed
        g = GRAVITY_MPS2

        # Dimensional stability derivatives
        Xu = -q_bar * S * 2 * par.CD0 / (m * V)
        Xw = q_bar * S * (par.CLa - 2 * par.K_drag * par.CLa * (par.CL0 + par.CLa * alpha_trim)) / (m * V)
        Zu = -q_bar * S * 2 * (par.CL0 + par.CLa * alpha_trim) / (m * V)
        Zw = -q_bar * S * (par.CLa + par.CD0) / (m * V)
        Zq = -q_bar * S * c * par.CLq / (2 * m * V)
        Mu = 0.0
        Mw = q_bar * S * c * par.Cma / (Iyy * V)
        Mq = q_bar * S * c ** 2 * par.Cmq / (2 * Iyy * V)

        # Control derivatives
        Zde = -q_bar * S * par.CLde / m
        Mde = q_bar * S * c * par.Cmde / Iyy

        A_lon = np.array([
            [Xu,   Xw,    0,   -g],
            [Zu,   Zw,    V+Zq, 0],
            [Mu,   Mw,    Mq,   0],
            [0,    0,     1,    0],
        ], dtype=np.float64)

        B_lon = np.array([
            [0,    0],
            [Zde,  0],
            [Mde,  0],
            [0,    0],
        ], dtype=np.float64)

        return A_lon, B_lon

    @property
    def uav_type(self) -> str:
        return "fixed_wing"

    @property
    def num_motors(self) -> int:
        return 1
