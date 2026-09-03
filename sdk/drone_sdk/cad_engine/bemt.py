"""
drone_sdk.cad_engine.bemt
=========================
Blade Element Momentum Theory (BEMT) for UAV Propeller Aerodynamics.

Implements a high-fidelity BEMT solver that couples momentum conservation
with sectional blade element aerodynamics, including:
1. Radial blade element discretization (chord and twist distributions).
2. Combined Prandtl tip-loss and hub-loss corrections.
3. Iterative solver for axial (a) and tangential (a') induced inflow factors.
4. Aerodynamic force integration: Thrust (T), Torque (Q), Mechanical Power (P).
5. Dimensionless coefficients: C_T, C_P, Figure of Merit (FM), and propulsive efficiency.

Literature:
  - Leishman, J. G. (2006). Principles of Helicopter Aerodynamics. Cambridge Univ. Press.
  - Glauert, H. (1935). Airplane Propellers. Aerodynamic Theory, Springer.

Python version: 3.9+
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Propeller geometry & aero definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PropellerGeometry:
    """Parametric geometry for a UAV propeller."""
    radius:        float = 0.127       # m (0.127 m = 5-inch radius / 10-inch diameter)
    hub_radius:    float = 0.015       # m (hub root radius)
    num_blades:    int   = 2
    root_chord:    float = 0.022       # m
    tip_chord:     float = 0.010       # m
    root_twist_deg: float = 24.0       # deg (pitch at root)
    tip_twist_deg:  float = 8.0        # deg (pitch at tip)
    cd0:           float = 0.015       # Profile zero-lift drag coefficient
    cl_alpha:      float = 5.73        # Lift curve slope (1/rad, ~2*pi)
    stall_aoa_deg: float = 14.0       # Stall angle of attack (deg)
    mass_kg:       float = 0.018       # Mass of single propeller (kg)

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def disk_area(self) -> float:
        return math.pi * (self.radius ** 2)

    def chord_at(self, r: float) -> float:
        """Linear or parabolic chord distribution along span."""
        r_rel = (r - self.hub_radius) / max(1e-6, self.radius - self.hub_radius)
        r_rel = max(0.0, min(1.0, r_rel))
        # Bell-like distribution tapering to tip
        return self.root_chord * (1.0 - 0.5 * r_rel)

    def twist_rad_at(self, r: float) -> float:
        """Twist angle in radians at radial station r."""
        r_rel = (r - self.hub_radius) / max(1e-6, self.radius - self.hub_radius)
        r_rel = max(0.0, min(1.0, r_rel))
        deg = self.root_twist_deg + (self.tip_twist_deg - self.root_twist_deg) * r_rel
        return math.radians(deg)

    @property
    def polar_moment_of_inertia(self) -> float:
        """Polar moment of inertia I_rotor = (1/12) * m * L^2."""
        length = 2.0 * self.radius
        return (1.0 / 12.0) * self.mass_kg * (length ** 2)


# ─────────────────────────────────────────────────────────────────────────────
#  BEMT analysis result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BEMTResult:
    """Complete aerodynamic output from BEMT propeller analysis."""
    rpm:              float
    omega_rad_s:      float
    v_inf:            float
    thrust_n:         float
    torque_nm:        float
    power_w:          float
    ct:               float        # Thrust coefficient: T / (rho * n^2 * D^4)
    cp:               float        # Power coefficient:  P / (rho * n^3 * D^5)
    figure_of_merit:  float        # Hover efficiency metric [0, 1]
    efficiency:       float        # Forward flight efficiency eta = T * V / P
    r_stations:       np.ndarray   # Radial station radii (m)
    dthrust_dr:       np.ndarray   # Sectional thrust per unit span (N/m)
    dtorque_dr:       np.ndarray   # Sectional torque per unit span (N*m/m)
    inflow_angles:    np.ndarray   # Inflow angle phi (rad)
    aoa_deg:          np.ndarray   # Section angle of attack (deg)

    def to_dict(self) -> dict:
        return {
            "rpm": round(self.rpm, 1),
            "thrust_n": round(self.thrust_n, 3),
            "torque_nm": round(self.torque_nm, 4),
            "power_w": round(self.power_w, 2),
            "ct": round(self.ct, 5),
            "cp": round(self.cp, 5),
            "figure_of_merit": round(self.figure_of_merit, 3),
            "efficiency": round(self.efficiency, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  BEMT Solver
# ─────────────────────────────────────────────────────────────────────────────

class BladeElementMomentumSolver:
    """Blade Element Momentum Theory solver for UAV multirotor propellers.

    Solves the coupled 1-D momentum conservation and 2-D blade element
    aerodynamic equations iteratively at each radial blade station.
    """

    def __init__(self, geometry: Optional[PropellerGeometry] = None) -> None:
        self._geom = geometry or PropellerGeometry()

    @property
    def geometry(self) -> PropellerGeometry:
        return self._geom

    def solve(
        self,
        rpm:          float,
        v_inf:        float = 0.0,
        rho:          float = 1.225,
        n_elements:   int   = 30,
        max_iter:     int   = 50,
        tol:          float = 1e-4,
    ) -> BEMTResult:
        """Compute propeller thrust, torque, and power at given RPM and freestream.

        Args:
            rpm:         Propeller rotational speed (revolutions per minute).
            v_inf:       Axial freestream velocity in m/s (0.0 for static hover).
            rho:         Air density in kg/m^3 (default: 1.225 kg/m^3 at sea level).
            n_elements:  Number of radial blade stations.
            max_iter:    Maximum Picard iteration loops per station.
            tol:         Convergence tolerance for induction factors.

        Returns:
            BEMTResult containing integrated forces and aerodynamic distributions.
        """
        geom = self._geom
        omega = rpm * (2.0 * math.pi / 60.0)
        n_rev_s = rpm / 60.0
        D = geom.diameter

        if rpm <= 1e-3:
            r_st = np.linspace(geom.hub_radius, geom.radius, n_elements)
            zeros = np.zeros(n_elements)
            return BEMTResult(
                rpm=rpm, omega_rad_s=0.0, v_inf=v_inf,
                thrust_n=0.0, torque_nm=0.0, power_w=0.0,
                ct=0.0, cp=0.0, figure_of_merit=0.0, efficiency=0.0,
                r_stations=r_st, dthrust_dr=zeros, dtorque_dr=zeros,
                inflow_angles=zeros, aoa_deg=zeros,
            )

        # Discretize blade span from hub to tip
        r_stations = np.linspace(geom.hub_radius, geom.radius, n_elements)
        dr = (geom.radius - geom.hub_radius) / max(1, n_elements - 1)

        dthrust_dr    = np.zeros(n_elements)
        dtorque_dr    = np.zeros(n_elements)
        inflow_angles = np.zeros(n_elements)
        aoa_deg       = np.zeros(n_elements)

        b = geom.num_blades

        for i, r in enumerate(r_stations):
            chord = geom.chord_at(r)
            theta = geom.twist_rad_at(r)
            v_theta = omega * r

            # Initial guess for induction factors
            a = 0.05
            a_prime = 0.005

            for _ in range(max_iter):
                # Effective velocities at blade section
                u_axial = v_inf + a * max(v_inf, 0.1) if v_inf > 0.5 else (v_inf + a * omega * r)
                u_tan   = v_theta * (1.0 - a_prime)

                phi = math.atan2(max(1e-4, u_axial), max(1e-4, u_tan))
                w_vel = math.sqrt(u_axial**2 + u_tan**2)

                # Angle of attack
                alpha = theta - phi
                alpha_deg_val = math.degrees(alpha)

                # Sectional 2D polars with stall limit
                if abs(alpha_deg_val) < geom.stall_aoa_deg:
                    cl = geom.cl_alpha * alpha
                    cd = geom.cd0 + 0.04 * (cl ** 2)
                else:
                    # Stalled flow: empirical flat plate stall
                    sign = 1.0 if alpha >= 0 else -1.0
                    cl = sign * 1.1 * math.sin(2.0 * alpha)
                    cd = geom.cd0 + 1.2 * (math.sin(alpha) ** 2)

                # Normal and tangential force coefficients
                cn = cl * math.cos(phi) - cd * math.sin(phi)
                ct = cl * math.sin(phi) + cd * math.cos(phi)

                # Prandtl tip & hub loss correction
                sin_phi = max(1e-4, abs(math.sin(phi)))
                f_tip_arg = (b * (geom.radius - r)) / (2.0 * r * sin_phi)
                f_tip = (2.0 / math.pi) * math.acos(math.exp(-min(50.0, f_tip_arg)))

                if r > geom.hub_radius:
                    f_hub_arg = (b * (r - geom.hub_radius)) / (2.0 * geom.hub_radius * sin_phi)
                    f_hub = (2.0 / math.pi) * math.acos(math.exp(-min(50.0, f_hub_arg)))
                else:
                    f_hub = 1.0

                f_total = max(1e-3, f_tip * f_hub)

                # Local solidity
                sigma = (b * chord) / (2.0 * math.pi * r)

                # Momentum balance equation for updated induction factor a
                denom_a = 4.0 * f_total * (sin_phi ** 2) + sigma * cn * math.cos(phi)
                if abs(denom_a) > 1e-6:
                    a_new = (sigma * cn) / denom_a
                else:
                    a_new = a

                # Momentum balance for tangential factor a'
                denom_ap = 4.0 * f_total * sin_phi * math.cos(phi) - sigma * ct
                if abs(denom_ap) > 1e-6:
                    ap_new = (sigma * ct) / denom_ap
                else:
                    ap_new = a_prime

                # Relaxation update
                a_new = max(-0.5, min(0.9, a_new))
                ap_new = max(-0.1, min(0.5, ap_new))

                diff = max(abs(a_new - a), abs(ap_new - a_prime))
                a = 0.7 * a + 0.3 * a_new
                a_prime = 0.7 * a_prime + 0.3 * ap_new

                if diff < tol:
                    break

            inflow_angles[i] = phi
            aoa_deg[i] = alpha_deg_val

            # Sectional aerodynamic forces
            w_vel = math.sqrt((v_inf * (1.0 + a))**2 + (v_theta * (1.0 - a_prime))**2)
            q_dyn = 0.5 * rho * (w_vel ** 2)

            dthrust_dr[i] = b * q_dyn * chord * cn
            dtorque_dr[i] = b * q_dyn * chord * ct * r

        # Spanwise trapezoidal integration
        thrust = float(np.trapezoid(dthrust_dr, r_stations)) if hasattr(np, "trapezoid") else float(np.trapz(dthrust_dr, r_stations))
        torque = float(np.trapezoid(dtorque_dr, r_stations)) if hasattr(np, "trapezoid") else float(np.trapz(dtorque_dr, r_stations))

        # Enforce non-negative thrust for forward pitch at positive RPM
        thrust = max(0.0, thrust)
        torque = max(0.0, torque)
        power  = torque * omega

        # Dimensionless coefficients
        if n_rev_s > 1e-4:
            ct_coeff = thrust / (rho * (n_rev_s ** 2) * (D ** 4))
            cp_coeff = power / (rho * (n_rev_s ** 3) * (D ** 5))
        else:
            ct_coeff = 0.0
            cp_coeff = 0.0

        # Figure of Merit (Hover)
        ideal_power = (thrust ** 1.5) / math.sqrt(2.0 * rho * geom.disk_area) if thrust > 0 else 0.0
        fm = ideal_power / max(power, 1e-5) if power > 0 else 0.0
        fm = min(1.0, max(0.0, fm))

        # Efficiency (Forward Flight)
        if v_inf > 1e-2 and power > 1e-3:
            eta = (thrust * v_inf) / power
            eta = min(1.0, max(0.0, eta))
        else:
            eta = fm

        return BEMTResult(
            rpm=rpm,
            omega_rad_s=omega,
            v_inf=v_inf,
            thrust_n=thrust,
            torque_nm=torque,
            power_w=power,
            ct=ct_coeff,
            cp=cp_coeff,
            figure_of_merit=fm,
            efficiency=eta,
            r_stations=r_stations,
            dthrust_dr=dthrust_dr,
            dtorque_dr=dtorque_dr,
            inflow_angles=inflow_angles,
            aoa_deg=aoa_deg,
        )

    def polar_sweep(
        self,
        rpm_array:  np.ndarray,
        v_inf:      float = 0.0,
        rho:        float = 1.225,
    ) -> Dict[str, np.ndarray]:
        """Sweep through an array of RPMs to produce propeller characteristic curves."""
        thrusts  = []
        torques  = []
        powers   = []
        fms      = []
        cts      = []
        cps      = []

        for rpm in rpm_array:
            res = self.solve(float(rpm), v_inf=v_inf, rho=rho)
            thrusts.append(res.thrust_n)
            torques.append(res.torque_nm)
            powers.append(res.power_w)
            fms.append(res.figure_of_merit)
            cts.append(res.ct)
            cps.append(res.cp)

        return {
            "rpm":             np.array(rpm_array),
            "thrust_n":        np.array(thrusts),
            "torque_nm":       np.array(torques),
            "power_w":         np.array(powers),
            "figure_of_merit": np.array(fms),
            "ct":              np.array(cts),
            "cp":              np.array(cps),
        }
