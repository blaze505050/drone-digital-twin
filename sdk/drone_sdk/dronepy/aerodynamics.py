"""
dronepy.aerodynamics
====================
Modular Aerodynamic Models for DronePy 6-DOF Flight Dynamics.
Supports analytical parasitic drag, OpenFOAM CFD databases, and PINN neural surrogates.
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class AerodynamicsModel(abc.ABC):
    """Abstract base class for all aerodynamic force and moment providers.

    The 6-DOF dynamics solver queries this interface without needing to know
    whether forces originate from analytical equations, CFD tables, or neural surrogates.
    """

    @abc.abstractmethod
    def compute_forces_and_moments(
        self,
        v_air_body: np.ndarray,
        omega_body: np.ndarray,
        air_density: float = 1.225,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute aerodynamic forces and moments in the body-fixed frame (FRD).

        Parameters
        ----------
        v_air_body : np.ndarray (3,)
            Relative airspeed vector in body frame: v_body - v_wind_body (m/s).
        omega_body : np.ndarray (3,)
            Body angular velocity vector [p, q, r] in rad/s.
        air_density : float
            Local ambient air density rho in kg/m^3.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (force_body_n, moment_body_nm)
        """
        raise NotImplementedError


@dataclass
class AnalyticalDrag(AerodynamicsModel):
    """Directional quadratic aerodynamic parasitic drag and rotational damping.

    F_drag,i = -0.5 * rho * Cd_i * A_i * |v_i| * v_i
    M_damp,i = -0.5 * rho * C_damp_i * |omega_i| * omega_i
    """
    cd_x: float = 0.85          # Forward frontal drag coefficient
    cd_y: float = 0.85          # Lateral sideways drag coefficient
    cd_z: float = 1.25          # Vertical ventral/downward drag coefficient
    area_frontal_m2: float = 0.025   # Frontal projection area
    area_side_m2: float = 0.030      # Side projection area
    area_planform_m2: float = 0.075  # Top/bottom projection area
    rotational_damping: np.ndarray = field(
        default_factory=lambda: np.array([0.002, 0.002, 0.003], dtype=np.float64)
    )

    @property
    def drag_coefficient(self) -> float:
        """Effective translational drag coefficient."""
        return float((self.cd_x + self.cd_y + self.cd_z) / 3.0)

    @drag_coefficient.setter
    def drag_coefficient(self, value: float) -> None:
        ratio = float(value) / max(1e-6, self.drag_coefficient)
        self.cd_x *= ratio
        self.cd_y *= ratio
        self.cd_z *= ratio

    def compute_forces_and_moments(
        self,
        v_air_body: np.ndarray,
        omega_body: np.ndarray,
        air_density: float = 1.225,
    ) -> Tuple[np.ndarray, np.ndarray]:
        v = np.asarray(v_air_body, dtype=np.float64)
        omega = np.asarray(omega_body, dtype=np.float64)

        # Directional drag forces
        cd_areas = np.array([
            self.cd_x * self.area_frontal_m2,
            self.cd_y * self.area_side_m2,
            self.cd_z * self.area_planform_m2,
        ], dtype=np.float64)

        # F_drag = -0.5 * rho * (Cd * A) * |v| * v
        force_body = -0.5 * air_density * cd_areas * np.abs(v) * v

        # Rotational aerodynamic damping
        moment_body = -self.rotational_damping * omega * np.abs(omega)

        return force_body, moment_body


class OpenFOAMAeroDatabase(AerodynamicsModel):
    """Aerodynamic model querying an OpenFOAM CFD coefficient database or lookup table.

    Interpolates C_L(alpha), C_D(alpha), C_m(alpha) as functions of angle of attack (AoA).
    """

    def __init__(
        self,
        reference_area_m2: float = 0.05,
        aoa_deg_table: Optional[np.ndarray] = None,
        cd_table: Optional[np.ndarray] = None,
        cl_table: Optional[np.ndarray] = None,
        cm_table: Optional[np.ndarray] = None,
    ) -> None:
        self.ref_area = reference_area_m2
        if aoa_deg_table is not None and cd_table is not None:
            self.aoa_table = np.asarray(aoa_deg_table, dtype=np.float64)
            self.cd_table = np.asarray(cd_table, dtype=np.float64)
            self.cl_table = np.asarray(cl_table, dtype=np.float64) if cl_table is not None else np.zeros_like(self.cd_table)
            self.cm_table = np.asarray(cm_table, dtype=np.float64) if cm_table is not None else np.zeros_like(self.cd_table)
        else:
            # Synthetic default OpenFOAM polar for drone body
            self.aoa_table = np.linspace(-30.0, 30.0, 13)
            self.cd_table = 0.8 + 1.2 * (np.sin(np.radians(self.aoa_table)) ** 2)
            self.cl_table = 0.6 * np.sin(np.radians(2.0 * self.aoa_table))
            self.cm_table = -0.05 * np.sin(np.radians(self.aoa_table))

    def compute_forces_and_moments(
        self,
        v_air_body: np.ndarray,
        omega_body: np.ndarray,
        air_density: float = 1.225,
    ) -> Tuple[np.ndarray, np.ndarray]:
        v = np.asarray(v_air_body, dtype=np.float64)
        speed = float(np.linalg.norm(v))
        if speed < 1e-4:
            return np.zeros(3), np.zeros(3)

        # Angle of attack alpha = arctan2(v_z, v_x)
        aoa_deg = math.degrees(math.atan2(v[2], v[0])) if abs(v[0]) > 1e-4 else 0.0
        q_dyn = 0.5 * air_density * (speed ** 2) * self.ref_area

        cd = float(np.interp(aoa_deg, self.aoa_table, self.cd_table))
        cl = float(np.interp(aoa_deg, self.aoa_table, self.cl_table))
        cm = float(np.interp(aoa_deg, self.aoa_table, self.cm_table))

        # Drag acts opposite to velocity vector
        drag_force = -q_dyn * cd * (v / speed)
        # Lift acts upwards (negative body z)
        lift_force = np.array([0.0, 0.0, -q_dyn * cl])
        total_force = drag_force + lift_force
        total_moment = np.array([0.0, q_dyn * cm * 0.25, 0.0])

        return total_force, total_moment


class NeuralAeroModel(AerodynamicsModel):
    """Physics-Informed Neural Network (PINN) or surrogate aerodynamic model.

    Evaluates aerodynamic forces and pitching moments using a compact neural network
    f(speed, alpha, beta) -> [Fx, Fy, Fz, Mx, My, Mz].
    """

    def __init__(self, surrogate_fn: Optional[Callable[[float, float, float], np.ndarray]] = None) -> None:
        self.surrogate_fn = surrogate_fn

    def compute_forces_and_moments(
        self,
        v_air_body: np.ndarray,
        omega_body: np.ndarray,
        air_density: float = 1.225,
    ) -> Tuple[np.ndarray, np.ndarray]:
        v = np.asarray(v_air_body, dtype=np.float64)
        speed = float(np.linalg.norm(v))
        if speed < 1e-4:
            return np.zeros(3), np.zeros(3)

        aoa = math.atan2(v[2], max(1e-4, v[0]))
        beta = math.asin(np.clip(v[1] / speed, -1.0, 1.0))

        if self.surrogate_fn is not None:
            preds = self.surrogate_fn(speed, aoa, beta)
            return preds[:3], preds[3:]
        else:
            # Physics-guided surrogate default
            q_inf = 0.5 * air_density * (speed ** 2)
            fx = -q_inf * 0.025 * math.cos(aoa)
            fy = -q_inf * 0.030 * math.sin(beta)
            fz = -q_inf * 0.075 * math.sin(aoa)
            return np.array([fx, fy, fz]), np.zeros(3)
