"""
dronepy.propellers
==================
Propeller Aerodynamic Models for DronePy.
Wraps UIUC wind-tunnel polars and Blade Element Momentum Theory (BEMT).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from drone_sdk.cad_engine.bemt import (
    BladeElementMomentumSolver,
    PropellerGeometry,
)
from drone_sdk.configuration.schema import PropellerConfig
from drone_sdk.physics.actuators import PropellerAerodynamics


@dataclass
class Propeller:
    """Parametric multirotor propeller model.

    Parameters
    ----------
    diameter_in : float
        Propeller diameter in inches (e.g. 10.0 for 10-inch prop).
    pitch_in : float
        Propeller pitch in inches (e.g. 4.5 for 4.5-inch pitch).
    num_blades : int
        Number of blades (typically 2 or 3).
    ct0 : float
        Static thrust coefficient C_T at J=0.
    cp0 : float
        Static power coefficient C_P at J=0.
    mass_g : float
        Mass of a single propeller in grams.
    """
    diameter_in: float = 10.0
    pitch_in: float = 4.5
    num_blades: int = 2
    ct0: float = 0.112
    cp0: float = 0.048
    mass_g: float = 14.5

    @property
    def diameter_m(self) -> float:
        return self.diameter_in * 0.0254

    @property
    def pitch_m(self) -> float:
        return self.pitch_in * 0.0254

    @property
    def radius_m(self) -> float:
        return self.diameter_m / 2.0

    @property
    def mass_kg(self) -> float:
        return self.mass_g / 1000.0

    def compute_thrust_and_torque(
        self,
        rpm: float,
        axial_inflow_mps: float = 0.0,
        air_density: float = 1.225,
    ) -> Tuple[float, float, float]:
        """Compute thrust (N), reaction torque (N·m), and power (W) at given RPM."""
        if rpm <= 1.0:
            return 0.0, 0.0, 0.0

        n = rpm / 60.0
        d = self.diameter_m
        j = axial_inflow_mps / (n * d) if n > 1e-3 else 0.0

        j_max = max(0.4, self.pitch_m / self.diameter_m * 1.2)
        ct = max(0.0, self.ct0 * (1.0 - (j / j_max) ** 2))
        cp = max(1e-4, self.cp0 * (1.0 - 0.5 * (j / j_max) ** 2))

        thrust_n = ct * air_density * (n ** 2) * (d ** 4)
        power_w = cp * air_density * (n ** 3) * (d ** 5)
        omega_rad_s = 2.0 * math.pi * n
        torque_nm = power_w / omega_rad_s if omega_rad_s > 1e-3 else 0.0

        return max(0.0, thrust_n), max(0.0, torque_nm), max(0.0, power_w)


class UIUCPropeller(Propeller):
    """Propeller calibrated directly against UIUC wind-tunnel experimental datasets."""

    def __init__(
        self,
        dataset_name: str = "uiuc_apc_10x4.5",
        diameter_in: float = 10.0,
        pitch_in: float = 4.5,
        num_blades: int = 2,
    ) -> None:
        super().__init__(diameter_in=diameter_in, pitch_in=pitch_in, num_blades=num_blades)
        p_cfg = PropellerConfig(
            diameter_m=self.diameter_m,
            pitch_m=self.pitch_m,
            polar_dataset=dataset_name,
        )
        self._aero = PropellerAerodynamics(p_cfg)

    def compute_thrust_and_torque(
        self,
        rpm: float,
        axial_inflow_mps: float = 0.0,
        air_density: float = 1.225,
    ) -> Tuple[float, float, float]:
        return self._aero.compute_aerodynamics(rpm, axial_inflow_mps, air_density)


class BEMTPropeller(Propeller):
    """Propeller using full iterative Blade Element Momentum Theory (BEMT)."""

    def __init__(
        self,
        diameter_in: float = 10.0,
        pitch_in: float = 4.5,
        num_blades: int = 2,
        root_chord_m: float = 0.022,
        tip_chord_m: float = 0.010,
        num_radial_elements: int = 20,
    ) -> None:
        super().__init__(diameter_in=diameter_in, pitch_in=pitch_in, num_blades=num_blades)
        self.geom = PropellerGeometry(
            radius=self.radius_m,
            hub_radius=0.015,
            num_blades=num_blades,
            root_chord=root_chord_m,
            tip_chord=tip_chord_m,
            root_twist_deg=22.0,
            tip_twist_deg=7.5,
        )
        self.num_radial_elements = num_radial_elements
        self.solver = BladeElementMomentumSolver(self.geom)

    def compute_thrust_and_torque(
        self,
        rpm: float,
        axial_inflow_mps: float = 0.0,
        air_density: float = 1.225,
    ) -> Tuple[float, float, float]:
        if rpm <= 10.0:
            return 0.0, 0.0, 0.0
        res = self.solver.solve(
            rpm=rpm,
            v_inf=axial_inflow_mps,
            rho=air_density,
            n_elements=self.num_radial_elements,
        )
        return max(0.0, res.thrust_n), max(0.0, res.torque_nm), max(0.0, res.power_w)
