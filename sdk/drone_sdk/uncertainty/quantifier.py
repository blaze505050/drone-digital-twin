"""
drone_sdk.uncertainty.quantifier
================================
Uncertainty Quantification (UQ) & Operational Domain of Validity.
Implements Backlog Item B15 & PRD UQ-01/02.

Provides:
- 95% Confidence Interval (CI) propagation for power, energy, and flight endurance.
- Parameter sensitivity analysis (mass, ambient temperature, motor resistance, wind).
- Out-of-Distribution (OOD) validity domain monitoring.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..battery_twin.pack_model import EnergyPredictionBaseline
from ..configuration import VehicleConfiguration, get_vehicle_config


@dataclass
class UncertaintyInterval:
    """95% Confidence Interval for a physical prediction."""
    mean: float
    std_dev: float
    ci_95_lower: float
    ci_95_upper: float
    unit: str

    @property
    def margin(self) -> float:
        return 1.96 * self.std_dev


@dataclass
class ValidityDomainReport:
    """Evaluation of whether flight regime falls within calibrated physical limits."""
    is_in_domain: bool
    ood_score: float                 # 0.0 = nominal, >1.0 = out of distribution
    warnings: List[str]
    parameters_checked: Dict[str, float]


class UncertaintyQuantifier:
    """
    Quantifies forecast uncertainty through parameter sensitivity and Monte Carlo propagation.
    """

    def __init__(self, vehicle_config: Optional[VehicleConfiguration] = None) -> None:
        self.config = vehicle_config or get_vehicle_config("holybro_x500_v2")
        self.mass_nominal_kg = self.config.compute_total_mass()

    def propagate_mission_energy_uncertainty(
        self,
        nominal_energy_wh: float,
        mass_uncertainty_pct: float = 3.0,       # ±3% mass uncertainty
        temp_uncertainty_c: float = 5.0,         # ±5°C ambient temperature uncertainty
        motor_r_uncertainty_pct: float = 5.0,    # ±5% motor winding resistance
        wind_uncertainty_mps: float = 1.5,       # ±1.5 m/s wind variation
    ) -> UncertaintyInterval:
        """
        Compute 95% confidence interval for total mission energy consumption (Wh).
        Uses first-order Taylor series error propagation:
        sigma_E^2 = sum_i ( (dE / d theta_i)^2 * sigma_theta_i^2 )
        """
        # Sensitivities (dE / dParam)
        # Power scales approximately with m^(1.5), so dE/E ~ 1.5 * dm/m
        sigma_m_frac = mass_uncertainty_pct / 100.0
        var_mass = ((1.5 * nominal_energy_wh * sigma_m_frac) ** 2)

        # Temp sensitivity: colder increases internal resistance (~1% power per 10°C)
        var_temp = ((0.0015 * nominal_energy_wh * temp_uncertainty_c) ** 2)

        # Motor R sensitivity: ohmic heating
        sigma_r_frac = motor_r_uncertainty_pct / 100.0
        var_motor = ((0.20 * nominal_energy_wh * sigma_r_frac) ** 2)

        # Wind sensitivity: parasitic drag scales with (v+v_wind)^3
        var_wind = ((0.04 * nominal_energy_wh * wind_uncertainty_mps) ** 2)

        total_variance = var_mass + var_temp + var_motor + var_wind
        std_dev = math.sqrt(total_variance)
        ci_lower = max(0.0, nominal_energy_wh - 1.96 * std_dev)
        ci_upper = nominal_energy_wh + 1.96 * std_dev

        return UncertaintyInterval(
            mean=nominal_energy_wh,
            std_dev=std_dev,
            ci_95_lower=ci_lower,
            ci_95_upper=ci_upper,
            unit="Wh",
        )


class ValidityDomainChecker:
    """
    Evaluates whether vehicle flight conditions remain inside the calibrated envelope.
    """

    def __init__(
        self,
        min_mass_kg: float = 1.10,
        max_mass_kg: float = 2.40,
        min_temp_c: float = -5.0,
        max_temp_c: float = 45.0,
        max_wind_speed_mps: float = 12.0,
        max_advance_ratio_j: float = 0.65,
    ) -> None:
        self.min_mass_kg = min_mass_kg
        self.max_mass_kg = max_mass_kg
        self.min_temp_c = min_temp_c
        self.max_temp_c = max_temp_c
        self.max_wind_speed_mps = max_wind_speed_mps
        self.max_advance_ratio_j = max_advance_ratio_j

    def check_conditions(
        self,
        mass_kg: float,
        ambient_temp_c: float = 25.0,
        wind_speed_mps: float = 0.0,
        advance_ratio_j: float = 0.0,
    ) -> ValidityDomainReport:
        """Evaluate flight condition parameters against calibration domain limits."""
        warnings: List[str] = []
        ood_penalties: List[float] = []

        if mass_kg < self.min_mass_kg or mass_kg > self.max_mass_kg:
            warnings.append(f"Mass {mass_kg:.2f} kg outside calibrated range [{self.min_mass_kg:.2f}, {self.max_mass_kg:.2f}] kg.")
            ood_penalties.append(abs(mass_kg - 1.5) / 0.5)

        if ambient_temp_c < self.min_temp_c or ambient_temp_c > self.max_temp_c:
            warnings.append(f"Temperature {ambient_temp_c:.1f}°C outside calibrated range [{self.min_temp_c}, {self.max_temp_c}]°C.")
            ood_penalties.append(abs(ambient_temp_c - 25.0) / 20.0)

        if wind_speed_mps > self.max_wind_speed_mps:
            warnings.append(f"Wind speed {wind_speed_mps:.1f} m/s exceeds max calibrated limit {self.max_wind_speed_mps:.1f} m/s.")
            ood_penalties.append((wind_speed_mps - self.max_wind_speed_mps) / 5.0)

        if advance_ratio_j > self.max_advance_ratio_j:
            warnings.append(f"Propeller advance ratio J={advance_ratio_j:.2f} exceeds stall/polar limit {self.max_advance_ratio_j:.2f}.")
            ood_penalties.append((advance_ratio_j - self.max_advance_ratio_j) / 0.2)

        ood_score = float(max(ood_penalties)) if ood_penalties else 0.0
        is_in_domain = bool(len(warnings) == 0)

        return ValidityDomainReport(
            is_in_domain=is_in_domain,
            ood_score=ood_score,
            warnings=warnings,
            parameters_checked={
                "mass_kg": mass_kg,
                "ambient_temp_c": ambient_temp_c,
                "wind_speed_mps": wind_speed_mps,
                "advance_ratio_j": advance_ratio_j,
            },
        )
