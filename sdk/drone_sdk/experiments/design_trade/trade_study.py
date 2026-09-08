"""
drone_sdk.experiments.design_trade.trade_study
==============================================
Measured Design Trade Study & Propulsion-Payload Optimization.
Implements Backlog Item B17 & PRD ENG-03/REP-02.

Evaluates:
1. Propeller Selection (APC 9x4.5 vs APC 10x4.5 vs APC 11x4.7).
2. Payload Scaling (0g to 1000g added payload).
3. Thrust-to-Weight (T/W) ratio, hover endurance, electrical efficiency (g/W), and motor thermal margin.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...battery_twin.pack_model import EnergyPredictionBaseline
from ...configuration import (
    PropellerConfig,
    VehicleConfiguration,
    get_vehicle_config,
)
from ...physics.actuators import PropellerAerodynamics


@dataclass
class TradePointResult:
    """Evaluation metrics for a single design candidate."""
    propeller_name: str
    propeller_diameter_inch: float
    payload_mass_kg: float
    total_mass_kg: float
    thrust_to_weight_ratio: float
    hover_throttle_pct: float
    hover_power_w: float
    hover_current_a: float
    hover_endurance_min: float
    hover_efficiency_g_per_w: float
    motor_temp_margin_c: float
    is_safe: bool


class DesignTradeStudy:
    """
    Executes design trade studies across propeller aerodynamics and payload capacities.
    """

    def __init__(self, base_config: Optional[VehicleConfiguration] = None) -> None:
        self.base_config = base_config or get_vehicle_config("holybro_x500_v2")

    def evaluate_propeller_and_payload_trades(
        self,
        propeller_candidates: Optional[List[PropellerConfig]] = None,
        payload_increments_kg: Optional[List[float]] = None,
    ) -> List[TradePointResult]:
        """
        Sweep propeller geometries and payload masses to compute flight performance matrix.
        """
        if propeller_candidates is None:
            propeller_candidates = [
                PropellerConfig(
                    name="APC 9x4.5 MR",
                    diameter_m=0.2286,  # 9"
                    pitch_m=0.1143,
                    polar_dataset="uiuc_apc_9x4.5",
                    ct0=0.108,
                    cp0=0.044,
                ),
                PropellerConfig(
                    name="APC 10x4.5 MR",
                    diameter_m=0.2540,  # 10"
                    pitch_m=0.1143,
                    polar_dataset="uiuc_apc_10x4.5",
                    ct0=0.112,
                    cp0=0.048,
                ),
                PropellerConfig(
                    name="APC 11x4.7 MR",
                    diameter_m=0.2794,  # 11"
                    pitch_m=0.1194,
                    polar_dataset="uiuc_apc_11x4.7",
                    ct0=0.118,
                    cp0=0.052,
                ),
            ]

        if payload_increments_kg is None:
            payload_increments_kg = [0.0, 0.250, 0.500, 0.750, 1.000]

        results: List[TradePointResult] = []
        base_mass = self.base_config.compute_total_mass()
        v_nom = self.base_config.battery.nominal_pack_voltage
        usable_wh = self.base_config.battery.nominal_energy_wh * 0.80  # 80% usable

        for prop in propeller_candidates:
            aero = PropellerAerodynamics(prop)
            d = prop.diameter_m
            d_inch = d / 0.0254

            # Maximum static thrust from 4 rotors at max pack voltage (16.8V for 4S)
            kv = self.base_config.motors[0].kv
            rpm_max = kv * self.base_config.battery.max_pack_voltage
            t_single_max, _, _ = aero.compute_aerodynamics(rpm_max)
            t_total_max_n = 4.0 * t_single_max

            for payload_kg in payload_increments_kg:
                total_mass = base_mass + payload_kg
                weight_n = total_mass * 9.80665

                # Thrust to weight ratio
                tw_ratio = t_total_max_n / max(1e-3, weight_n)

                # Required thrust per rotor in hover
                t_req_single_n = weight_n / 4.0

                # Solve required RPM in hover: T = ct0 * rho * n^2 * D^4
                ct0, cp0, _ = aero.get_coefficients(0.0)
                rho = 1.225
                n_sq = t_req_single_n / max(1e-9, ct0 * rho * (d ** 4))
                n_hover = math.sqrt(max(0.0, n_sq))
                rpm_hover = n_hover * 60.0

                # Hover power per rotor: P_mech = cp0 * rho * n^3 * D^5
                p_mech_single = cp0 * rho * (n_hover ** 3) * (d ** 5)
                # Electrical conversion efficiency ~ 78%
                p_elec_single = (p_mech_single / 0.78) + (self.base_config.motors[0].i_0 * (v_nom / 4.0))
                p_total_hover_w = (4.0 * p_elec_single) + self.base_config.avionics.idle_power_w

                # Current draw
                i_hover = p_total_hover_w / v_nom

                # Hover throttle approx: rpm_hover / rpm_max
                throttle_pct = (rpm_hover / max(1.0, rpm_max)) * 100.0

                # Hover endurance
                endurance_hours = usable_wh / max(1.0, p_total_hover_w)
                endurance_min = endurance_hours * 60.0

                # Grams per watt efficiency: (Total mass in grams) / (Total hover watts)
                eff_g_w = (total_mass * 1000.0) / max(1.0, p_total_hover_w)

                # Thermal rise estimate
                i_motor_single = (i_hover - 0.5) / 4.0
                i2r_heating_w = (i_motor_single ** 2) * self.base_config.motors[0].r_m
                temp_rise_c = i2r_heating_w * 4.5  # Thermal resistance ~4.5 °C/W
                thermal_margin_c = 100.0 - (25.0 + temp_rise_c)

                # Safety criteria: T/W >= 1.6, throttle <= 75%, thermal margin > 20°C
                is_safe = bool(tw_ratio >= 1.6 and throttle_pct <= 75.0 and thermal_margin_c > 20.0)

                results.append(TradePointResult(
                    propeller_name=prop.name,
                    propeller_diameter_inch=d_inch,
                    payload_mass_kg=payload_kg,
                    total_mass_kg=total_mass,
                    thrust_to_weight_ratio=tw_ratio,
                    hover_throttle_pct=throttle_pct,
                    hover_power_w=p_total_hover_w,
                    hover_current_a=i_hover,
                    hover_endurance_min=endurance_min,
                    hover_efficiency_g_per_w=eff_g_w,
                    motor_temp_margin_c=thermal_margin_c,
                    is_safe=is_safe,
                ))

        return results

    def format_trade_table(self, results: List[TradePointResult]) -> str:
        """Format trade results into clean markdown table."""
        lines = [
            "| Propeller | Payload (kg) | AUW (kg) | T/W Ratio | Hover Throt (%) | Hover Power (W) | Endurance (min) | Eff (g/W) | Safe? |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in results:
            safe_str = "YES" if r.is_safe else "NO"
            lines.append(
                f"| {r.propeller_name} | {r.payload_mass_kg:.2f} | {r.total_mass_kg:.2f} | "
                f"{r.thrust_to_weight_ratio:.2f} | {r.hover_throttle_pct:.1f}% | {r.hover_power_w:.1f} | "
                f"{r.hover_endurance_min:.1f} | {r.hover_efficiency_g_per_w:.2f} | {safe_str} |"
            )
        return "\n".join(lines)
