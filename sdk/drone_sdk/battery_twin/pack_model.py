"""
drone_sdk.battery_twin.pack_model
=================================
Multi-Cell Series-Parallel (NsNp) Battery Pack Thevenin ECM & Energy Prediction Baseline.
Implements Backlog Item B12 & PRD ENG-01/02.

Features:
- Non-linear OCV-SOC curve for Li-ion / LiPo chemistries.
- Dynamic Thevenin RC polarization branch with matrix-exponential state transition.
- Temperature and SOC dependent internal resistance.
- Cumulative energy integration (Watt-hours = Integral(V_term * I_pack * dt)).
- Prospective flight endurance & remaining usable energy prediction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..configuration.schema import BatteryPackConfig

PackConfig = BatteryPackConfig


@dataclass
class PackState:
    """State of the battery pack at a specific point in time."""
    timestamp: float = 0.0
    soc: float = 1.0                 # State of Charge [0.0, 1.0]
    v_terminal: float = 16.8         # Pack terminal voltage under load (V)
    v_ocv: float = 16.8              # Open-circuit pack voltage (V)
    v_rc1: float = 0.0               # Polarization RC voltage drop per cell (V)
    current_a: float = 0.0           # Discharging current (A)
    energy_consumed_wh: float = 0.0  # Cumulative electrical energy consumed (Wh)
    remaining_energy_wh: float = 74.0# Estimated remaining energy to 3.2V cutoff (Wh)
    temperature_c: float = 25.0      # Pack core temperature (°C)
    power_w: float = 0.0             # Instantaneous electrical power draw (W)
    is_cutoff: bool = False          # True if pack voltage dropped below safe cutoff


class PackECMModel:
    """
    Physics-based Equivalent Circuit Model (Thevenin 1-RC) for multirotor battery packs.
    """

    def __init__(self, config: Optional[BatteryPackConfig] = None) -> None:
        self.config = config or BatteryPackConfig()
        self.ns = self.config.cells_in_series
        self.np = self.config.cells_in_parallel
        self.q_cell_coulombs = self.config.cell_capacity_ah * 3600.0  # Coulombs
        self.state = PackState()
        self.reset()

    def reset(self, initial_soc: float = 1.0, initial_temp_c: float = 25.0) -> PackState:
        """Reset pack state to initial SOC and ambient temperature."""
        soc = float(np.clip(initial_soc, 0.0, 1.0))
        voc_cell = self.get_ocv_cell(soc)
        v_pack = self.ns * voc_cell
        nom_wh = self.config.nominal_energy_wh * soc

        self.state = PackState(
            timestamp=0.0,
            soc=soc,
            v_terminal=v_pack,
            v_ocv=v_pack,
            v_rc1=0.0,
            current_a=0.0,
            energy_consumed_wh=0.0,
            remaining_energy_wh=nom_wh,
            temperature_c=initial_temp_c,
            power_w=0.0,
            is_cutoff=False,
        )
        return self.state

    @staticmethod
    def get_ocv_cell(soc: float) -> float:
        """
        Empirical LiPo / Li-ion cell Open Circuit Voltage (OCV) as a function of SOC.
        Calibrated from 4.20V (100% SOC) to 3.20V (0% SOC).
        """
        s = float(np.clip(soc, 0.0, 1.0))
        # High-order polynomial fit representing the flat plateau (3.7-3.85V) and steep discharge knees
        voc = (
            3.20
            + 0.55 * s
            + 0.45 * (s ** 2)
            - 0.50 * (s ** 3)
            + 0.50 * (s ** 4)
        )
        # Additional drop at extreme low SOC (<10%)
        if s < 0.10:
            voc -= 0.25 * (1.0 - s / 0.10)
        return float(np.clip(voc, 3.0, 4.25))

    def step(
        self,
        current_load_a: float,
        dt_sec: float,
        ambient_temp_c: float = 25.0,
    ) -> PackState:
        """
        Advance battery pack state by dt_sec under current load I_load (A).
        """
        dt = max(1e-4, float(dt_sec))
        i_pack = max(0.0, float(current_load_a))
        i_cell = i_pack / self.np

        # 1. Coulomb counting SOC update
        delta_q = i_cell * dt  # Coulombs
        new_soc = self.state.soc - (delta_q / self.q_cell_coulombs)
        new_soc = float(np.clip(new_soc, 0.0, 1.0))

        # 2. Temperature-dependent series resistance R0
        temp_factor = 1.0 + 0.015 * (25.0 - self.state.temperature_c)
        temp_factor = max(0.6, temp_factor)
        r0_cell = self.config.cell_r0 * temp_factor
        r1_cell = self.config.cell_r1
        c1_cell = self.config.cell_c1

        # 3. Polarization RC branch exact continuous transition
        tau_rc = max(1e-2, r1_cell * c1_cell)
        decay = math.exp(-dt / tau_rc)
        new_v_rc1 = self.state.v_rc1 * decay + r1_cell * (1.0 - decay) * i_cell

        # 4. Terminal voltage calculation
        voc_cell = self.get_ocv_cell(new_soc)
        v_cell_terminal = voc_cell - (i_cell * r0_cell) - new_v_rc1
        v_cell_terminal = max(0.0, v_cell_terminal)
        v_pack_terminal = self.ns * v_cell_terminal
        v_pack_ocv = self.ns * voc_cell

        # Check cutoff
        is_cutoff = bool(v_pack_terminal <= self.config.min_pack_voltage)

        # 5. Energy and Power integration
        power_w = v_pack_terminal * i_pack
        delta_joules = power_w * dt
        delta_wh = delta_joules / 3600.0
        new_energy_consumed = self.state.energy_consumed_wh + delta_wh
        rem_energy_wh = max(0.0, self.config.nominal_energy_wh * new_soc)

        # 6. Thermal lumped capacitance update
        # Heat generated = I^2 * R_internal
        r_pack_total = (self.ns / self.np) * (r0_cell + r1_cell)
        q_gen_w = (i_pack ** 2) * r_pack_total
        # Heat dissipation to ambient: h * A * (T - T_amb)
        q_diss_w = 0.8 * (self.state.temperature_c - ambient_temp_c)
        # Thermal mass: m * cp (cp ~ 900 J/(kg*K))
        m_cp = self.config.pack_mass_kg * 900.0
        delta_t_c = ((q_gen_w - q_diss_w) / m_cp) * dt
        new_temp_c = self.state.temperature_c + delta_t_c

        # Update state
        self.state = PackState(
            timestamp=self.state.timestamp + dt,
            soc=new_soc,
            v_terminal=v_pack_terminal,
            v_ocv=v_pack_ocv,
            v_rc1=new_v_rc1,
            current_a=i_pack,
            energy_consumed_wh=new_energy_consumed,
            remaining_energy_wh=rem_energy_wh,
            temperature_c=new_temp_c,
            power_w=power_w,
            is_cutoff=is_cutoff,
        )
        return self.state


class EnergyPredictionBaseline:
    """
    Fixed Analytical & Empirical Baseline for Mission Energy & Endurance Forecasting.
    Evaluates power requirements and integrates electrical Wh across mission phases.
    """

    def __init__(self, pack_config: BatteryPackConfig, vehicle_mass_kg: float) -> None:
        self.pack_config = pack_config
        self.vehicle_mass_kg = vehicle_mass_kg
        self.ecm = PackECMModel(pack_config)

    def estimate_hover_power(self, air_density: float = 1.225) -> Tuple[float, float]:
        """
        Estimate hover electrical power (W) and hover current draw (A) using momentum theory.
        P_mech_hover = T^(3/2) / sqrt(2 * rho * A_disk) / FM
        """
        weight_n = self.vehicle_mass_kg * 9.80665
        # Total disk area for 4x 10" propellers
        d_prop = 0.254  # 10 inches
        area_total = 4 * (math.pi * (d_prop / 2.0) ** 2)
        figure_of_merit = 0.70  # Standard small multirotor propeller FOM
        p_induced = (weight_n ** 1.5) / (math.sqrt(2.0 * air_density * area_total) * figure_of_merit)

        # Motor + ESC electrical efficiency ~ 78%
        motor_efficiency = 0.78
        p_elec_rotors = p_induced / motor_efficiency
        p_avionics = 8.5  # Base avionics draw
        p_total_w = p_elec_rotors + p_avionics

        v_nom = self.pack_config.nominal_pack_voltage
        current_hover_a = p_total_w / v_nom
        return p_total_w, current_hover_a

    def predict_hover_endurance_sec(self, usable_soc_fraction: float = 0.80) -> float:
        """
        Calculate maximum hover flight time (seconds) down to reserve cutoff (e.g. 20% SOC).
        """
        p_hover_w, _ = self.estimate_hover_power()
        usable_energy_wh = self.pack_config.nominal_energy_wh * usable_soc_fraction
        endurance_hours = usable_energy_wh / p_hover_w
        return float(endurance_hours * 3600.0)

    def simulate_mission_profile(
        self,
        flight_segments: List[Tuple[float, float]], # List of (duration_sec, power_w)
        dt_sec: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Simulate energy drain along a profile of flight segments.
        Returns complete profile metrics: total energy (Wh), final SOC, minimum voltage.
        """
        self.ecm.reset()
        t_hist = []
        v_hist = []
        soc_hist = []
        wh_hist = []

        for duration_sec, power_w in flight_segments:
            n_steps = max(1, int(duration_sec / dt_sec))
            for _ in range(n_steps):
                # Approximate current I = P / V_term
                v_est = max(10.0, self.ecm.state.v_terminal)
                i_load = power_w / v_est
                st = self.ecm.step(i_load, dt_sec)
                t_hist.append(st.timestamp)
                v_hist.append(st.v_terminal)
                soc_hist.append(st.soc)
                wh_hist.append(st.energy_consumed_wh)
                if st.is_cutoff:
                    break

        return {
            "total_duration_sec": self.ecm.state.timestamp,
            "total_energy_consumed_wh": self.ecm.state.energy_consumed_wh,
            "final_soc": self.ecm.state.soc,
            "min_terminal_voltage_v": min(v_hist) if v_hist else 0.0,
            "is_cutoff_triggered": self.ecm.state.is_cutoff,
            "time_series": {
                "time_sec": np.array(t_hist),
                "voltage_v": np.array(v_hist),
                "soc": np.array(soc_hist),
                "energy_wh": np.array(wh_hist),
            }
        }
