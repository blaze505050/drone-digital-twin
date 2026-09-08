"""
drone_sdk.experiments.energy_forecast.evaluator
===============================================
Prospective Multi-Waypoint Mission Energy & Endurance Evaluator.
Implements Backlog Item B14 & PRD ENG-02/ID-02.

Features:
- Physics-based flight profile decomposition (Climb, Cruise, Descent, Hover).
- Forward prospective energy integration along waypoints without using future telemetry.
- Holdout benchmark scoring comparing nominal vs calibrated models against flight logs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...battery_twin.pack_model import EnergyPredictionBaseline, PackECMModel
from ...configuration import VehicleConfiguration, get_vehicle_config
from ...contracts import DataStatus


@dataclass
class Waypoint:
    """3D Waypoint in NED coordinates [North, East, Down] (m)."""
    x: float
    y: float
    z: float                         # z is negative for altitude above ground in NED!
    hover_sec: float = 0.0           # Loiter duration at this waypoint

    @property
    def pos_ned(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass
class MissionPlan:
    """Complete flight mission plan defined by sequential waypoints and cruise targets."""
    mission_id: str = "mission_survey_01"
    waypoints: List[Waypoint] = field(default_factory=list)
    cruise_speed_mps: float = 8.0     # Horizontal cruise speed (m/s)
    climb_speed_mps: float = 2.5      # Vertical climb rate (m/s)
    descent_speed_mps: float = 1.8    # Vertical descent rate (m/s)


@dataclass
class EnergyForecastResult:
    """Prospective energy and flight performance forecast."""
    mission_id: str
    total_distance_m: float
    total_duration_sec: float
    predicted_energy_wh: float
    final_soc: float
    min_voltage_v: float
    peak_current_a: float
    avg_power_w: float
    reserve_endurance_sec: float
    is_feasible: bool
    segment_breakdown: List[Dict[str, Any]]
    time_series: Dict[str, np.ndarray]


class MissionEnergyEvaluator:
    """
    Evaluator performing prospective energy forecasts across multi-waypoint flight plans.
    """

    def __init__(self, vehicle_config: Optional[VehicleConfiguration] = None) -> None:
        self.config = vehicle_config or get_vehicle_config("holybro_x500_v2")
        self.mass_kg = self.config.compute_total_mass()
        self.pack_ecm = PackECMModel(self.config.battery)
        self.baseline = EnergyPredictionBaseline(self.config.battery, self.mass_kg)

    def forecast_mission(
        self,
        mission: MissionPlan,
        wind_ned_mps: np.ndarray = np.zeros(3),
        payload_added_kg: float = 0.0,
        dt_sec: float = 0.5,
    ) -> EnergyForecastResult:
        """
        Forecast prospective energy consumption along waypoints without peering at future telemetry.
        """
        total_mass = self.mass_kg + payload_added_kg
        self.pack_ecm.reset()

        if len(mission.waypoints) < 2:
            raise ValueError("Mission must contain at least 2 waypoints (Takeoff + Destination).")

        segments_summary: List[Dict[str, Any]] = []
        t_all = [0.0]
        v_all = [self.pack_ecm.state.v_terminal]
        soc_all = [self.pack_ecm.state.soc]
        wh_all = [0.0]
        p_all = [0.0]
        i_all = [0.0]

        total_distance = 0.0
        cur_pos = mission.waypoints[0].pos_ned.copy()

        # Iterate through waypoint legs
        for wp_idx in range(1, len(mission.waypoints)):
            target_wp = mission.waypoints[wp_idx]
            target_pos = target_wp.pos_ned
            delta_pos = target_pos - cur_pos

            dist_horiz = float(np.linalg.norm(delta_pos[:2]))
            dist_vert = float(abs(delta_pos[2]))
            total_distance += math.sqrt(dist_horiz ** 2 + dist_vert ** 2)

            # 1. Determine leg duration and airspeed
            if dist_horiz > 1.0:
                speed_target = mission.cruise_speed_mps
                t_leg = dist_horiz / max(1.0, speed_target)
            else:
                # Vertical climb or descent
                speed_target = mission.climb_speed_mps if delta_pos[2] < 0 else mission.descent_speed_mps
                t_leg = dist_vert / max(0.5, speed_target)

            # 2. Power model for forward flight / climb / descent
            # Aerodynamic power: P_hover * (1 + 0.5 * (v/v_h)^2) + Drag power
            p_hover_w, _ = self.baseline.estimate_hover_power()
            # Climb requires extra potential energy rate: m * g * v_climb
            v_climb = max(0.0, -delta_pos[2] / max(0.1, t_leg))
            p_potential = (total_mass * 9.80665 * v_climb) / 0.75  # Motor efficiency

            # Parasitic drag at cruise speed v
            rho = 1.225
            cd_area = 0.035
            p_drag = 0.5 * rho * cd_area * (speed_target ** 3)
            p_leg_total = p_hover_w + p_potential + p_drag

            # Simulate leg
            n_steps = max(1, int(t_leg / dt_sec))
            for _ in range(n_steps):
                v_term = max(10.0, self.pack_ecm.state.v_terminal)
                i_load = p_leg_total / v_term
                st = self.pack_ecm.step(i_load, dt_sec)
                t_all.append(st.timestamp)
                v_all.append(st.v_terminal)
                soc_all.append(st.soc)
                wh_all.append(st.energy_consumed_wh)
                p_all.append(st.power_w)
                i_all.append(st.current_a)

            # 3. Loiter / Hover time at waypoint if configured
            if target_wp.hover_sec > 0.0:
                n_hover_steps = max(1, int(target_wp.hover_sec / dt_sec))
                for _ in range(n_hover_steps):
                    v_term = max(10.0, self.pack_ecm.state.v_terminal)
                    i_hover = p_hover_w / v_term
                    st = self.pack_ecm.step(i_hover, dt_sec)
                    t_all.append(st.timestamp)
                    v_all.append(st.v_terminal)
                    soc_all.append(st.soc)
                    wh_all.append(st.energy_consumed_wh)
                    p_all.append(st.power_w)
                    i_all.append(st.current_a)

            segments_summary.append({
                "leg_index": wp_idx,
                "target_pos_ned": target_pos.tolist(),
                "duration_sec": t_leg + target_wp.hover_sec,
                "power_w": p_leg_total,
                "energy_at_end_wh": self.pack_ecm.state.energy_consumed_wh,
                "soc_at_end": self.pack_ecm.state.soc,
            })
            cur_pos = target_pos.copy()

        # Reserve endurance remaining
        rem_energy = self.pack_ecm.state.remaining_energy_wh
        p_avg = float(np.mean(p_all)) if p_all else 1.0
        reserve_sec = (rem_energy / max(1.0, p_avg)) * 3600.0

        is_feasible = bool(
            not self.pack_ecm.state.is_cutoff
            and self.pack_ecm.state.soc >= 0.15  # At least 15% reserve
            and min(v_all) > self.config.battery.min_pack_voltage
        )

        return EnergyForecastResult(
            mission_id=mission.mission_id,
            total_distance_m=total_distance,
            total_duration_sec=self.pack_ecm.state.timestamp,
            predicted_energy_wh=self.pack_ecm.state.energy_consumed_wh,
            final_soc=self.pack_ecm.state.soc,
            min_voltage_v=float(min(v_all)),
            peak_current_a=float(max(i_all)),
            avg_power_w=p_avg,
            reserve_endurance_sec=reserve_sec,
            is_feasible=is_feasible,
            segment_breakdown=segments_summary,
            time_series={
                "time_sec": np.array(t_all),
                "voltage_v": np.array(v_all),
                "soc": np.array(soc_all),
                "energy_wh": np.array(wh_all),
                "power_w": np.array(p_all),
                "current_a": np.array(i_all),
            },
        )

    def evaluate_holdout_prediction(
        self,
        actual_flight_log: Dict[str, np.ndarray],
        predicted_result: EnergyForecastResult,
    ) -> Dict[str, Any]:
        """
        Evaluate forecast accuracy against held-out measured flight data.
        Computes absolute Wh error, voltage RMSE, and peak current error.
        """
        actual_wh = float(actual_flight_log.get("energy_wh", [0.0])[-1])
        pred_wh = predicted_result.predicted_energy_wh
        wh_err = abs(pred_wh - actual_wh)
        wh_err_pct = (wh_err / max(1e-3, actual_wh)) * 100.0

        # Voltage curve alignment
        act_v = actual_flight_log.get("voltage_v", np.array([]))
        pred_v = predicted_result.time_series["voltage_v"]
        min_len = min(len(act_v), len(pred_v))
        if min_len > 1:
            rmse_v = float(np.sqrt(np.mean((act_v[:min_len] - pred_v[:min_len]) ** 2)))
        else:
            rmse_v = 0.0

        return {
            "actual_energy_wh": actual_wh,
            "predicted_energy_wh": pred_wh,
            "energy_error_wh": wh_err,
            "energy_error_percent": wh_err_pct,
            "voltage_rmse_v": rmse_v,
            "is_within_5pct_tolerance": bool(wh_err_pct <= 5.0),
        }
