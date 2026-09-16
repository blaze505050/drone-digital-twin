"""
dronepy.calibration
===================
Online and offline digital twin system identification & calibration engine.
Wraps drone_sdk.identification.identifier.BoundedParameterIdentifier to calibrate
physical parameters (mass, drag, motor thrust scale) against flight logs while
enforcing strict safety bounds (+/- 15% physical shift).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from drone_sdk.identification.identifier import (
    BoundedParameterIdentifier,
    IdentificationDataset,
    ParameterSet,
)
from drone_sdk.dronepy.drone import Drone
from drone_sdk.dronepy.results import FlightResult


@dataclass
class CalibrationResult:
    """Outcome of a Digital Twin calibration run."""
    success: bool
    parameter_set: ParameterSet
    message: str
    condition_number: float = 0.0
    nominal_params: Dict[str, float] = field(default_factory=dict)
    calibrated_params: Dict[str, float] = field(default_factory=dict)
    shifts_pct: Dict[str, float] = field(default_factory=dict)
    validation_rmse: float = 0.0

    def summary(self) -> str:
        status_sym = "[SUCCESS]" if self.success else "[REJECTED/FAILED]"
        lines = [
            f"=== DronePy Calibration Result: {status_sym} ===",
            f"Message: {self.message}",
            f"Condition Number: {self.condition_number:.2f}",
            f"Validation RMSE: {self.validation_rmse:.4f}",
            "Parameter Shifts:",
        ]
        for k, v in self.shifts_pct.items():
            lines.append(f"  - {k}: {v:+.2f}%")
        return "\n".join(lines)


class DigitalTwinCalibrator:
    """
    Calibrates a Drone model against flight telemetry data.
    """

    def __init__(
        self,
        max_condition_number: float = 100.0,
        min_excitation_variance: float = 0.005,
        max_parameter_shift_pct: float = 15.0,
    ) -> None:
        self.identifier = BoundedParameterIdentifier(
            max_condition_number=max_condition_number,
            min_excitation_variance=min_excitation_variance,
            max_parameter_shift_pct=max_parameter_shift_pct,
        )

    def extract_dataset(
        self,
        data: Union[FlightResult, pd.DataFrame, Dict[str, Any], IdentificationDataset],
    ) -> IdentificationDataset:
        """Converts diverse flight log formats into an IdentificationDataset."""
        if isinstance(data, IdentificationDataset):
            return data

        if isinstance(data, FlightResult):
            time_sec = data.time
            n = len(time_sec)
            cmds = data.motor_commands if data.motor_commands is not None and len(data.motor_commands) == n else np.ones((n, 4)) * 0.5
            accel = data.accel_body if data.accel_body is not None and len(data.accel_body) == n else np.zeros((n, 3))
            omega = data.omega_body if data.omega_body is not None and len(data.omega_body) == n else np.zeros((n, 3))
            return IdentificationDataset(
                time_sec=time_sec,
                motor_commands=cmds,
                measured_accel_ned=accel,
                measured_omega_body=omega,
                battery_voltage_v=data.battery_voltage,
                battery_current_a=data.battery_current,
            )

        if pd is not None and isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            if pd is not None:
                df = pd.DataFrame(data)
            else:
                time_sec = np.asarray(data.get("time_sec", data.get("time", np.arange(10) * 0.01)), dtype=float)
                cmds = np.asarray(data.get("motor_commands", np.ones((len(time_sec), 4)) * 0.5), dtype=float)
                accel = np.asarray(data.get("measured_accel_ned", np.zeros((len(time_sec), 3))), dtype=float)
                omega = np.asarray(data.get("measured_omega_body", np.zeros((len(time_sec), 3))), dtype=float)
                return IdentificationDataset(time_sec=time_sec, motor_commands=cmds, measured_accel_ned=accel, measured_omega_body=omega)
        else:
            raise TypeError(f"Unsupported flight data format: {type(data)}")

        time_sec = df["time_sec"].values if "time_sec" in df else df["time"].values

        # Motor commands or RPMs normalized
        motor_cols = [c for c in df.columns if c.startswith("motor_rpm_") or c.startswith("rpm_")]
        if motor_cols:
            rpms = df[motor_cols].values
            # Normalize to 0-1 based on max observed or typical max 15000 RPM
            max_val = np.max(rpms) if np.max(rpms) > 1.0 else 1.0
            motor_commands = np.clip(rpms / max(max_val, 1e-3), 0.0, 1.0)
        elif "throttle" in df.columns:
            motor_commands = np.column_stack([df["throttle"].values] * 4)
        else:
            motor_commands = np.ones((len(time_sec), 4)) * 0.5

        # Accelerations NED
        accel_cols = ["accel_x", "accel_y", "accel_z"]
        if all(c in df.columns for c in accel_cols):
            accel = df[accel_cols].values
        elif all(c in df.columns for c in ["ax", "ay", "az"]):
            accel = df[["ax", "ay", "az"]].values
        else:
            accel = np.zeros((len(time_sec), 3))

        # Omega Body
        omega_cols = ["p_rad_s", "q_rad_s", "r_rad_s"]
        if all(c in df.columns for c in omega_cols):
            omega = df[omega_cols].values
        elif all(c in df.columns for c in ["p", "q", "r"]):
            omega = df[["p", "q", "r"]].values
        else:
            omega = np.zeros((len(time_sec), 3))

        return IdentificationDataset(
            time_sec=time_sec,
            motor_commands=motor_commands,
            measured_accel_ned=accel,
            measured_omega_body=omega,
            battery_voltage_v=df["battery_voltage_v"].values if "battery_voltage_v" in df else None,
            battery_current_a=df["battery_current_a"].values if "battery_current_a" in df else None,
        )

    def calibrate(
        self,
        drone: Drone,
        flight_data: Union[FlightResult, pd.DataFrame, Dict[str, Any], IdentificationDataset],
        inplace: bool = False,
    ) -> Tuple[Drone, CalibrationResult]:
        """
        Executes bounded parameter identification and returns the calibrated Drone and result report.
        """
        dataset = self.extract_dataset(flight_data)
        
        # Split train / validation if sufficient samples
        if dataset.sample_count >= 40:
            split_idx = int(dataset.sample_count * 0.7)
            train_set = IdentificationDataset(
                time_sec=dataset.time_sec[:split_idx],
                motor_commands=dataset.motor_commands[:split_idx],
                measured_accel_ned=dataset.measured_accel_ned[:split_idx],
                measured_omega_body=dataset.measured_omega_body[:split_idx],
            )
            val_set = IdentificationDataset(
                time_sec=dataset.time_sec[split_idx:],
                motor_commands=dataset.motor_commands[split_idx:],
                measured_accel_ned=dataset.measured_accel_ned[split_idx:],
                measured_omega_body=dataset.measured_omega_body[split_idx:],
            )
        else:
            train_set = dataset
            val_set = dataset

        success, candidate, reason = self.identifier.fit_candidate_parameters(train_set)
        
        target_drone = drone if inplace else drone.copy()

        nominal_params = {
            "mass_kg": float(target_drone.mass_kg),
            "drag_coefficient": float(target_drone.aerodynamics.drag_coefficient),
            "thrust_scale": 1.0,
        }

        if not success:
            res = CalibrationResult(
                success=False,
                parameter_set=candidate,
                message=reason,
                condition_number=0.0,
                nominal_params=nominal_params,
                calibrated_params=nominal_params,
                shifts_pct={"mass": 0.0, "drag": 0.0, "thrust": 0.0},
            )
            return target_drone, res

        # Evaluate on validation holdout
        val_accepted, active, val_msg = self.identifier.evaluate_candidate_holdout(candidate, val_set)

        if val_accepted:
            # Apply shifts to the drone
            target_drone.mass_kg *= active.mass_scale
            target_drone.aerodynamics.drag_coefficient *= active.drag_scale
            for m in target_drone.motors.motors:
                m.thrust_coefficient *= active.motor_thrust_scale

            calibrated_params = {
                "mass_kg": float(target_drone.mass_kg),
                "drag_coefficient": float(target_drone.aerodynamics.drag_coefficient),
                "thrust_scale": active.motor_thrust_scale,
            }
            shifts = {
                "mass": (active.mass_scale - 1.0) * 100.0,
                "drag": (active.drag_scale - 1.0) * 100.0,
                "thrust": (active.motor_thrust_scale - 1.0) * 100.0,
            }
            res = CalibrationResult(
                success=True,
                parameter_set=active,
                message=val_msg,
                condition_number=float(candidate.fit_metrics.get("cond_num", 1.0)),
                nominal_params=nominal_params,
                calibrated_params=calibrated_params,
                shifts_pct=shifts,
                validation_rmse=float(active.fit_metrics.get("val_rmse", 0.0)),
            )
            return target_drone, res
        else:
            res = CalibrationResult(
                success=False,
                parameter_set=active,
                message=val_msg,
                condition_number=float(candidate.fit_metrics.get("cond_num", 1.0)),
                nominal_params=nominal_params,
                calibrated_params=nominal_params,
                shifts_pct={"mass": 0.0, "drag": 0.0, "thrust": 0.0},
            )
            return target_drone, res
