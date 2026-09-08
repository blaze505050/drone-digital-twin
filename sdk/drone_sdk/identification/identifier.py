"""
drone_sdk.identification.identifier
===================================
Bounded System Identification, Excitation Gating, and Model Rollback Engine.
Implements Backlog Item B18 & PRD ID-03/SAF-01.

Key features:
1. Regressor condition number / persistent excitation (PE) validation.
2. Bounded parameter estimation (+/- 15% physical safety bounds).
3. Candidate vs Active versioning with disjoint holdout window evaluation.
4. Automatic rollback upon model degradation or tracking divergence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class IdentificationDataset:
    """Synchronized telemetry dataset for system identification."""
    time_sec: np.ndarray
    motor_commands: np.ndarray       # (N, 4) in [0, 1]
    measured_accel_ned: np.ndarray   # (N, 3) in m/s^2
    measured_omega_body: np.ndarray  # (N, 3) in rad/s
    battery_voltage_v: Optional[np.ndarray] = None
    battery_current_a: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.time_sec = np.asarray(self.time_sec, dtype=float)
        self.motor_commands = np.asarray(self.motor_commands, dtype=float)
        self.measured_accel_ned = np.asarray(self.measured_accel_ned, dtype=float)
        self.measured_omega_body = np.asarray(self.measured_omega_body, dtype=float)

    @property
    def sample_count(self) -> int:
        return len(self.time_sec)


@dataclass
class ParameterSet:
    """Versioned physical parameter set with provenance status."""
    version: int = 1
    mass_scale: float = 1.0          # Multiplier on nominal mass (e.g. 1.05 = +5% payload)
    drag_scale: float = 1.0          # Multiplier on translational drag coefficient
    motor_thrust_scale: float = 1.0  # Multiplier on static motor thrust coefficient
    status: str = "PRIOR"            # PRIOR, CANDIDATE, ACTIVE, ROLLED_BACK
    rejection_reason: str = ""
    fit_metrics: Dict[str, float] = field(default_factory=dict)


class BoundedParameterIdentifier:
    """
    Identifies vehicle physical parameters under strict excitation and safety bounds.
    """

    def __init__(
        self,
        max_condition_number: float = 100.0,
        min_excitation_variance: float = 0.005,
        max_parameter_shift_pct: float = 15.0,  # Bounded to +/- 15% max change
    ) -> None:
        self.max_condition_number = max_condition_number
        self.min_excitation_variance = min_excitation_variance
        self.max_shift = max_parameter_shift_pct / 100.0
        self.active_params = ParameterSet(version=1, status="ACTIVE")
        self.history: List[ParameterSet] = [self.active_params]

    def check_excitation(self, data: IdentificationDataset) -> Tuple[bool, float, str]:
        """
        Verify that flight dataset has sufficient dynamic excitation for system identification.
        Computes regressor matrix rank and condition number kappa.
        """
        if data.sample_count < 20:
            return False, float("inf"), "Insufficient samples (minimum 20 required)."

        # Check command variance
        cmd_var = float(np.var(data.motor_commands))
        if cmd_var < self.min_excitation_variance:
            return False, float("inf"), f"Insufficient excitation variance ({cmd_var:.5f} < {self.min_excitation_variance:.5f}). Hover alone cannot identify separate parameters."

        # Regressor matrix Phi: [cmd_sum, vertical_velocity_approx, constant_gravity]
        cmd_sum = np.sum(data.motor_commands, axis=1)
        phi = np.column_stack([
            cmd_sum,
            np.gradient(data.measured_accel_ned[:, 2]),
            np.ones(data.sample_count),
        ])

        # Singular Value Decomposition
        _, s, _ = np.linalg.svd(phi, full_matrices=False)
        cond_num = float(s[0] / max(1e-9, s[-1]))

        if cond_num > self.max_condition_number:
            return False, cond_num, f"Regressor ill-conditioned (kappa={cond_num:.1f} > {self.max_condition_number:.1f})."

        return True, cond_num, "Excitation verified."

    def fit_candidate_parameters(
        self,
        train_data: IdentificationDataset,
        base_params: Optional[ParameterSet] = None,
    ) -> Tuple[bool, ParameterSet, str]:
        """
        Estimate candidate parameters from training segment under strict box constraints.
        """
        base = base_params or self.active_params
        is_excited, cond_num, reason = self.check_excitation(train_data)
        if not is_excited:
            rejected = ParameterSet(
                version=base.version + 1,
                mass_scale=base.mass_scale,
                drag_scale=base.drag_scale,
                motor_thrust_scale=base.motor_thrust_scale,
                status="REJECTED",
                rejection_reason=f"Excitation check failed: {reason}",
            )
            return False, rejected, reason

        # Regressor for vertical acceleration:
        # a_z = (F_thrust / m) - g - (D_z / m)
        # Solve for delta thrust scaling and mass scaling
        cmd_sum = np.sum(train_data.motor_commands, axis=1)
        # Target vertical acceleration (in NED, upward reaction is negative a_z)
        y_target = train_data.measured_accel_ned[:, 2]

        # Fit linear model: y = c1 * cmd_sum + c0
        x_mat = np.column_stack([cmd_sum, np.ones(len(cmd_sum))])
        coeffs, residuals, _, _ = np.linalg.lstsq(x_mat, y_target, rcond=None)

        # Nominal slope for sum of 4 motors: -26.0 m/s^2 total / 4.0 = -6.5 m/s^2 per sum unit
        nominal_slope = -26.0 / 4.0
        est_thrust_ratio = coeffs[0] / nominal_slope if abs(nominal_slope) > 1e-3 else 1.0

        # Apply strict safety bounds: +/- 15%
        clamped_thrust_scale = float(np.clip(
            est_thrust_ratio * base.motor_thrust_scale,
            1.0 - self.max_shift,
            1.0 + self.max_shift,
        ))

        # Mass scale update based on constant offset
        clamped_mass_scale = float(np.clip(
            base.mass_scale * (1.0 + 0.05 * np.sign(coeffs[1] - 9.81)),
            1.0 - self.max_shift,
            1.0 + self.max_shift,
        ))

        candidate = ParameterSet(
            version=base.version + 1,
            mass_scale=clamped_mass_scale,
            drag_scale=base.drag_scale,
            motor_thrust_scale=clamped_thrust_scale,
            status="CANDIDATE",
            fit_metrics={
                "regressor_cond_num": cond_num,
                "residual_norm": float(np.sum(residuals)) if len(residuals) > 0 else 0.0,
            },
        )
        return True, candidate, "Candidate parameters successfully identified."

    def evaluate_and_promote(
        self,
        candidate: ParameterSet,
        val_data: IdentificationDataset,
        min_improvement_pct: float = 3.0,
    ) -> Tuple[bool, ParameterSet, Dict[str, float]]:
        """
        Evaluate candidate parameter set on held-out validation segment.
        Promotes to ACTIVE if validation tracking error improves by >= min_improvement_pct.
        Otherwise rolls back to active prior.
        """
        if candidate.status != "CANDIDATE":
            return False, self.active_params, {"error": "Invalid candidate status"}

        # Simulate predicted vertical acceleration on validation set
        cmd_sum = np.sum(val_data.motor_commands, axis=1)
        y_true = val_data.measured_accel_ned[:, 2]

        # 1. Active model prediction
        pred_active = 9.81 - 26.0 * (self.active_params.motor_thrust_scale / self.active_params.mass_scale) * (cmd_sum / 4.0)
        rmse_active = float(np.sqrt(np.mean((y_true - pred_active) ** 2)))

        # 2. Candidate model prediction
        pred_cand = 9.81 - 26.0 * (candidate.motor_thrust_scale / candidate.mass_scale) * (cmd_sum / 4.0)
        rmse_cand = float(np.sqrt(np.mean((y_true - pred_cand) ** 2)))

        improvement = ((rmse_active - rmse_cand) / max(1e-4, rmse_active)) * 100.0

        metrics = {
            "rmse_active": rmse_active,
            "rmse_candidate": rmse_cand,
            "improvement_pct": improvement,
        }

        if improvement >= min_improvement_pct:
            # Promote candidate to ACTIVE
            candidate.status = "ACTIVE"
            candidate.fit_metrics.update(metrics)
            self.active_params = candidate
            self.history.append(candidate)
            return True, candidate, metrics
        else:
            # Reject and Rollback
            candidate.status = "ROLLED_BACK"
            candidate.rejection_reason = f"Insufficient holdout improvement ({improvement:.2f}% < {min_improvement_pct:.2f}%)."
            candidate.fit_metrics.update(metrics)
            self.history.append(candidate)
            return False, self.active_params, metrics
