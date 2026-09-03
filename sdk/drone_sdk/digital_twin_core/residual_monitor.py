"""
drone_sdk.digital_twin_core.residual_monitor
============================================
Twin/Reality Residual Monitor & Fault Detector.

Continuously compares the parallel physics twin's forward state prediction
against the actual vehicle state (derived from high-rate telemetry or MEKF).

A growing residual between predicted and measured states is the foundational
indicator of a physical discrepancy:
  - High velocity residual + low attitude error  → unmodeled wind or drag shift
  - High attitude residual + asymmetric torque   → motor degradation / prop damage
  - Persistent steady-state altitude residual   → payload / mass increase
  - Rapid divergent residual                    → structural failure or sensor loss

Feeds directly into predictive_maintenance.HealthIndex.

Python version: 3.9+
"""
from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from drone_sdk.state_manager.schema import DroneStateVector, HealthStatus


@dataclass
class ResidualReport:
    """Diagnostic report of twin vs reality discrepancy."""
    timestamp:       float
    status:          HealthStatus
    pos_error_m:     float
    vel_error_ms:    float
    att_error_deg:   float
    health_score:    float          # Scalar [0.0, 1.0] where 1.0 = perfect match
    rmse_pos_window: float
    r2_window:       float
    probable_cause:  str

    def to_dict(self) -> dict:
        return {
            "timestamp":       round(self.timestamp, 3),
            "status":          self.status.name,
            "pos_error_m":     round(self.pos_error_m, 4),
            "vel_error_ms":    round(self.vel_error_ms, 4),
            "att_error_deg":   round(self.att_error_deg, 3),
            "health_score":    round(self.health_score, 3),
            "rmse_pos_window": round(self.rmse_pos_window, 4),
            "r2_window":       round(self.r2_window, 4),
            "probable_cause":  self.probable_cause,
        }


class TwinResidualMonitor:
    """Real-time monitor tracking state divergence between twin and vehicle."""

    def __init__(
        self,
        nominal_pos_threshold_m: float = 0.60,
        degraded_pos_threshold_m: float = 2.00,
        nominal_vel_threshold_ms: float = 0.80,
        window_size: int = 50,
    ) -> None:
        self.nominal_pos_thresh = nominal_pos_threshold_m
        self.degraded_pos_thresh = degraded_pos_threshold_m
        self.nominal_vel_thresh = nominal_vel_threshold_ms
        self.window_size = window_size

        self._pos_err_history: Deque[float] = collections.deque(maxlen=window_size)
        self._pred_history: Deque[np.ndarray] = collections.deque(maxlen=window_size)
        self._act_history: Deque[np.ndarray] = collections.deque(maxlen=window_size)

        self._latest_report: Optional[ResidualReport] = None

    def update(
        self,
        predicted: DroneStateVector,
        actual: DroneStateVector,
    ) -> ResidualReport:
        """Evaluate tracking residuals between twin and actual state."""
        pos_pred = np.array([predicted.x, predicted.y, predicted.z])
        pos_act  = np.array([actual.x, actual.y, actual.z])
        vel_pred = np.array([predicted.vx, predicted.vy, predicted.vz])
        vel_act  = np.array([actual.vx, actual.vy, actual.vz])

        pos_err = float(np.linalg.norm(pos_pred - pos_act))
        vel_err = float(np.linalg.norm(vel_pred - vel_act))

        # Attitude error: 2 * acos(|q_pred . q_act|)
        q_pred = np.array([predicted.q0, predicted.q1, predicted.q2, predicted.q3])
        q_act  = np.array([actual.q0, actual.q1, actual.q2, actual.q3])
        dot_q = abs(float(np.dot(q_pred, q_act)))
        dot_q = min(1.0, dot_q)
        att_err_rad = 2.0 * math.acos(dot_q)
        att_err_deg = math.degrees(att_err_rad)

        # Store history
        self._pos_err_history.append(pos_err)
        self._pred_history.append(pos_pred)
        self._act_history.append(pos_act)

        # Window metrics
        rmse_pos = float(np.sqrt(np.mean(np.array(self._pos_err_history) ** 2)))

        if len(self._act_history) >= 5:
            acts = np.array(self._act_history)
            preds = np.array(self._pred_history)
            ss_res = np.sum((acts - preds) ** 2)
            ss_tot = np.sum((acts - np.mean(acts, axis=0)) ** 2)
            r2 = float(1.0 - (ss_res / max(1e-6, ss_tot)))
            r2 = max(-1.0, min(1.0, r2))
        else:
            r2 = 1.0

        # Health score: exponentially decays with position and velocity error
        health_score = math.exp(-0.8 * pos_err - 0.5 * vel_err)
        health_score = max(0.0, min(1.0, health_score))

        # Health status & cause classification
        if pos_err <= self.nominal_pos_thresh and vel_err <= self.nominal_vel_thresh:
            status = HealthStatus.NOMINAL
            cause = "nominal_matching"
        elif pos_err <= self.degraded_pos_thresh:
            status = HealthStatus.DEGRADED
            if att_err_deg > 15.0:
                cause = "motor_thrust_asymmetry_or_damage"
            elif abs(pos_pred[2] - pos_act[2]) > 0.6 * pos_err:
                cause = "mass_shift_or_battery_sag"
            else:
                cause = "external_aerodynamic_disturbance"
        else:
            status = HealthStatus.CRITICAL
            cause = "severe_model_divergence_or_actuator_fault"

        report = ResidualReport(
            timestamp=time.time(),
            status=status,
            pos_error_m=pos_err,
            vel_error_ms=vel_err,
            att_error_deg=att_err_deg,
            health_score=health_score,
            rmse_pos_window=rmse_pos,
            r2_window=r2,
            probable_cause=cause,
        )
        self._latest_report = report
        return report

    @property
    def latest_report(self) -> Optional[ResidualReport]:
        return self._latest_report

    def feed_into_health_index(self, health_index: object, weight: float = 0.30) -> None:
        """Seamlessly push twin residual score into predictive_maintenance.HealthIndex."""
        if self._latest_report is not None and hasattr(health_index, "update"):
            health_index.update("twin_residual", score=self._latest_report.health_score, weight=weight)
