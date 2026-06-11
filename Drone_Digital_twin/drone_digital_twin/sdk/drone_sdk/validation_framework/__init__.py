"""
drone_sdk.validation_framework
==============================
Validation Framework — Module 13 of the UAV Digital Twin Platform.

Provides rigorous comparison of simulation outputs against:
  - Analytical reference solutions (for unit validation)
  - Hardware-in-the-loop flight data (for integration validation)
  - OpenFOAM CFD results (for aerodynamics validation)
  - Published benchmark datasets (EuRoC MAV, Zurich UAV)

Validation taxonomy (following ASME V&V 10-2006)
-------------------------------------------------
  UNIT         — individual model vs analytical solution
  INTEGRATION  — subsystem interaction (e.g. EKF vs GPS truth)
  SYSTEM       — end-to-end sim vs flight log
  CFD          — aerodynamics vs OpenFOAM or wind tunnel

Metrics
-------
  MAE     — Mean Absolute Error
  RMSE    — Root Mean Square Error
  MaxAE   — Maximum Absolute Error
  NRMSE   — Normalised RMSE (% of signal range)
  R²      — Coefficient of determination
  Theil-U — Forecast quality metric
  FFT     — Frequency-domain comparison for oscillatory signals

Python version: 3.9+
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class ValidationLevel(str, Enum):
    UNIT        = "unit"
    INTEGRATION = "integration"
    SYSTEM      = "system"
    CFD         = "cfd"


class ValidationStatus(str, Enum):
    PASS    = "PASS"
    FAIL    = "FAIL"
    WARNING = "WARNING"
    SKIP    = "SKIP"


# ─────────────────────────────────────────────────────────────────────────────
#  Core metric computation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationMetrics:
    """Statistical comparison metrics between predicted and reference signals."""
    mae:      float   # Mean Absolute Error
    rmse:     float   # Root Mean Square Error
    max_ae:   float   # Maximum Absolute Error
    nrmse:    float   # Normalised RMSE (0–1 fraction of signal range)
    r2:       float   # R² coefficient of determination (1 = perfect)
    bias:     float   # Systematic offset (mean of residuals)
    std_err:  float   # Standard deviation of residuals
    n:        int     # Number of samples

    @property
    def is_good(self) -> bool:
        """Heuristic: R² > 0.99 and NRMSE < 2%."""
        return self.r2 > 0.99 and self.nrmse < 0.02

    def to_dict(self) -> dict:
        return {
            "mae":    round(self.mae,    6),
            "rmse":   round(self.rmse,   6),
            "max_ae": round(self.max_ae, 6),
            "nrmse":  round(self.nrmse,  6),
            "r2":     round(self.r2,     6),
            "bias":   round(self.bias,   6),
            "std_err":round(self.std_err,6),
            "n":      self.n,
        }

    def __repr__(self) -> str:
        return (
            f"Metrics(MAE={self.mae:.4f}, RMSE={self.rmse:.4f}, "
            f"MaxAE={self.max_ae:.4f}, NRMSE={self.nrmse:.4f}, "
            f"R²={self.r2:.4f}, bias={self.bias:.4f})"
        )


class MetricCalculator:
    """Compute ValidationMetrics between two time series."""

    @staticmethod
    def compute(
        predicted:  np.ndarray,
        reference:  np.ndarray,
        epsilon:    float = 1e-10,
    ) -> ValidationMetrics:
        """Compute all metrics between predicted and reference arrays.

        Args:
            predicted:  Simulation / model output.
            reference:  Ground truth / reference.
            epsilon:    Small value to prevent division by zero.

        Returns:
            ValidationMetrics instance.
        """
        p = np.asarray(predicted, dtype=np.float64).ravel()
        r = np.asarray(reference, dtype=np.float64).ravel()

        if len(p) != len(r):
            raise ValueError(
                f"Length mismatch: predicted={len(p)}, reference={len(r)}"
            )

        residuals = p - r
        n         = len(residuals)

        mae     = float(np.mean(np.abs(residuals)))
        rmse    = float(np.sqrt(np.mean(residuals**2)))
        max_ae  = float(np.max(np.abs(residuals)))
        bias    = float(np.mean(residuals))
        std_err = float(np.std(residuals))

        # Normalised RMSE
        r_range = float(np.max(r) - np.min(r))
        nrmse   = rmse / (r_range + epsilon)

        # R²
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((r - np.mean(r))**2))
        r2     = 1.0 - ss_res / (ss_tot + epsilon)

        return ValidationMetrics(
            mae=mae, rmse=rmse, max_ae=max_ae,
            nrmse=nrmse, r2=r2, bias=bias,
            std_err=std_err, n=n,
        )

    @staticmethod
    def compute_fft(
        predicted: np.ndarray,
        reference: np.ndarray,
        dt:        float,
    ) -> dict:
        """Frequency-domain comparison using FFT power spectral density.

        Returns:
            Dict with dominant frequencies for each signal and
            PSD correlation coefficient.
        """
        p = np.asarray(predicted, dtype=float).ravel()
        r = np.asarray(reference, dtype=float).ravel()
        n = min(len(p), len(r))
        p, r = p[:n], r[:n]

        freqs = np.fft.rfftfreq(n, d=dt)
        psd_p = np.abs(np.fft.rfft(p))**2
        psd_r = np.abs(np.fft.rfft(r))**2

        # Dominant frequency
        dom_p = float(freqs[np.argmax(psd_p[1:])+1])
        dom_r = float(freqs[np.argmax(psd_r[1:])+1])

        # PSD correlation
        corr = float(np.corrcoef(psd_p, psd_r)[0, 1]) if n > 1 else 0.0

        return {
            "dominant_freq_predicted_hz": dom_p,
            "dominant_freq_reference_hz": dom_r,
            "psd_correlation":            round(corr, 4),
            "freq_error_hz":              abs(dom_p - dom_r),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Validation result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of one validation test."""
    test_name:   str
    level:       ValidationLevel
    signal_name: str
    status:      ValidationStatus
    metrics:     Optional[ValidationMetrics]
    threshold:   Optional[dict]        # e.g. {"rmse": 0.01, "r2": 0.99}
    message:     str  = ""
    duration_s:  float = 0.0
    timestamp:   float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "test_name":   self.test_name,
            "level":       self.level.value,
            "signal_name": self.signal_name,
            "status":      self.status.value,
            "metrics":     self.metrics.to_dict() if self.metrics else None,
            "threshold":   self.threshold,
            "message":     self.message,
            "duration_s":  round(self.duration_s, 3),
            "timestamp":   self.timestamp,
        }

    def __repr__(self) -> str:
        icon = {"PASS": "✓", "FAIL": "✗", "WARNING": "⚠", "SKIP": "○"}
        return (
            f"[{icon.get(self.status.value,'?')} {self.status.value}] "
            f"{self.test_name}/{self.signal_name} — {self.message}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Validation benchmark
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationThreshold:
    """Pass/fail thresholds for a signal."""
    rmse:    Optional[float] = None    # Max allowable RMSE
    mae:     Optional[float] = None
    max_ae:  Optional[float] = None
    nrmse:   Optional[float] = None    # e.g. 0.02 = 2%
    r2_min:  Optional[float] = None    # Min R² (e.g. 0.99)

    def check(self, m: ValidationMetrics) -> Tuple[ValidationStatus, str]:
        """Return (status, message) given a metrics object."""
        issues = []
        if self.rmse    is not None and m.rmse   > self.rmse:
            issues.append(f"RMSE {m.rmse:.4f} > {self.rmse}")
        if self.mae     is not None and m.mae    > self.mae:
            issues.append(f"MAE {m.mae:.4f} > {self.mae}")
        if self.max_ae  is not None and m.max_ae > self.max_ae:
            issues.append(f"MaxAE {m.max_ae:.4f} > {self.max_ae}")
        if self.nrmse   is not None and m.nrmse  > self.nrmse:
            issues.append(f"NRMSE {m.nrmse:.4f} > {self.nrmse}")
        if self.r2_min  is not None and m.r2     < self.r2_min:
            issues.append(f"R² {m.r2:.4f} < {self.r2_min}")

        if issues:
            return ValidationStatus.FAIL, "; ".join(issues)
        return ValidationStatus.PASS, "All thresholds satisfied"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class ValidationTest:
    """A single named validation test comparing one signal.

    Usage::

        test = ValidationTest(
            "EKF position accuracy",
            ValidationLevel.INTEGRATION,
            signal_name="x_position",
            threshold=ValidationThreshold(rmse=0.5, r2_min=0.99),
        )
        result = test.run(predicted=ekf_x, reference=truth_x)
    """

    def __init__(
        self,
        name:        str,
        level:       ValidationLevel,
        signal_name: str = "signal",
        threshold:   Optional[ValidationThreshold] = None,
        dt:          float = 0.01,
    ) -> None:
        self.name        = name
        self.level       = level
        self.signal_name = signal_name
        self.threshold   = threshold or ValidationThreshold()
        self.dt          = dt

    def run(
        self,
        predicted: np.ndarray,
        reference: np.ndarray,
    ) -> ValidationResult:
        """Execute the validation test.

        Args:
            predicted: Model/simulation output array.
            reference: Ground truth / reference array.

        Returns:
            ValidationResult with status and metrics.
        """
        t0 = time.time()
        try:
            metrics = MetricCalculator.compute(predicted, reference)
            status, msg = self.threshold.check(metrics)
        except Exception as exc:  # noqa: BLE001
            return ValidationResult(
                test_name   = self.name,
                level       = self.level,
                signal_name = self.signal_name,
                status      = ValidationStatus.FAIL,
                metrics     = None,
                threshold   = self.threshold.to_dict(),
                message     = f"Exception: {exc}",
                duration_s  = time.time() - t0,
            )

        return ValidationResult(
            test_name   = self.name,
            level       = self.level,
            signal_name = self.signal_name,
            status      = status,
            metrics     = metrics,
            threshold   = self.threshold.to_dict(),
            message     = msg,
            duration_s  = time.time() - t0,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Validation suite
# ─────────────────────────────────────────────────────────────────────────────

class ValidationSuite:
    """Collection of ValidationTests forming a complete test plan.

    Usage::

        suite = ValidationSuite("EKF Validation Suite")
        suite.add_test(ValidationTest("pos_x", ...))
        suite.add_test(ValidationTest("pos_y", ...))

        data = load_flight_log("log.h5")
        results = suite.run_all(data["ekf"], data["gps_truth"])

        report = suite.generate_report()
        suite.save_report("/tmp/validation_report.json")
    """

    def __init__(
        self,
        name:    str,
        version: str = "1.0.0",
    ) -> None:
        self.name     = name
        self.version  = version
        self._tests:   List[ValidationTest]   = []
        self._results: List[ValidationResult] = []

    def add_test(self, test: ValidationTest) -> "ValidationSuite":
        self._tests.append(test)
        return self

    def run_all(
        self,
        predicted_dict: Dict[str, np.ndarray],
        reference_dict: Dict[str, np.ndarray],
    ) -> List[ValidationResult]:
        """Run all tests.

        Args:
            predicted_dict: {signal_name: array} from simulation.
            reference_dict: {signal_name: array} from reference.

        Returns:
            List of ValidationResult.
        """
        self._results = []
        for test in self._tests:
            if test.signal_name not in predicted_dict:
                self._results.append(ValidationResult(
                    test_name=test.name, level=test.level,
                    signal_name=test.signal_name,
                    status=ValidationStatus.SKIP,
                    metrics=None, threshold=None,
                    message=f"Signal '{test.signal_name}' not in predicted data",
                ))
                continue
            if test.signal_name not in reference_dict:
                self._results.append(ValidationResult(
                    test_name=test.name, level=test.level,
                    signal_name=test.signal_name,
                    status=ValidationStatus.SKIP,
                    metrics=None, threshold=None,
                    message=f"Signal '{test.signal_name}' not in reference data",
                ))
                continue
            result = test.run(
                predicted_dict[test.signal_name],
                reference_dict[test.signal_name],
            )
            self._results.append(result)
        return self._results

    @property
    def results(self) -> List[ValidationResult]:
        return list(self._results)

    @property
    def n_pass(self) -> int:
        return sum(1 for r in self._results if r.status == ValidationStatus.PASS)

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self._results if r.status == ValidationStatus.FAIL)

    @property
    def n_skip(self) -> int:
        return sum(1 for r in self._results if r.status == ValidationStatus.SKIP)

    @property
    def overall_pass(self) -> bool:
        return self.n_fail == 0 and len(self._results) > 0

    def generate_report(self) -> dict:
        return {
            "suite_name":  self.name,
            "version":     self.version,
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": {
                "total":   len(self._results),
                "pass":    self.n_pass,
                "fail":    self.n_fail,
                "skip":    self.n_skip,
                "overall": "PASS" if self.overall_pass else "FAIL",
            },
            "results": [r.to_dict() for r in self._results],
        }

    def save_report(self, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.generate_report(), indent=2))
        return p

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  {self.name}  —  {self.n_pass}✓  {self.n_fail}✗  {self.n_skip}○")
        print(f"{'='*60}")
        for r in self._results:
            print(f"  {r}")
        print()

    def __repr__(self) -> str:
        return (
            f"ValidationSuite({self.name!r}, "
            f"tests={len(self._tests)}, "
            f"pass={self.n_pass}, fail={self.n_fail})"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Built-in benchmark suites
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkSuites:
    """Factory for standard validation suites used in aerospace research."""

    @staticmethod
    def ekf_position_suite(
        pos_rmse_m:  float = 0.5,
        vel_rmse_ms: float = 0.2,
        att_rmse_rad: float = 0.05,
    ) -> ValidationSuite:
        """Standard EKF state estimation validation suite."""
        suite = ValidationSuite("EKF Position & Attitude Validation")
        for sig, thr in [
            ("x",          ValidationThreshold(rmse=pos_rmse_m,   r2_min=0.99)),
            ("y",          ValidationThreshold(rmse=pos_rmse_m,   r2_min=0.99)),
            ("z",          ValidationThreshold(rmse=pos_rmse_m,   r2_min=0.99)),
            ("vx",         ValidationThreshold(rmse=vel_rmse_ms,  r2_min=0.98)),
            ("vy",         ValidationThreshold(rmse=vel_rmse_ms,  r2_min=0.98)),
            ("vz",         ValidationThreshold(rmse=vel_rmse_ms,  r2_min=0.98)),
            ("roll",       ValidationThreshold(rmse=att_rmse_rad, r2_min=0.97)),
            ("pitch",      ValidationThreshold(rmse=att_rmse_rad, r2_min=0.97)),
            ("yaw",        ValidationThreshold(rmse=att_rmse_rad, r2_min=0.97)),
        ]:
            suite.add_test(ValidationTest(
                f"EKF_{sig}", ValidationLevel.INTEGRATION,
                signal_name=sig, threshold=thr,
            ))
        return suite

    @staticmethod
    def dynamics_suite(
        pos_nrmse:   float = 0.02,
        att_nrmse:   float = 0.05,
    ) -> ValidationSuite:
        """6-DOF dynamics model validation suite."""
        suite = ValidationSuite("6-DOF Dynamics Validation")
        for sig, thr in [
            ("x",     ValidationThreshold(nrmse=pos_nrmse, r2_min=0.995)),
            ("y",     ValidationThreshold(nrmse=pos_nrmse, r2_min=0.995)),
            ("z",     ValidationThreshold(nrmse=pos_nrmse, r2_min=0.995)),
            ("roll",  ValidationThreshold(nrmse=att_nrmse, r2_min=0.99)),
            ("pitch", ValidationThreshold(nrmse=att_nrmse, r2_min=0.99)),
            ("yaw",   ValidationThreshold(nrmse=att_nrmse, r2_min=0.99)),
        ]:
            suite.add_test(ValidationTest(
                f"Dyn_{sig}", ValidationLevel.SYSTEM,
                signal_name=sig, threshold=thr,
            ))
        return suite

    @staticmethod
    def battery_suite(
        voltage_rmse: float = 0.05,
        soc_rmse:     float = 0.02,
    ) -> ValidationSuite:
        """Battery digital twin validation suite."""
        suite = ValidationSuite("Battery Digital Twin Validation")
        for sig, thr in [
            ("voltage", ValidationThreshold(rmse=voltage_rmse, r2_min=0.99)),
            ("soc",     ValidationThreshold(rmse=soc_rmse,     r2_min=0.99)),
            ("current", ValidationThreshold(rmse=0.5,          r2_min=0.95)),
        ]:
            suite.add_test(ValidationTest(
                f"Battery_{sig}", ValidationLevel.UNIT,
                signal_name=sig, threshold=thr,
            ))
        return suite

    @staticmethod
    def aero_suite(
        cl_rmse: float = 0.05,
        cd_rmse: float = 0.01,
    ) -> ValidationSuite:
        """Aerodynamics validation vs CFD reference."""
        suite = ValidationSuite("Aerodynamics CFD Validation")
        for sig, thr in [
            ("CL", ValidationThreshold(rmse=cl_rmse, r2_min=0.99)),
            ("CD", ValidationThreshold(rmse=cd_rmse, r2_min=0.99)),
            ("Cm", ValidationThreshold(rmse=0.02,    r2_min=0.95)),
        ]:
            suite.add_test(ValidationTest(
                f"Aero_{sig}", ValidationLevel.CFD,
                signal_name=sig, threshold=thr,
            ))
        return suite
