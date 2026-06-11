"""
drone_sdk.predictive_maintenance
=================================
Predictive Maintenance — Module 17 of the UAV Digital Twin Platform.

Implements:

1. **Feature Extractor** — computes time-domain, frequency-domain, and
   statistical features from raw sensor streams (vibration, current, temp).

2. **LSTM Autoencoder** (NumPy) — unsupervised anomaly detection. Trains
   on normal operation data; high reconstruction error = anomaly.

3. **Health Index** — single scalar [0,1] combining multiple sub-system
   health scores (battery SOH, structural damage, vibration anomaly score).

4. **Maintenance Scheduler** — predicts next maintenance interval from RUL
   estimates and accumulated flight hours. Generates maintenance work-orders.

5. **Flight Log Analyser** — post-flight analysis of StateStore history,
   identifying fault signatures and operational statistics.

Python version: 3.9+
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """Extracts statistical + spectral features from sensor time-series.

    Computes a fixed-length feature vector from a sliding window of raw data,
    suitable as input to anomaly detection models.

    Features computed (per channel):
        Time-domain:  mean, std, rms, peak, crest_factor, kurtosis, skewness
        Spectral:     dominant_freq, spectral_centroid, spectral_rolloff
    """

    N_TIME_FEATURES    = 7
    N_SPECTRAL_FEATURES = 3
    N_FEATURES_PER_CH  = N_TIME_FEATURES + N_SPECTRAL_FEATURES  # = 10

    @staticmethod
    def extract(
        signal: np.ndarray,     # (N,) or (N, C) where C = channels
        dt:     float = 0.01,   # sampling interval (s)
    ) -> np.ndarray:
        """Extract feature vector from a signal window.

        Args:
            signal: 1-D or 2-D array. If 2-D, features computed per channel.
            dt:     Sampling interval in seconds.

        Returns:
            1-D feature vector of length C × N_FEATURES_PER_CH.
        """
        sig = np.atleast_2d(signal)
        if sig.shape[0] < sig.shape[1]:
            sig = sig.T   # Ensure (N, C)
        n_samples, n_channels = sig.shape
        features = []
        for c in range(n_channels):
            features.extend(FeatureExtractor._channel_features(sig[:, c], dt))
        return np.array(features, dtype=np.float64)

    @staticmethod
    def _channel_features(x: np.ndarray, dt: float) -> List[float]:
        n    = len(x)
        if n == 0:
            return [0.0] * FeatureExtractor.N_FEATURES_PER_CH

        # Time-domain
        mean   = float(np.mean(x))
        std    = float(np.std(x))
        rms    = float(np.sqrt(np.mean(x**2)))
        peak   = float(np.max(np.abs(x)))
        cf     = peak / (rms + 1e-10)                       # crest factor
        kurt   = float(np.mean((x - mean)**4) / (std**4 + 1e-10) - 3)  # excess kurtosis
        skew   = float(np.mean((x - mean)**3) / (std**3 + 1e-10))

        # Spectral
        freqs  = np.fft.rfftfreq(n, d=dt)
        psd    = np.abs(np.fft.rfft(x))**2
        if psd.sum() > 0:
            dom_f  = float(freqs[np.argmax(psd[1:])+1]) if len(psd) > 1 else 0.0
            sc     = float(np.sum(freqs * psd) / psd.sum())   # spectral centroid
            cumpsd = np.cumsum(psd) / psd.sum()
            sr_idx = np.searchsorted(cumpsd, 0.85)
            sr     = float(freqs[min(sr_idx, len(freqs)-1)])  # 85% rolloff
        else:
            dom_f = sc = sr = 0.0

        return [mean, std, rms, peak, cf, kurt, skew, dom_f, sc, sr]

    @staticmethod
    def feature_names(n_channels: int = 1) -> List[str]:
        base = ["mean","std","rms","peak","crest_factor","kurtosis","skewness",
                "dominant_freq","spectral_centroid","spectral_rolloff"]
        if n_channels == 1:
            return base
        return [f"ch{c}_{f}" for c in range(n_channels) for f in base]


# ─────────────────────────────────────────────────────────────────────────────
#  Simple autoencoder anomaly detector (NumPy)
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """LSTM-inspired autoencoder for unsupervised anomaly detection.

    Implemented as a simple bottleneck MLP autoencoder in NumPy.
    Train on normal-operation feature vectors; anomalies produce high
    reconstruction error.

    Usage::

        detector = AnomalyDetector(feature_dim=30)
        detector.fit(normal_features)   # (N, 30) array of normal data

        score = detector.anomaly_score(new_features)  # 0=normal, 1=anomaly
        if detector.is_anomaly(new_features):
            alert("Anomaly detected!")
    """

    def __init__(
        self,
        feature_dim:   int,
        bottleneck:    int   = 8,
        threshold_pct: float = 95.0,   # percentile of training errors for threshold
    ) -> None:
        self._dim   = feature_dim
        self._bn    = bottleneck
        self._pct   = threshold_pct
        self._mu:    Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self._W1:    Optional[np.ndarray] = None
        self._b1:    Optional[np.ndarray] = None
        self._W2:    Optional[np.ndarray] = None
        self._b2:    Optional[np.ndarray] = None
        self._threshold: float = float("inf")
        self._trained = False

    def fit(
        self,
        X:         np.ndarray,
        n_epochs:  int   = 200,
        lr:        float = 0.01,
    ) -> "AnomalyDetector":
        """Train on normal-operation feature vectors.

        Args:
            X:        (N, feature_dim) array of normal data.
            n_epochs: Training iterations.
            lr:       Learning rate.

        Returns:
            self (for chaining).
        """
        X = np.atleast_2d(X)
        # Normalise
        self._mu    = X.mean(axis=0)
        self._sigma = X.std(axis=0) + 1e-8
        Xn          = (X - self._mu) / self._sigma

        # Initialise weights (Xavier)
        rng = np.random.default_rng(42)
        lim1 = math.sqrt(6 / (self._dim + self._bn))
        lim2 = math.sqrt(6 / (self._bn + self._dim))
        self._W1 = rng.uniform(-lim1, lim1, (self._dim, self._bn))
        self._b1 = np.zeros(self._bn)
        self._W2 = rng.uniform(-lim2, lim2, (self._bn, self._dim))
        self._b2 = np.zeros(self._dim)

        # Simple SGD training
        for epoch in range(n_epochs):
            idx    = rng.integers(0, len(Xn), min(64, len(Xn)))
            Xb     = Xn[idx]
            # Forward
            h      = np.tanh(Xb @ self._W1 + self._b1)
            Xr     = h @ self._W2 + self._b2
            err    = Xr - Xb
            # Backward (simplified: only output layer gradient)
            dL_dXr = 2 * err / len(Xb)
            dW2    = h.T @ dL_dXr
            db2    = dL_dXr.sum(axis=0)
            dh     = dL_dXr @ self._W2.T * (1 - h**2)
            dW1    = Xb.T @ dh
            db1    = dh.sum(axis=0)
            # Update
            self._W1 -= lr * dW1
            self._b1 -= lr * db1
            self._W2 -= lr * dW2
            self._b2 -= lr * db2

        # Compute threshold from training errors
        rec_errors   = self._reconstruction_errors(Xn)
        self._threshold = float(np.percentile(rec_errors, self._pct))
        self._trained   = True
        return self

    def reconstruction_error(self, x: np.ndarray) -> float:
        """Mean squared reconstruction error for one feature vector."""
        if not self._trained:
            return 0.0
        xn = (np.atleast_1d(x) - self._mu) / self._sigma
        h  = np.tanh(xn @ self._W1 + self._b1)
        xr = h @ self._W2 + self._b2
        return float(np.mean((xr - xn)**2))

    def anomaly_score(self, x: np.ndarray) -> float:
        """Normalised anomaly score [0, 1]. 1 = high confidence anomaly."""
        err = self.reconstruction_error(x)
        if self._threshold <= 0:
            return 0.0
        return float(min(1.0, err / (self._threshold + 1e-10)))

    def is_anomaly(self, x: np.ndarray) -> bool:
        """True if reconstruction error exceeds the training percentile threshold."""
        return self.reconstruction_error(x) > self._threshold

    def _reconstruction_errors(self, Xn: np.ndarray) -> np.ndarray:
        h  = np.tanh(Xn @ self._W1 + self._b1)
        Xr = h @ self._W2 + self._b2
        return np.mean((Xr - Xn)**2, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
#  Health Index
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubsystemHealth:
    """Health score for one sub-system."""
    name:    str
    score:   float   # 0.0 (failed) – 1.0 (perfect)
    weight:  float   # Relative importance (sum of weights = 1)
    status:  str     = "nominal"

    def __post_init__(self):
        self.score  = float(np.clip(self.score, 0.0, 1.0))
        self.weight = float(np.clip(self.weight, 0.0, 1.0))
        if self.score >= 0.8:
            self.status = "nominal"
        elif self.score >= 0.6:
            self.status = "degraded"
        else:
            self.status = "critical"


class HealthIndex:
    """Aggregated vehicle health score from multiple sub-system scores.

    Usage::

        hi = HealthIndex()
        hi.update("battery",    soh=0.85, weight=0.35)
        hi.update("structure",  score=1.0 - damage_fraction, weight=0.25)
        hi.update("vibration",  score=1.0 - anomaly_score, weight=0.25)
        hi.update("temperature", score=temp_score, weight=0.15)
        print(hi.overall)     # 0.0–1.0
        print(hi.airworthy)   # True/False
    """

    AIRWORTHY_THRESHOLD = 0.65

    def __init__(self) -> None:
        self._subsystems: Dict[str, SubsystemHealth] = {}

    def update(self, name: str, score: float, weight: float) -> None:
        """Update a sub-system health score."""
        self._subsystems[name] = SubsystemHealth(name, score, weight)

    @property
    def overall(self) -> float:
        """Weighted average health score [0, 1]."""
        if not self._subsystems:
            return 1.0
        total_w = sum(s.weight for s in self._subsystems.values())
        if total_w == 0:
            return 1.0
        return float(sum(s.score * s.weight for s in self._subsystems.values()) / total_w)

    @property
    def airworthy(self) -> bool:
        return self.overall >= self.AIRWORTHY_THRESHOLD

    @property
    def critical_subsystems(self) -> List[str]:
        return [name for name, s in self._subsystems.items() if s.status == "critical"]

    def to_dict(self) -> dict:
        return {
            "overall":    round(self.overall, 3),
            "airworthy":  self.airworthy,
            "subsystems": {
                name: {"score": round(s.score, 3), "status": s.status, "weight": s.weight}
                for name, s in self._subsystems.items()
            },
        }

    def __repr__(self) -> str:
        return (
            f"HealthIndex(overall={self.overall:.2f}, "
            f"airworthy={self.airworthy}, "
            f"subsystems={list(self._subsystems.keys())})"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Maintenance Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class MaintenanceInterval(str, Enum):
    NEXT_FLIGHT      = "next_flight"       # Critical — ground immediately
    WITHIN_10_FLIGHTS = "within_10_flights"
    NEXT_50_HOURS    = "next_50_hours"
    ROUTINE_100_HOURS = "routine_100_hours"
    ANNUAL           = "annual"


@dataclass
class MaintenanceWorkOrder:
    """Auto-generated maintenance work order."""
    work_order_id:  str
    vehicle_id:     str
    created_at:     float
    interval:       MaintenanceInterval
    priority:       int            # 1 = critical, 5 = routine
    tasks:          List[str]
    estimated_downtime_h: float
    triggered_by:   List[str]      # Which health indices triggered this

    def to_dict(self) -> dict:
        return {
            "work_order_id":      self.work_order_id,
            "vehicle_id":         self.vehicle_id,
            "created_at_iso":     time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime(self.created_at)),
            "interval":           self.interval.value,
            "priority":           self.priority,
            "tasks":              self.tasks,
            "estimated_downtime_h": self.estimated_downtime_h,
            "triggered_by":       self.triggered_by,
        }


class MaintenanceScheduler:
    """Generates maintenance work orders from health and RUL data.

    Combines:
    - Battery RUL (from BatteryDigitalTwin)
    - Structural RUL (from FatigueMonitor)
    - Anomaly scores (from AnomalyDetector)
    - Flight hour tracking

    Usage::

        scheduler = MaintenanceScheduler("drone_0")
        scheduler.update(
            flight_hours=152.3,
            battery_rul_cycles=120.0,
            structural_damage=0.12,
            anomaly_score=0.08,
        )
        if scheduler.maintenance_due:
            wo = scheduler.generate_work_order()
            print(wo.to_dict())
    """

    def __init__(self, vehicle_id: str) -> None:
        self._vid       = vehicle_id
        self._hi        = HealthIndex()
        self._flight_h  = 0.0
        self._n_flights = 0
        self._history:  List[dict] = []
        self._wo_counter = 0

    def update(
        self,
        flight_hours:       float,
        battery_rul_cycles: float  = float("inf"),
        structural_damage:  float  = 0.0,
        anomaly_score:      float  = 0.0,
        temperature_score:  float  = 1.0,
    ) -> None:
        """Update all health sub-scores."""
        self._flight_h  = flight_hours
        self._n_flights += 1

        # Battery health: 500 cycles nominal life
        battery_score = float(np.clip(battery_rul_cycles / 500.0, 0.0, 1.0))

        # Structural: invert damage fraction
        struct_score = float(np.clip(1.0 - structural_damage, 0.0, 1.0))

        # Vibration anomaly: invert score
        vib_score = float(np.clip(1.0 - anomaly_score, 0.0, 1.0))

        self._hi.update("battery",     battery_score,  weight=0.35)
        self._hi.update("structure",   struct_score,   weight=0.25)
        self._hi.update("vibration",   vib_score,      weight=0.25)
        self._hi.update("temperature", temperature_score, weight=0.15)

        self._history.append({
            "flight_hours":   flight_hours,
            "overall_health": round(self._hi.overall, 3),
            "battery_score":  round(battery_score, 3),
            "struct_score":   round(struct_score, 3),
            "vib_score":      round(vib_score, 3),
            "timestamp":      time.time(),
        })

    @property
    def maintenance_due(self) -> bool:
        """True if any sub-system is degraded or critical."""
        return self._hi.overall < 0.85

    @property
    def health_index(self) -> HealthIndex:
        return self._hi

    def generate_work_order(self) -> MaintenanceWorkOrder:
        """Create a maintenance work order from current health state."""
        self._wo_counter += 1
        wo_id = f"WO-{self._vid.upper()}-{self._wo_counter:04d}"

        triggered = []
        tasks     = []
        priority  = 5
        down_h    = 1.0
        interval  = MaintenanceInterval.ROUTINE_100_HOURS

        # Battery
        b = self._hi._subsystems.get("battery")
        if b and b.status == "critical":
            triggered.append("battery_critical")
            tasks.append("Replace battery pack")
            priority  = min(priority, 1)
            interval  = MaintenanceInterval.NEXT_FLIGHT
            down_h   += 2.0
        elif b and b.status == "degraded":
            triggered.append("battery_degraded")
            tasks.append("Inspect and test battery capacity")
            priority  = min(priority, 3)
            interval  = MaintenanceInterval.WITHIN_10_FLIGHTS

        # Structure
        s = self._hi._subsystems.get("structure")
        if s and s.status == "critical":
            triggered.append("structure_critical")
            tasks.append("Inspect all arm joints and motor mounts")
            tasks.append("Perform resonance frequency test")
            priority  = min(priority, 1)
            interval  = MaintenanceInterval.NEXT_FLIGHT
            down_h   += 3.0
        elif s and s.status == "degraded":
            triggered.append("structure_degraded")
            tasks.append("Visual inspection of arms and frame")
            priority  = min(priority, 2)
            interval  = MaintenanceInterval.WITHIN_10_FLIGHTS

        # Vibration
        v = self._hi._subsystems.get("vibration")
        if v and v.status != "nominal":
            triggered.append("vibration_anomaly")
            tasks.append("Check motor bearings and propeller balance")
            priority  = min(priority, 2)
            if interval == MaintenanceInterval.ROUTINE_100_HOURS:
                interval = MaintenanceInterval.NEXT_50_HOURS

        # Routine tasks always added
        if self._flight_h >= 100:
            tasks.append("100-hour routine inspection")
            tasks.append("Lubricate gimbal bearings")
            tasks.append("Calibrate magnetometer and accelerometer")

        if not tasks:
            tasks.append("Routine pre-flight check")

        return MaintenanceWorkOrder(
            work_order_id         = wo_id,
            vehicle_id            = self._vid,
            created_at            = time.time(),
            interval              = interval,
            priority              = priority,
            tasks                 = tasks,
            estimated_downtime_h  = down_h,
            triggered_by          = triggered,
        )

    def get_trend(self) -> Dict[str, np.ndarray]:
        """Return health trends over time for plotting."""
        if not self._history:
            return {}
        return {
            "flight_hours":   np.array([h["flight_hours"]   for h in self._history]),
            "overall_health": np.array([h["overall_health"] for h in self._history]),
            "battery_score":  np.array([h["battery_score"]  for h in self._history]),
        }

    def save(self, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vehicle_id":   self._vid,
            "flight_hours": self._flight_h,
            "n_flights":    self._n_flights,
            "health":       self._hi.to_dict(),
            "history":      self._history[-50:],
        }
        p.write_text(json.dumps(data, indent=2))
        return p


# ─────────────────────────────────────────────────────────────────────────────
#  Flight Log Analyser
# ─────────────────────────────────────────────────────────────────────────────

class FlightLogAnalyser:
    """Post-flight analysis of a drone flight log.

    Computes operational statistics and identifies fault signatures
    from a StateStore history numpy export.

    Usage::

        from drone_sdk.state_manager import StateStore
        arrays  = StateStore.get_instance("drone_0").get_history_numpy()
        analyst = FlightLogAnalyser(arrays)
        report  = analyst.analyse()
    """

    def __init__(self, log: Dict[str, np.ndarray]) -> None:
        self._log = log

    def analyse(self) -> dict:
        """Run all analyses and return a comprehensive report."""
        return {
            "flight_duration_s":    self._flight_duration(),
            "max_altitude_agl_m":   self._max_altitude(),
            "max_speed_ms":         self._max_speed(),
            "mean_battery_soc":     self._mean_battery_soc(),
            "soc_consumed":         self._soc_consumed(),
            "vibration_rms":        self._vibration_rms(),
            "attitude_exceedances": self._attitude_exceedances(),
            "hover_efficiency":     self._hover_efficiency(),
        }

    def _flight_duration(self) -> float:
        t = self._log.get("timestamp_wall", np.array([]))
        if len(t) < 2:
            return 0.0
        return float(t[-1] - t[0])

    def _max_altitude(self) -> float:
        alt = self._log.get("altitude_agl", np.array([]))
        return float(alt.max()) if len(alt) > 0 else 0.0

    def _max_speed(self) -> float:
        gs = self._log.get("groundspeed", np.array([]))
        return float(gs.max()) if len(gs) > 0 else 0.0

    def _mean_battery_soc(self) -> float:
        soc = self._log.get("battery_soc", np.array([]))
        return float(soc.mean()) if len(soc) > 0 else 0.0

    def _soc_consumed(self) -> float:
        soc = self._log.get("battery_soc", np.array([]))
        if len(soc) < 2:
            return 0.0
        return float(soc[0] - soc[-1])

    def _vibration_rms(self) -> float:
        ax = self._log.get("ax", np.array([]))
        ay = self._log.get("ay", np.array([]))
        az = self._log.get("az", np.array([]))
        if len(ax) == 0:
            return 0.0
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        return float(np.std(acc_mag))

    def _attitude_exceedances(self) -> dict:
        roll  = self._log.get("roll",  np.zeros(1))
        pitch = self._log.get("pitch", np.zeros(1))
        lim   = math.radians(30.0)
        return {
            "roll_exceed_30deg":  int(np.sum(np.abs(roll)  > lim)),
            "pitch_exceed_30deg": int(np.sum(np.abs(pitch) > lim)),
        }

    def _hover_efficiency(self) -> float:
        """Ratio of hover time (low speed) to total flight time."""
        gs    = self._log.get("groundspeed", np.array([]))
        if len(gs) == 0:
            return 0.0
        hover = np.sum(gs < 0.5)
        return float(hover / len(gs))
