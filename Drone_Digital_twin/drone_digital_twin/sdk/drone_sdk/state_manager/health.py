"""
state_manager.health
====================
HealthMonitor — data freshness and quality monitoring for a single vehicle.

The monitor tracks three things:
1. **Update rate** — how many state updates per second are arriving.
2. **Validation pass rate** — what fraction of updates are physically valid.
3. **Data age** — how stale the latest state snapshot is.

From these it computes an overall ``health_score ∈ [0.0, 1.0]`` and maps it
to a HealthStatus enum value.

Algorithm
---------
The update rate is estimated with an exponential moving average (EMA):

    rate_ema = α × (1 / Δt) + (1 - α) × rate_ema

where ``Δt`` is the inter-arrival time of successive updates and ``α`` is a
smoothing factor (default 0.1 for a ~10-sample window).

The validation pass rate is tracked with a fixed-length sliding window of
recent outcomes (True/False per update).

Overall score is a weighted sum::

    score = w_rate × rate_score
          + w_valid × valid_score
          + w_age   × age_score

where each sub-score is normalised to [0, 1].
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional

from .schema import HealthStatus, VehicleConfig

logger = logging.getLogger(__name__)


# ── Metrics snapshot ──────────────────────────────────────────────────────────

class HealthMetrics:
    """Immutable snapshot of health metrics at one point in time.

    All timing values are in milliseconds unless noted.
    """

    __slots__ = (
        "update_rate_hz",
        "expected_rate_hz",
        "data_age_ms",
        "max_age_ms",
        "validation_pass_rate",
        "total_updates",
        "total_invalid",
        "rate_score",
        "valid_score",
        "age_score",
        "overall_score",
        "status",
        "snapshot_mono",
    )

    def __init__(
        self,
        *,
        update_rate_hz: float,
        expected_rate_hz: float,
        data_age_ms: float,
        max_age_ms: float,
        validation_pass_rate: float,
        total_updates: int,
        total_invalid: int,
        rate_score: float,
        valid_score: float,
        age_score: float,
        overall_score: float,
        status: HealthStatus,
        snapshot_mono: float,
    ) -> None:
        self.update_rate_hz       = update_rate_hz
        self.expected_rate_hz     = expected_rate_hz
        self.data_age_ms          = data_age_ms
        self.max_age_ms           = max_age_ms
        self.validation_pass_rate = validation_pass_rate
        self.total_updates        = total_updates
        self.total_invalid        = total_invalid
        self.rate_score           = rate_score
        self.valid_score          = valid_score
        self.age_score            = age_score
        self.overall_score        = overall_score
        self.status               = status
        self.snapshot_mono        = snapshot_mono

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {k: getattr(self, k) for k in self.__slots__
                if k != "status"}  # exclude enum for plain dict

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"HealthMetrics(rate={self.update_rate_hz:.1f}/{self.expected_rate_hz:.1f} Hz, "
            f"age={self.data_age_ms:.0f} ms, "
            f"valid={self.validation_pass_rate:.0%}, "
            f"score={self.overall_score:.2f}, "
            f"status={self.status.name})"
        )


# ── HealthMonitor ─────────────────────────────────────────────────────────────

class HealthMonitor:
    """Tracks data quality for one vehicle's StateStore.

    Typically accessed through ``StateStore.health`` rather than directly.

    Args:
        config:       VehicleConfig carrying thresholds and expected rate.
        ema_alpha:    EMA smoothing factor for rate estimation (0 < α ≤ 1).
        valid_window: Rolling window size for validation pass rate (updates).

    Example::

        monitor = HealthMonitor(config)
        monitor.record_update(time.monotonic(), is_valid=True)
        metrics = monitor.get_metrics()
        print(metrics.overall_score)   # 0.95
    """

    # Weights for the combined health score
    WEIGHT_RATE  = 0.40
    WEIGHT_VALID = 0.40
    WEIGHT_AGE   = 0.20

    def __init__(
        self,
        config: VehicleConfig,
        ema_alpha: float = 0.10,
        valid_window: int = 50,
    ) -> None:
        self._config       = config
        self._alpha        = ema_alpha
        self._lock         = threading.RLock()

        # State
        self._rate_ema: float           = 0.0     # Estimated update rate, Hz
        self._last_update_mono: Optional[float] = None  # monotonic of last update
        self._valid_window: deque[bool] = deque(maxlen=valid_window)

        self._total_updates: int = 0
        self._total_invalid: int = 0
        self._last_status: HealthStatus = HealthStatus.NO_DATA

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_update(self, mono: float, *, is_valid: bool) -> None:
        """Record that a state update arrived.

        Args:
            mono:      Monotonic timestamp of the update (time.monotonic()).
            is_valid:  Whether the state passed PhysicsValidator.
        """
        with self._lock:
            self._total_updates += 1
            if not is_valid:
                self._total_invalid += 1

            self._valid_window.append(is_valid)

            # Update rate EMA
            if self._last_update_mono is not None:
                delta = mono - self._last_update_mono
                if delta > 0:
                    instantaneous_rate = 1.0 / delta
                    if self._rate_ema == 0.0:
                        self._rate_ema = instantaneous_rate  # cold start
                    else:
                        self._rate_ema = (
                            self._alpha * instantaneous_rate
                            + (1.0 - self._alpha) * self._rate_ema
                        )

            self._last_update_mono = mono

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_metrics(self) -> HealthMetrics:
        """Compute and return a snapshot of current health metrics.

        This is O(1) in the rate EMA and O(valid_window) for pass rate
        (bounded constant).
        """
        now = time.monotonic()
        with self._lock:
            rate_ema         = self._rate_ema
            last_update      = self._last_update_mono
            total_updates    = self._total_updates
            total_invalid    = self._total_invalid
            valid_window_snap = list(self._valid_window)

        # Data age
        if last_update is None:
            data_age_ms = float("inf")
        else:
            data_age_ms = (now - last_update) * 1_000.0

        # Sub-scores
        rate_score  = self._compute_rate_score(rate_ema)
        valid_score = self._compute_valid_score(valid_window_snap)
        age_score   = self._compute_age_score(data_age_ms)

        overall = (
            self.WEIGHT_RATE  * rate_score
            + self.WEIGHT_VALID * valid_score
            + self.WEIGHT_AGE   * age_score
        )
        overall = max(0.0, min(1.0, overall))

        # Status classification
        if last_update is None:
            status = HealthStatus.NO_DATA
        elif overall >= self._config.min_health_score:
            status = HealthStatus.NOMINAL
        elif overall >= self._config.critical_health_score:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.CRITICAL

        # Validation pass rate
        valid_pass = sum(valid_window_snap) / len(valid_window_snap) if valid_window_snap else 0.0

        return HealthMetrics(
            update_rate_hz       = rate_ema,
            expected_rate_hz     = self._config.expected_update_rate_hz,
            data_age_ms          = data_age_ms if data_age_ms != float("inf") else -1.0,
            max_age_ms           = self._config.max_state_age_ms,
            validation_pass_rate = valid_pass,
            total_updates        = total_updates,
            total_invalid        = total_invalid,
            rate_score           = rate_score,
            valid_score          = valid_score,
            age_score            = age_score,
            overall_score        = overall,
            status               = status,
            snapshot_mono        = now,
        )

    def get_score(self) -> float:
        """Return the overall health score without building a full HealthMetrics."""
        return self.get_metrics().overall_score

    def get_status(self) -> HealthStatus:
        """Return the current HealthStatus classification."""
        return self.get_metrics().status

    def is_stale(self, max_age_ms: Optional[float] = None) -> bool:
        """Return True if no update has been received within the age threshold.

        Args:
            max_age_ms: Override the config threshold (ms).  Uses
                        ``config.max_state_age_ms`` if None.
        """
        threshold = max_age_ms if max_age_ms is not None else self._config.max_state_age_ms
        with self._lock:
            if self._last_update_mono is None:
                return True
            age_ms = (time.monotonic() - self._last_update_mono) * 1_000.0
        return age_ms > threshold

    def reset(self) -> None:
        """Reset all accumulated statistics (e.g. after store.reset())."""
        with self._lock:
            self._rate_ema        = 0.0
            self._last_update_mono = None
            self._valid_window.clear()
            self._total_updates   = 0
            self._total_invalid   = 0

    # ── Internal sub-score helpers ────────────────────────────────────────────

    def _compute_rate_score(self, rate_hz: float) -> float:
        """Score for how close the actual update rate is to the expected rate.

        Returns 1.0 when rate == expected, 0.0 when rate == 0.
        Rates above expected are capped at 1.0 (a fast source is fine).
        """
        expected = self._config.expected_update_rate_hz
        if expected <= 0:
            return 1.0
        if rate_hz <= 0:
            return 0.0
        return min(1.0, rate_hz / expected)

    def _compute_valid_score(self, window: list[bool]) -> float:
        """Score for validation pass rate over the rolling window."""
        if not window:
            return 0.0
        return sum(window) / len(window)

    def _compute_age_score(self, data_age_ms: float) -> float:
        """Score for data freshness.

        Returns 1.0 when age is 0, linearly decays to 0.0 at max_state_age_ms,
        then stays 0.0 beyond.
        """
        max_age = self._config.max_state_age_ms
        if max_age <= 0:
            return 1.0
        if data_age_ms == float("inf"):
            return 0.0
        return max(0.0, 1.0 - data_age_ms / max_age)

    def __repr__(self) -> str:  # noqa: D105
        m = self.get_metrics()
        return (
            f"HealthMonitor(rate={m.update_rate_hz:.1f} Hz, "
            f"score={m.overall_score:.2f}, status={m.status.name})"
        )
