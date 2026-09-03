"""
telemetry_engine.metrics
========================
TelemetryMetrics — real-time performance tracking for the TelemetryEngine.

Tracks:
* Per-source packet rate, drop rate, error count
* Engine-wide aggregate throughput and latency
* Loop timing jitter (for diagnosing slow callbacks)

Design: metrics are updated in the engine background thread and read by
any thread (lock-free read via atomic float/int assignments on CPython).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EngineMetrics:
    """Immutable snapshot of overall TelemetryEngine performance."""
    tick_rate_hz:          float = 0.0
    tick_jitter_ms:        float = 0.0   # std-dev of tick interval, ms
    total_updates_sec:     float = 0.0   # Updates per second across all sources
    total_updates:         int   = 0
    total_ticks:           int   = 0
    uptime_s:              float = 0.0
    active_sources:        int   = 0
    source_metrics:        dict  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tick_rate_hz":      round(self.tick_rate_hz, 2),
            "tick_jitter_ms":    round(self.tick_jitter_ms, 3),
            "total_updates_sec": round(self.total_updates_sec, 1),
            "total_updates":     self.total_updates,
            "total_ticks":       self.total_ticks,
            "uptime_s":          round(self.uptime_s, 1),
            "active_sources":    self.active_sources,
            "sources":           self.source_metrics,
        }


class MetricsCollector:
    """Accumulates and summarises TelemetryEngine performance statistics.

    Thread-safe for concurrent reads (monitoring dashboard) and writes
    (engine background thread).
    """

    TICK_WINDOW = 200   # Rolling window for tick interval stats

    def __init__(self, target_rate_hz: float) -> None:
        self._target_hz  = target_rate_hz
        self._lock       = threading.Lock()
        self._start_mono: Optional[float] = None
        self._last_tick_mono: Optional[float] = None

        self._total_ticks:   int = 0
        self._total_updates: int = 0

        # Rolling tick intervals for jitter calculation (seconds)
        self._tick_intervals: deque[float] = deque(maxlen=self.TICK_WINDOW)

        # Per-source counters: source_id → {"packets": int, "dropped": int}
        self._source_counters: Dict[str, Dict[str, int]] = {}

        # EMA for aggregate update rate
        self._update_rate_ema: float = 0.0
        self._alpha: float = 0.05

    def record_engine_start(self) -> None:
        with self._lock:
            self._start_mono = time.monotonic()

    def record_tick(self, n_updates: int) -> None:
        """Call once per engine loop iteration."""
        now = time.monotonic()
        with self._lock:
            if self._last_tick_mono is not None:
                interval = now - self._last_tick_mono
                self._tick_intervals.append(interval)
                # Update rate EMA
                if interval > 0:
                    inst_up_rate = n_updates / interval
                    self._update_rate_ema = (
                        self._alpha * inst_up_rate
                        + (1 - self._alpha) * self._update_rate_ema
                    )
            self._last_tick_mono  = now
            self._total_ticks    += 1
            self._total_updates  += n_updates

    def record_source_poll(self, source_id: str, n_packets: int, n_dropped: int) -> None:
        with self._lock:
            counter = self._source_counters.setdefault(
                source_id, {"packets": 0, "dropped": 0, "errors": 0}
            )
            counter["packets"] += n_packets
            counter["dropped"] += n_dropped

    def record_source_error(self, source_id: str) -> None:
        with self._lock:
            counter = self._source_counters.setdefault(
                source_id, {"packets": 0, "dropped": 0, "errors": 0}
            )
            counter["errors"] += 1

    def get_snapshot(self, n_active_sources: int) -> EngineMetrics:
        """Return an immutable EngineMetrics snapshot."""
        now = time.monotonic()
        with self._lock:
            intervals = list(self._tick_intervals)
            total_ticks   = self._total_ticks
            total_updates = self._total_updates
            update_rate   = self._update_rate_ema
            start_mono    = self._start_mono
            source_counts = {k: dict(v) for k, v in self._source_counters.items()}

        # Compute tick rate and jitter
        if len(intervals) >= 2:
            mean_interval = sum(intervals) / len(intervals)
            tick_rate_hz  = 1.0 / mean_interval if mean_interval > 0 else 0.0
            variance      = sum((x - mean_interval)**2 for x in intervals) / len(intervals)
            jitter_ms     = (variance ** 0.5) * 1000.0
        else:
            tick_rate_hz = 0.0
            jitter_ms    = 0.0

        uptime = (now - start_mono) if start_mono else 0.0

        return EngineMetrics(
            tick_rate_hz      = tick_rate_hz,
            tick_jitter_ms    = jitter_ms,
            total_updates_sec = update_rate,
            total_updates     = total_updates,
            total_ticks       = total_ticks,
            uptime_s          = uptime,
            active_sources    = n_active_sources,
            source_metrics    = source_counts,
        )

    def reset(self) -> None:
        with self._lock:
            self._total_ticks   = 0
            self._total_updates = 0
            self._tick_intervals.clear()
            self._source_counters.clear()
            self._update_rate_ema = 0.0
            self._start_mono = time.monotonic()
