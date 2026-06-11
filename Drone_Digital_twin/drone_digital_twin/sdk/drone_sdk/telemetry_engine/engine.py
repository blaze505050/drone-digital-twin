"""
telemetry_engine.engine
=======================
TelemetryEngine — the real-time telemetry ingestion orchestrator.

Responsibilities
----------------
1. Manage a registry of TelemetrySource plug-ins.
2. Run a background thread that polls all connected sources at a
   configurable tick rate (default 100 Hz).
3. Pass collected DroneStateUpdates through the SourceArbiter.
4. Push arbitrated updates to the appropriate StateStore.
5. Optionally record every update to an HDF5 file.
6. Emit rich performance metrics continuously.

Architecture
------------

    ┌──────────────────────────────────────────────────────┐
    │                  TelemetryEngine                     │
    │                                                      │
    │  Source 1 ──poll()──┐                                │
    │  Source 2 ──poll()──┤──► SourceArbiter ──► StateStore│
    │  Source N ──poll()──┘                    │            │
    │                                          └──► HDF5    │
    │                                               Recorder│
    └──────────────────────────────────────────────────────┘

Thread model
------------
* **Engine thread** (daemon): polls sources, arbitrates, pushes to StateStore.
* **Caller thread**: calls start(), stop(), add_source(), remove_source().
* **Callback thread**: same as Engine thread (event callbacks fire inline).

Timing
------
The engine uses ``threading.Event.wait(timeout)`` for its sleep to allow
clean shutdown without sleeping through the full interval.  Each tick measures
actual elapsed time and adjusts for drift so the target rate is maintained
even when poll callbacks are slow.

Python version: 3.9+
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, List, Optional

from drone_sdk.state_manager import StateStore
from drone_sdk.state_manager.exceptions import StoreNotFoundError

from .arbiter import SourceArbiter
from .exceptions import (
    DuplicateSourceError,
    EngineAlreadyRunningError,
    SourceNotFoundError,
)
from .metrics import EngineMetrics, MetricsCollector
from .source import SourceStatus, TelemetrySource

logger = logging.getLogger(__name__)


# ── Engine configuration ──────────────────────────────────────────────────────

class EngineConfig:
    """Configuration for TelemetryEngine.

    Args:
        tick_rate_hz:         How often to poll all sources per second.
        arbiter_failover_ms:  Source staleness threshold for failover.
        max_updates_per_tick: Hard cap on updates processed in one tick.
        auto_create_stores:   If True, create StateStores for unknown vehicles.
    """

    def __init__(
        self,
        tick_rate_hz:         float = 100.0,
        arbiter_failover_ms:  float = 500.0,
        max_updates_per_tick: int   = 1_000,
        auto_create_stores:   bool  = True,
    ) -> None:
        self.tick_rate_hz         = tick_rate_hz
        self.arbiter_failover_ms  = arbiter_failover_ms
        self.max_updates_per_tick = max_updates_per_tick
        self.auto_create_stores   = auto_create_stores

    @property
    def tick_interval_s(self) -> float:
        return 1.0 / self.tick_rate_hz


# ── TelemetryEngine ───────────────────────────────────────────────────────────

class TelemetryEngine:
    """Real-time telemetry ingestion engine.

    Manages a set of TelemetrySource objects, polls them at a configured
    rate, arbitrates multi-source conflicts, and pushes updates to
    the appropriate StateStore instance.

    Quick-start::

        engine = TelemetryEngine()
        engine.add_source(my_source)
        engine.start()

        # … application runs …

        engine.stop()

    Advanced::

        config = EngineConfig(tick_rate_hz=200.0)
        engine = TelemetryEngine(config=config)
        engine.add_source(px4_source,   auto_connect=True)
        engine.add_source(imu_source,   auto_connect=True)
        engine.start()

        # Register a tick hook (called every loop iteration)
        engine.add_tick_hook(my_dashboard_updater)

        metrics = engine.get_metrics()
        engine.stop()
    """

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self._config   = config or EngineConfig()
        self._lock     = threading.RLock()
        self._stop_evt = threading.Event()

        # Source registry: source_id → TelemetrySource
        self._sources: Dict[str, TelemetrySource] = {}

        # One arbiter per vehicle_id
        self._arbiters: Dict[str, SourceArbiter] = {}

        # Background thread
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Performance metrics
        self._metrics = MetricsCollector(self._config.tick_rate_hz)

        # User-registered tick hooks: called once per engine loop
        self._tick_hooks: List[Callable[[], None]] = []

        # Counters for diagnostics
        self._total_ticks:   int = 0
        self._total_updates: int = 0

        logger.info(
            "TelemetryEngine created (%.0f Hz, failover=%.0f ms)",
            self._config.tick_rate_hz,
            self._config.arbiter_failover_ms,
        )

    # ── Source management ─────────────────────────────────────────────────────

    def add_source(
        self,
        source: TelemetrySource,
        auto_connect: bool = False,
    ) -> None:
        """Register a TelemetrySource with the engine.

        Args:
            source:       The source to register.
            auto_connect: If True, call ``source.connect()`` immediately.

        Raises:
            DuplicateSourceError: If source.source_id is already registered.
        """
        with self._lock:
            if source.source_id in self._sources:
                raise DuplicateSourceError(source.source_id)
            self._sources[source.source_id] = source

            # Ensure this vehicle has an arbiter
            vid = source.vehicle_id
            if vid not in self._arbiters:
                self._arbiters[vid] = SourceArbiter(
                    failover_threshold_ms=self._config.arbiter_failover_ms
                )
            self._arbiters[vid].register(source.source_id, source.priority)

        if auto_connect:
            source.connect()

        logger.info(
            "Engine: added source '%s' vehicle='%s' priority=%s",
            source.source_id, source.vehicle_id, source.priority.name,
        )

    def remove_source(self, source_id: str, disconnect: bool = True) -> TelemetrySource:
        """Unregister a source.

        Args:
            source_id:  ID of the source to remove.
            disconnect: If True, call ``source.disconnect()`` before removal.

        Returns:
            The removed TelemetrySource.

        Raises:
            SourceNotFoundError: If source_id is not registered.
        """
        with self._lock:
            if source_id not in self._sources:
                raise SourceNotFoundError(source_id)
            source = self._sources.pop(source_id)
            arbiter = self._arbiters.get(source.vehicle_id)
            if arbiter:
                arbiter.deregister(source_id)

        if disconnect:
            try:
                source.disconnect()
            except Exception:  # noqa: BLE001
                logger.exception("Engine: exception disconnecting source '%s'", source_id)

        logger.info("Engine: removed source '%s'", source_id)
        return source

    def get_source(self, source_id: str) -> TelemetrySource:
        """Retrieve a registered source by ID."""
        with self._lock:
            if source_id not in self._sources:
                raise SourceNotFoundError(source_id)
            return self._sources[source_id]

    def list_sources(self) -> List[str]:
        """Return list of registered source IDs."""
        with self._lock:
            return list(self._sources.keys())

    def connect_all(self) -> Dict[str, bool]:
        """Call connect() on every registered source.

        Returns:
            Dict mapping source_id → True (connected) / False (error).
        """
        results: Dict[str, bool] = {}
        with self._lock:
            sources = list(self._sources.values())
        for source in sources:
            try:
                source.connect()
                results[source.source_id] = True
            except Exception:  # noqa: BLE001
                logger.exception("Engine: failed to connect source '%s'", source.source_id)
                results[source.source_id] = False
        return results

    def disconnect_all(self) -> None:
        """Call disconnect() on every registered source."""
        with self._lock:
            sources = list(self._sources.values())
        for source in sources:
            try:
                source.disconnect()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Engine: exception disconnecting '%s'", source.source_id
                )

    # ── Engine lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background polling thread.

        Raises:
            EngineAlreadyRunningError: If the engine is already running.
        """
        with self._lock:
            if self._running:
                raise EngineAlreadyRunningError(
                    "TelemetryEngine is already running. Call stop() first."
                )
            self._running = True
            self._stop_evt.clear()

        self._metrics.record_engine_start()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="TelemetryEngine",
            daemon=True,   # Die when the main thread exits
        )
        self._thread.start()
        logger.info("TelemetryEngine started at %.0f Hz", self._config.tick_rate_hz)

    def stop(self, timeout_s: float = 5.0) -> bool:
        """Stop the background polling thread.

        Args:
            timeout_s: Maximum seconds to wait for the thread to join.

        Returns:
            True if the thread stopped within timeout; False otherwise.
        """
        with self._lock:
            if not self._running:
                return True
            self._running = False

        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            stopped = not self._thread.is_alive()
        else:
            stopped = True

        logger.info("TelemetryEngine stopped (clean=%s)", stopped)
        return stopped

    @property
    def is_running(self) -> bool:
        """True when the background thread is active."""
        with self._lock:
            return self._running

    # ── Tick hooks ────────────────────────────────────────────────────────────

    def add_tick_hook(self, fn: Callable[[], None]) -> None:
        """Register a function to be called once per engine tick.

        Useful for custom processing that needs to run in sync with telemetry.
        Keep hooks fast (< 1 ms) to avoid degrading tick rate.
        """
        with self._lock:
            self._tick_hooks.append(fn)

    def remove_tick_hook(self, fn: Callable[[], None]) -> bool:
        """Remove a previously registered tick hook."""
        with self._lock:
            try:
                self._tick_hooks.remove(fn)
                return True
            except ValueError:
                return False

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_metrics(self) -> EngineMetrics:
        """Return a snapshot of engine performance metrics."""
        with self._lock:
            n_sources = len(self._sources)
        return self._metrics.get_snapshot(n_sources)

    def get_source_info(self) -> List[dict]:
        """Return per-source metric snapshots."""
        with self._lock:
            sources = list(self._sources.values())
        return [s.get_info().to_dict() for s in sources]

    # ── Manual update injection (testing / demos) ─────────────────────────────

    def inject_update(self, update) -> bool:
        """Directly push a DroneStateUpdate to the appropriate StateStore.

        Bypasses source polling and arbitration.  Useful for testing or
        bridging external data pipelines.

        Returns:
            True if the update was accepted by the StateStore; False otherwise.
        """
        return self._push_to_store(update)

    # ── Background loop ───────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Main engine loop.  Runs in the background thread."""
        interval = self._config.tick_interval_s
        next_tick = time.monotonic()

        while not self._stop_evt.is_set():
            # Poll all sources and push updates
            n_updates = self._tick()

            # Record metrics
            self._metrics.record_tick(n_updates)

            # Run tick hooks
            self._run_hooks()

            # Precise sleep: compensate for processing time
            next_tick += interval
            sleep_remaining = next_tick - time.monotonic()
            if sleep_remaining > 0:
                self._stop_evt.wait(timeout=sleep_remaining)
            elif sleep_remaining < -interval:
                # We've fallen more than one full tick behind — re-sync
                next_tick = time.monotonic()
                logger.warning(
                    "TelemetryEngine tick overrun: %.1f ms behind",
                    -sleep_remaining * 1000,
                )

    def _tick(self) -> int:
        """One engine tick: poll → arbitrate → push.  Returns total updates pushed."""
        with self._lock:
            sources = list(self._sources.values())

        # Group sources by vehicle_id
        by_vehicle: Dict[str, Dict[str, list]] = {}
        for source in sources:
            if source.status != SourceStatus.CONNECTED:
                continue
            try:
                updates = source.poll()
                source._record_poll(len(updates))
            except Exception:  # noqa: BLE001
                source._record_error()
                self._metrics.record_source_error(source.source_id)
                logger.exception("Engine: poll error on source '%s'", source.source_id)
                updates = []

            if updates:
                vid = source.vehicle_id
                by_vehicle.setdefault(vid, {})[source.source_id] = updates
                self._metrics.record_source_poll(
                    source.source_id, len(updates), 0
                )

        # Arbitrate and push for each vehicle
        total_pushed = 0
        for vehicle_id, updates_by_source in by_vehicle.items():
            arbiter = self._arbiters.get(vehicle_id)
            if arbiter is None:
                # No arbiter for this vehicle — pass through all
                all_updates = []
                for us in updates_by_source.values():
                    all_updates.extend(us)
                arbitrated = all_updates
            else:
                arbitrated = arbiter.arbitrate(updates_by_source)

            # Cap at max_updates_per_tick to prevent runaway processing
            capped = arbitrated[: self._config.max_updates_per_tick]

            for upd in capped:
                if self._push_to_store(upd):
                    total_pushed += 1

        self._total_ticks   += 1
        self._total_updates += total_pushed
        return total_pushed

    def _push_to_store(self, update) -> bool:
        """Push one DroneStateUpdate to its StateStore."""
        vehicle_id = update.vehicle_id
        try:
            store = StateStore.get_instance(vehicle_id)
        except StoreNotFoundError:
            if self._config.auto_create_stores:
                store = StateStore.create(vehicle_id)
                logger.info(
                    "Engine: auto-created StateStore for vehicle '%s'", vehicle_id
                )
            else:
                logger.warning(
                    "Engine: no StateStore for vehicle '%s' — update dropped",
                    vehicle_id,
                )
                return False
        return store.update_partial(update)

    def _run_hooks(self) -> None:
        with self._lock:
            hooks = list(self._tick_hooks)
        for hook in hooks:
            try:
                hook()
            except Exception:  # noqa: BLE001
                logger.exception("Engine: exception in tick hook %r", hook)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_status_summary(self) -> dict:
        """JSON-serialisable status for monitoring dashboards."""
        metrics = self.get_metrics()
        with self._lock:
            n_sources = len(self._sources)
            running   = self._running
        return {
            "running":         running,
            "n_sources":       n_sources,
            "tick_rate_hz":    round(metrics.tick_rate_hz, 1),
            "tick_jitter_ms":  round(metrics.tick_jitter_ms, 2),
            "updates_per_sec": round(metrics.total_updates_sec, 1),
            "total_updates":   metrics.total_updates,
            "total_ticks":     metrics.total_ticks,
            "uptime_s":        round(metrics.uptime_s, 1),
            "sources":         self.get_source_info(),
        }

    def __repr__(self) -> str:  # noqa: D105
        with self._lock:
            n = len(self._sources)
            running = self._running
        return (
            f"TelemetryEngine(running={running}, "
            f"sources={n}, "
            f"rate={self._config.tick_rate_hz:.0f} Hz)"
        )
