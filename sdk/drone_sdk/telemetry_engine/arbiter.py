"""
telemetry_engine.arbiter
========================
SourceArbiter — priority-based multi-source conflict resolution.

When multiple TelemetrySource objects provide updates for the same vehicle
simultaneously, the arbiter decides which updates to pass through to the
StateStore and which to discard.

Strategy
--------
1. **Priority wins**: Lower SourcePriority integer value wins.
   HARDWARE(0) > SITL(1) > ROS2(2) > SIMULATION(3) > REPLAY(4) > MANUAL(5).

2. **Failover**: If the highest-priority source stops producing data (goes
   stale beyond ``failover_threshold_ms``), the next-priority source
   takes over automatically.

3. **Source mixing**: For orthogonal fields (e.g. one source provides position,
   another provides battery), both are allowed through regardless of priority.
   Conflict only matters when two sources supply the same fields.

4. **Passthrough mode**: When only one source is active, all its updates pass
   through unchanged (no overhead).

Usage by the engine::

    arbiter = SourceArbiter(failover_threshold_ms=500.0)
    arbiter.register("px4", SourcePriority.HARDWARE)
    arbiter.register("sitl", SourcePriority.SITL)
    allowed = arbiter.arbitrate(updates)   # list filtered to allowed sources
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .source import SourcePriority

logger = logging.getLogger(__name__)


@dataclass
class _ArbiterEntry:
    source_id:    str
    priority:     SourcePriority
    last_seen_mono: Optional[float] = None
    is_active:    bool = True


class SourceArbiter:
    """Multi-source priority arbiter.

    Args:
        failover_threshold_ms: How long (ms) a source must be silent before
                               the next-priority source takes over.

    Thread safety: All methods are guarded by an RLock.
    """

    def __init__(self, failover_threshold_ms: float = 500.0) -> None:
        self._threshold_ms = failover_threshold_ms
        self._lock = threading.RLock()
        # source_id → entry
        self._sources: Dict[str, _ArbiterEntry] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, source_id: str, priority: SourcePriority) -> None:
        """Register a source for arbitration."""
        with self._lock:
            self._sources[source_id] = _ArbiterEntry(
                source_id=source_id, priority=priority
            )
        logger.debug("Arbiter: registered source '%s' priority=%s", source_id, priority.name)

    def deregister(self, source_id: str) -> bool:
        """Remove a source from arbitration.  Returns True if it existed."""
        with self._lock:
            existed = source_id in self._sources
            self._sources.pop(source_id, None)
        return existed

    # ── Arbitration ───────────────────────────────────────────────────────────

    def arbitrate(
        self,
        updates_by_source: Dict[str, list],   # source_id → List[DroneStateUpdate]
    ) -> List:
        """Return the subset of updates that should be forwarded.

        Args:
            updates_by_source: Mapping of source_id → list of DroneStateUpdates.

        Returns:
            Flat list of DroneStateUpdates from the winning source(s).
        """
        if not updates_by_source:
            return []

        # Fast path: single source, no conflict
        if len(updates_by_source) == 1:
            source_id, updates = next(iter(updates_by_source.items()))
            self._record_activity(source_id)
            return list(updates)

        now = time.monotonic()
        with self._lock:
            # Find the highest-priority source with recent data
            winning_priority = SourcePriority.MANUAL + 1  # sentinel
            winner_id: Optional[str] = None

            for source_id in updates_by_source:
                entry = self._sources.get(source_id)
                if entry is None:
                    continue  # unregistered source — pass through anyway
                if entry.priority < winning_priority:
                    winning_priority = entry.priority
                    winner_id = source_id

            if winner_id is None:
                # No registered sources — pass everything through
                all_updates = []
                for us in updates_by_source.values():
                    all_updates.extend(us)
                return all_updates

            # Check if a higher-priority source is currently stale
            # If so, fall back to the next available source
            effective_winner = self._resolve_failover(
                winner_id, updates_by_source, now
            )

            # Record activity
            for sid in updates_by_source:
                entry = self._sources.get(sid)
                if entry:
                    entry.last_seen_mono = now

        return list(updates_by_source.get(effective_winner, []))

    def record_no_data(self, source_id: str) -> None:
        """Inform the arbiter that a source produced zero updates this tick."""
        # Staleness is computed lazily in arbitrate() — nothing to do here.
        pass

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_active_source(
        self,
        updates_by_source: Optional[Dict[str, list]] = None,
    ) -> Optional[str]:
        """Return the source_id of the current winner (or None if undecided)."""
        with self._lock:
            best_priority = SourcePriority.MANUAL + 1
            best_id: Optional[str] = None
            for sid, entry in self._sources.items():
                if not entry.is_active:
                    continue
                if entry.priority < best_priority:
                    best_priority = entry.priority
                    best_id = sid
        return best_id

    def get_source_info(self) -> List[dict]:
        with self._lock:
            return [
                {
                    "source_id": e.source_id,
                    "priority":  e.priority.name,
                    "last_seen_ms": (
                        (time.monotonic() - e.last_seen_mono) * 1000
                        if e.last_seen_mono else None
                    ),
                }
                for e in self._sources.values()
            ]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _record_activity(self, source_id: str) -> None:
        with self._lock:
            entry = self._sources.get(source_id)
            if entry:
                entry.last_seen_mono = time.monotonic()

    def _resolve_failover(
        self,
        primary_id: str,
        updates_by_source: Dict[str, list],
        now: float,
    ) -> str:
        """Check if primary is stale; if so return next-best source that has data."""
        primary_entry = self._sources.get(primary_id)
        if primary_entry is None:
            return primary_id

        # Primary is in the update dict — it's active this tick
        if primary_id in updates_by_source:
            return primary_id

        # Primary is registered but produced no updates this tick.
        # Check staleness.
        if primary_entry.last_seen_mono is not None:
            age_ms = (now - primary_entry.last_seen_mono) * 1000
            if age_ms < self._threshold_ms:
                return primary_id   # still within grace period

        # Primary is stale — find next best source
        best_priority = SourcePriority.MANUAL + 1
        best_id: Optional[str] = None
        for sid, entry in self._sources.items():
            if sid == primary_id:
                continue
            if sid in updates_by_source and entry.priority < best_priority:
                best_priority = entry.priority
                best_id = sid

        if best_id is not None:
            logger.info(
                "Arbiter: source '%s' stale → failing over to '%s'",
                primary_id, best_id,
            )
            return best_id

        return primary_id   # no alternative — stick with primary
