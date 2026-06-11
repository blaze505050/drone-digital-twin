"""
telemetry_engine.replayer
=========================
TelemetryReplayer — plays back HDF5 flight logs into the live system.

Use cases
---------
* Validate AI/ML models (PINN, FDD, Battery DT) against real flight data.
* Reproduce and debug in-flight anomalies.
* Generate training datasets at arbitrary playback speed.
* Regression testing: compare current algorithm output against recorded truth.

Playback modes
--------------
* **REALTIME**    — playback at 1× original recording speed.
* **FAST** (N×)   — playback at N× speed (default 10×).
* **INSTANT**     — push all states as fast as possible (for ML training).

Usage::

    replayer = TelemetryReplayer("flight_log.h5", speed=1.0)
    replayer.attach_store("drone_0")

    replayer.start()
    # … wait for completion …
    replayer.stop()

Or in a loop::

    for state in replayer.iter_states("drone_0", speed=10.0):
        model.predict(state)

Python version: 3.9+
"""
from __future__ import annotations

import logging
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import numpy as np

from drone_sdk.state_manager import (
    ArmingState,
    DataSource,
    DroneStateVector,
    FlightMode,
    HealthStatus,
    StateStore,
)
from drone_sdk.state_manager.exceptions import StoreNotFoundError

from .exceptions import ReplayError

logger = logging.getLogger(__name__)

# Fields that must be loaded from HDF5 to reconstruct a DroneStateVector.
# Subset of _SCALAR_FIELDS from recorder.py (excludes enum/bool for reconstruction).
_RECONSTRUCT_FIELDS = [
    "timestamp_wall", "timestamp_mono", "timestamp_sim", "sequence",
    "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az",
    "roll", "pitch", "yaw", "q0", "q1", "q2", "q3",
    "roll_rate", "pitch_rate", "yaw_rate",
    "omega1", "omega2", "omega3", "omega4",
    "battery_voltage", "battery_current", "battery_soc", "battery_soh",
    "battery_remaining_wh", "battery_temperature",
    "gps_lat", "gps_lon", "gps_alt_msl",
    "gps_fix_type", "gps_satellites", "gps_hdop", "gps_vdop",
    "altitude_agl", "altitude_amsl", "airspeed",
    "groundspeed", "vertical_speed", "heading",
    "flight_mode", "arming_state",
    "health_score", "is_valid", "latency_ms",
]


class PlaybackMode(Enum):
    REALTIME = auto()
    FAST     = auto()
    INSTANT  = auto()


class TelemetryReplayer:
    """Replays an HDF5 flight log into the live StateStore system.

    Args:
        path:          Path to the HDF5 file produced by TelemetryRecorder.
        speed:         Playback speed multiplier (1.0 = realtime, 10.0 = 10×).
                       Use ``float('inf')`` for INSTANT mode.
        loop:          If True, replay loops indefinitely.
        on_complete:   Callback fired when replay finishes (non-loop mode).
    """

    def __init__(
        self,
        path:        str | Path,
        speed:       float   = 1.0,
        loop:        bool    = False,
        on_complete: Optional[Callable[[], None]] = None,
    ) -> None:
        self._path        = Path(path)
        self._speed       = speed
        self._loop        = loop
        self._on_complete = on_complete

        self._lock     = threading.RLock()
        self._running  = False
        self._paused   = False
        self._stop_evt = threading.Event()
        self._thread:  Optional[threading.Thread] = None

        # vehicle_id → vehicle_id (target stores)
        self._target_vehicles: List[str] = []

        if not self._path.exists():
            raise ReplayError(f"HDF5 file not found: {self._path}")

    def attach_store(self, vehicle_id: str) -> None:
        """Set a vehicle whose states will be replayed."""
        with self._lock:
            if vehicle_id not in self._target_vehicles:
                self._target_vehicles.append(vehicle_id)

    def start(self) -> None:
        """Start replay in a background thread."""
        with self._lock:
            if self._running:
                raise ReplayError("Replayer is already running.")
            self._running = True
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._replay_loop,
            name="TelemetryReplayer",
            daemon=True,
        )
        self._thread.start()
        logger.info("TelemetryReplayer started (speed=%.1f×)", self._speed)

    def stop(self) -> None:
        """Stop the replay loop and wait for the background thread."""
        with self._lock:
            self._running = False
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=10.0)
        logger.info("TelemetryReplayer stopped.")

    def pause(self) -> None:
        """Pause replay (can be resumed)."""
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        """Resume a paused replay."""
        with self._lock:
            self._paused = False

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    # ── Synchronous iterator (for ML training loops) ──────────────────────────

    def iter_states(
        self,
        vehicle_id: str,
        speed: Optional[float] = None,
    ) -> Iterator[DroneStateVector]:
        """Yield all recorded states for a vehicle, respecting playback speed.

        Args:
            vehicle_id: Vehicle whose log to iterate.
            speed:      Override instance speed for this iteration.

        Yields:
            DroneStateVector, one per recorded frame.

        Example::

            for state in replayer.iter_states("drone_0", speed=float("inf")):
                prediction = pinn_model.forward(state.position_ned())
        """
        effective_speed = speed if speed is not None else self._speed
        states = self._load_states(vehicle_id)

        if not states:
            return

        replay_start = time.monotonic()
        first_t = states[0].timestamp_wall

        for state in states:
            if effective_speed != float("inf"):
                elapsed_real  = time.monotonic() - replay_start
                elapsed_sim   = (state.timestamp_wall - first_t) / effective_speed
                sleep = elapsed_sim - elapsed_real
                if sleep > 0:
                    time.sleep(sleep)
            yield state

    # ── Internal ──────────────────────────────────────────────────────────────

    def _replay_loop(self) -> None:
        """Background replay thread."""
        while not self._stop_evt.is_set():
            with self._lock:
                targets = list(self._target_vehicles)

            for vehicle_id in targets:
                if self._stop_evt.is_set():
                    break
                self._replay_vehicle(vehicle_id)

            if not self._loop:
                break
            if not self._stop_evt.is_set() and self._loop:
                logger.info("TelemetryReplayer: looping replay.")

        with self._lock:
            self._running = False

        if self._on_complete:
            try:
                self._on_complete()
            except Exception:  # noqa: BLE001
                logger.exception("Replayer: on_complete callback raised")

    def _replay_vehicle(self, vehicle_id: str) -> None:
        """Replay all states for one vehicle into its StateStore."""
        states = self._load_states(vehicle_id)
        if not states:
            logger.warning("Replayer: no states found for vehicle '%s'", vehicle_id)
            return

        # Ensure StateStore exists
        try:
            store = StateStore.get_instance(vehicle_id)
        except StoreNotFoundError:
            store = StateStore.create(vehicle_id)
            logger.info("Replayer: auto-created StateStore for '%s'", vehicle_id)

        first_wall   = states[0].timestamp_wall
        replay_start = time.monotonic()

        for state in states:
            if self._stop_evt.is_set():
                return

            # Wait while paused
            while self._paused and not self._stop_evt.is_set():
                time.sleep(0.05)

            if self._speed != float("inf"):
                elapsed_real = time.monotonic() - replay_start
                elapsed_sim  = (state.timestamp_wall - first_wall) / self._speed
                sleep = elapsed_sim - elapsed_real
                if sleep > 0.001:
                    self._stop_evt.wait(timeout=sleep)

            # Stamp replay source
            replayed = state.copy_with(source=DataSource.REPLAY)
            store.update(replayed)

        logger.info(
            "Replayer: finished vehicle '%s' (%d states)", vehicle_id, len(states)
        )

    def _load_states(self, vehicle_id: str) -> List[DroneStateVector]:
        """Load all states for one vehicle from HDF5."""
        try:
            import h5py
        except ImportError as exc:
            raise ReplayError("h5py not installed. Run: pip install h5py") from exc

        states: List[DroneStateVector] = []
        try:
            with h5py.File(self._path, "r") as f:
                grp_path = f"vehicles/{vehicle_id}/state"
                if grp_path not in f:
                    logger.error(
                        "Replayer: '%s' not found in %s", grp_path, self._path
                    )
                    return []

                grp = f[grp_path]
                n = grp["timestamp_wall"].shape[0]

                # Load all columns into numpy arrays first (single I/O per field)
                columns: dict = {}
                for field_name in _RECONSTRUCT_FIELDS:
                    if field_name in grp:
                        columns[field_name] = grp[field_name][:]
                    else:
                        logger.warning(
                            "Replayer: field '%s' missing from log", field_name
                        )

                # Reconstruct DroneStateVector for each row
                for i in range(n):
                    kwargs = {"vehicle_id": vehicle_id, "source": DataSource.REPLAY}
                    for field_name, arr in columns.items():
                        val = arr[i]
                        # Convert numpy scalar to Python type
                        if isinstance(val, (np.integer,)):
                            val = int(val)
                        elif isinstance(val, (np.floating,)):
                            val = float(val)
                        elif isinstance(val, (np.bool_,)):
                            val = bool(val)
                        kwargs[field_name] = val

                    # Restore enums
                    kwargs["flight_mode"]  = FlightMode(int(kwargs.get("flight_mode", 0)))
                    kwargs["arming_state"] = ArmingState(int(kwargs.get("arming_state", 0)))
                    kwargs["health_status"] = HealthStatus.NO_DATA
                    kwargs["is_valid"] = bool(kwargs.get("is_valid", 0))

                    states.append(DroneStateVector(**kwargs))

        except Exception as exc:
            logger.exception("Replayer: failed to load '%s': %s", vehicle_id, exc)

        logger.info(
            "Replayer: loaded %d states for '%s' from %s",
            len(states), vehicle_id, self._path,
        )
        return states
