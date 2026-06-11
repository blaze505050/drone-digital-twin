"""
telemetry_engine.recorder
=========================
TelemetryRecorder — high-performance HDF5 flight log writer.

Writes every DroneStateVector pushed to a StateStore into a structured
HDF5 file.  The file layout is:

    flight_log.h5
    ├── vehicles/
    │   ├── drone_0/
    │   │   ├── state/          # One dataset per DroneStateVector field
    │   │   │   ├── timestamp_wall  (float64, shape=(N,))
    │   │   │   ├── x               (float64, shape=(N,))
    │   │   │   ├── y               (float64, shape=(N,))
    │   │   │   ⋮
    │   │   └── metadata/
    │   │       ├── vehicle_id       (str)
    │   │       ├── start_wall       (float64)
    │   │       └── vehicle_config   (JSON string)
    │   └── drone_1/
    │       └── …
    └── session_metadata (JSON string)

Design
------
* Write-buffering: states are accumulated in a list and flushed to disk
  every ``flush_interval_s`` seconds (default 1.0) or when the buffer
  reaches ``flush_count`` entries (default 500).  This amortises the HDF5
  write overhead across many updates.

* Chunked datasets with compression (gzip level 4) are used to minimise
  file size while maintaining fast random access.

* The recorder subscribes to ``EventType.STATE_UPDATED`` on each vehicle's
  StateStore, so it requires no changes to the TelemetryEngine.

* Graceful close: ``stop()`` flushes all pending buffers and closes the file.

Dependencies: h5py (pip install h5py)

Python version: 3.9+
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from drone_sdk.state_manager import DroneStateVector, EventType, StateStore

from .exceptions import RecordingError

logger = logging.getLogger(__name__)

# Fields of DroneStateVector to record (excludes non-numeric metadata).
_SCALAR_FIELDS = [
    "timestamp_wall", "timestamp_mono", "timestamp_sim", "sequence",
    "x", "y", "z",
    "vx", "vy", "vz",
    "ax", "ay", "az",
    "roll", "pitch", "yaw",
    "q0", "q1", "q2", "q3",
    "roll_rate", "pitch_rate", "yaw_rate",
    "omega1", "omega2", "omega3", "omega4",
    "battery_voltage", "battery_current",
    "battery_soc", "battery_soh",
    "battery_remaining_wh", "battery_temperature",
    "gps_lat", "gps_lon", "gps_alt_msl",
    "gps_fix_type", "gps_satellites", "gps_hdop", "gps_vdop",
    "altitude_agl", "altitude_amsl", "airspeed",
    "groundspeed", "vertical_speed", "heading",
    "flight_mode",   # stored as int
    "arming_state",  # stored as int
    "health_score",
    "is_valid",      # stored as 0/1 int
    "latency_ms",
]


class _VehicleBuffer:
    """Per-vehicle write buffer and HDF5 dataset manager."""

    def __init__(self, vehicle_id: str, hdf_group, chunk_size: int = 500) -> None:
        self._vehicle_id = vehicle_id
        self._grp        = hdf_group
        self._chunk      = chunk_size
        self._lock       = threading.Lock()
        self._buffer: List[DroneStateVector] = []
        self._n_written  = 0

        # Lazily create datasets on first flush
        self._datasets_created = False

    def append(self, state: DroneStateVector) -> None:
        with self._lock:
            self._buffer.append(state)

    def pending(self) -> int:
        with self._lock:
            return len(self._buffer)

    def flush(self) -> int:
        """Write buffered states to HDF5.  Returns number of records written."""
        import numpy as np

        with self._lock:
            if not self._buffer:
                return 0
            batch = self._buffer[:]
            self._buffer.clear()

        n = len(batch)

        try:
            if not self._datasets_created:
                self._create_datasets()
                self._datasets_created = True

            # Build column arrays
            for field_name in _SCALAR_FIELDS:
                ds = self._grp["state"][field_name]
                values = self._extract_column(batch, field_name)
                arr = np.array(values, dtype=ds.dtype)

                # Resize and write
                old_size = ds.shape[0]
                ds.resize(old_size + n, axis=0)
                ds[old_size:old_size + n] = arr

            self._n_written += n
            logger.debug(
                "Recorder[%s]: flushed %d states (total=%d)",
                self._vehicle_id, n, self._n_written,
            )
        except Exception as exc:
            logger.exception(
                "Recorder[%s]: HDF5 flush error: %s", self._vehicle_id, exc
            )

        return n

    def _create_datasets(self) -> None:
        """Create extendable datasets for each recorded field."""
        import numpy as np

        state_grp = self._grp.require_group("state")

        int_fields   = {"sequence", "gps_fix_type", "gps_satellites",
                         "flight_mode", "arming_state", "is_valid"}
        # float_fields = set(_SCALAR_FIELDS) - int_fields  # implicit: else-branch

        for field_name in _SCALAR_FIELDS:
            if field_name in int_fields:
                dtype = np.int64
            else:
                dtype = np.float64

            state_grp.create_dataset(
                field_name,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=(self._chunk,),
                compression="gzip",
                compression_opts=4,
            )

    @staticmethod
    def _extract_column(
        batch: List[DroneStateVector],
        field_name: str,
    ) -> list:
        """Extract one field from a list of states as a flat list."""
        values = []
        for state in batch:
            val = getattr(state, field_name)
            # Enum → int, bool → int
            if hasattr(val, "value"):
                val = int(val)
            elif isinstance(val, bool):
                val = int(val)
            values.append(val)
        return values


class TelemetryRecorder:
    """Records all DroneStateVector updates to a structured HDF5 file.

    Usage::

        recorder = TelemetryRecorder("flight_log.h5")
        recorder.attach_store("drone_0")  # subscribe to that vehicle's events
        recorder.start()

        # … flight …

        recorder.stop()   # flushes buffers and closes file

    Reading logs::

        import h5py
        with h5py.File("flight_log.h5", "r") as f:
            x   = f["vehicles/drone_0/state/x"][:]
            t   = f["vehicles/drone_0/state/timestamp_wall"][:]

    Args:
        path:              Output HDF5 file path.
        flush_interval_s:  How often to flush buffers to disk.
        flush_count:       Buffer size trigger for an early flush.
        chunk_size:        HDF5 dataset chunk size (rows per chunk).
    """

    def __init__(
        self,
        path: str | Path,
        flush_interval_s: float = 1.0,
        flush_count:      int   = 500,
        chunk_size:       int   = 500,
    ) -> None:
        self._path            = Path(path)
        self._flush_interval  = flush_interval_s
        self._flush_count     = flush_count
        self._chunk_size      = chunk_size

        self._lock    = threading.Lock()
        self._running = False
        self._file    = None   # h5py.File
        self._buffers: Dict[str, _VehicleBuffer] = {}
        self._sub_ids: Dict[str, str] = {}   # vehicle_id → subscriber_id
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    def attach_store(self, vehicle_id: str) -> None:
        """Subscribe to STATE_UPDATED events from a vehicle's StateStore.

        The StateStore must already exist.

        Args:
            vehicle_id: Vehicle to record.
        """
        store = StateStore.get_instance(vehicle_id)
        sid = store.subscribe(
            EventType.STATE_UPDATED,
            self._on_state_updated,
            subscriber_id=f"recorder_{vehicle_id}",
        )
        with self._lock:
            self._sub_ids[vehicle_id] = sid
        logger.info("Recorder: attached to vehicle '%s'", vehicle_id)

    def detach_store(self, vehicle_id: str) -> None:
        """Unsubscribe from a vehicle's StateStore."""
        with self._lock:
            sid = self._sub_ids.pop(vehicle_id, None)
        if sid:
            try:
                store = StateStore.get_instance(vehicle_id)
                store.unsubscribe(sid)
            except Exception:  # noqa: BLE001
                pass

    def start(self) -> None:
        """Open the HDF5 file and start the flush thread."""
        try:
            import h5py
        except ImportError as exc:
            raise RecordingError(
                "h5py is not installed. Run: pip install h5py"
            ) from exc

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._path, "w")

        # Write session metadata
        assert self._file is not None, "HDF5 file handle not initialized"
        self._file.attrs["session_start_wall"] = time.time()
        self._file.attrs["session_start_iso"]  = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._file.attrs["platform"]           = "drone_digital_twin v1.0"

        self._vehicles_grp = self._file.require_group("vehicles")

        with self._lock:
            self._running = True
        self._stop_evt.clear()

        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="TelemetryRecorder",
            daemon=True,
        )
        self._flush_thread.start()
        logger.info("TelemetryRecorder started: %s", self._path)

    def stop(self) -> int:
        """Flush all buffers, close file, and unsubscribe from all stores.

        Returns:
            Total number of states written to file.
        """
        with self._lock:
            self._running = False
        self._stop_evt.set()

        if self._flush_thread:
            self._flush_thread.join(timeout=10.0)

        total = self._flush_all_buffers()

        if self._file:
            self._file.attrs["session_end_wall"]   = time.time()
            self._file.attrs["total_states_written"] = total
            self._file.close()
            self._file = None

        # Detach all stores
        for vid in list(self._sub_ids.keys()):
            self.detach_store(vid)

        logger.info("TelemetryRecorder stopped. Total states written: %d", total)
        return total

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_pending_count(self) -> Dict[str, int]:
        """Return number of buffered (unflushed) states per vehicle."""
        with self._lock:
            return {vid: buf.pending() for vid, buf in self._buffers.items()}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_state_updated(self, event) -> None:
        """Callback: receives STATE_UPDATED events from StateStore."""
        state: DroneStateVector = event.data
        vehicle_id = state.vehicle_id

        with self._lock:
            if not self._running:
                return
            if vehicle_id not in self._buffers:
                grp = self._vehicles_grp.require_group(vehicle_id)
                self._buffers[vehicle_id] = _VehicleBuffer(
                    vehicle_id, grp, self._chunk_size
                )
            buf = self._buffers[vehicle_id]

        buf.append(state)

        # Early flush if buffer is large
        if buf.pending() >= self._flush_count:
            buf.flush()

    def _flush_loop(self) -> None:
        """Background thread: periodic buffer flush."""
        while not self._stop_evt.is_set():
            self._stop_evt.wait(timeout=self._flush_interval)
            self._flush_all_buffers()

    def _flush_all_buffers(self) -> int:
        with self._lock:
            bufs = list(self._buffers.values())
        total = sum(buf.flush() for buf in bufs)
        return total
