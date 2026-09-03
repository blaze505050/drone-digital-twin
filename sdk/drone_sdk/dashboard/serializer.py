"""
dashboard.serializer
====================
StateSerializer — converts DroneStateVector to JSON with field selection,
unit conversion, and differential compression.

Three serialisation profiles
-----------------------------
FULL        All fields. Used for initial client connection sync.
REALTIME    High-frequency fields only (position, attitude, battery SoC).
            Used for the live 3D visualisation at 20-50 Hz.
TELEMETRY   Extended flight data for the data panel (all but covariances).

Differential mode
-----------------
In REALTIME mode, only fields that changed by more than their threshold are
included in the JSON payload.  This reduces WebSocket traffic by ~60-80%
during steady-state flight.

Unit presentation
-----------------
Internal units (radians, NED metres) are converted to display units
(degrees, ENU-friendly labels) for the web UI.  The raw internal values
are always available via the "raw" sub-key when include_raw=True.

Python version: 3.9+
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from drone_sdk.state_manager import DroneStateVector


class SerialiseProfile(str, Enum):
    FULL      = "full"
    REALTIME  = "realtime"
    TELEMETRY = "telemetry"


# ── Field groups ──────────────────────────────────────────────────────────────

_REALTIME_FIELDS: Set[str] = {
    "vehicle_id", "sequence", "timestamp_wall", "timestamp_sim",
    # Position (NED)
    "x", "y", "z", "altitude_agl",
    # Velocity
    "vx", "vy", "vz", "groundspeed", "vertical_speed",
    # Attitude (display as degrees)
    "roll", "pitch", "yaw", "heading",
    # Angular rates
    "roll_rate", "pitch_rate", "yaw_rate",
    # Quaternion (for 3D renderer)
    "q0", "q1", "q2", "q3",
    # Rotor
    "omega1", "omega2", "omega3", "omega4",
    # Battery
    "battery_soc", "battery_voltage",
    # Status
    "flight_mode", "arming_state", "health_status", "health_score", "is_valid",
}

_TELEMETRY_FIELDS: Set[str] = _REALTIME_FIELDS | {
    "battery_current", "battery_soh", "battery_remaining_wh", "battery_temperature",
    "gps_lat", "gps_lon", "gps_alt_msl",
    "gps_fix_type", "gps_satellites", "gps_hdop", "gps_vdop",
    "altitude_amsl", "airspeed",
    "ax", "ay", "az",
    "latency_ms", "source",
}

# Change thresholds for differential serialisation
_DIFF_THRESHOLDS: Dict[str, float] = {
    "x": 0.005, "y": 0.005, "z": 0.005,
    "vx": 0.01, "vy": 0.01, "vz": 0.01,
    "roll": 0.001, "pitch": 0.001, "yaw": 0.002,
    "q0": 0.0005, "q1": 0.0005, "q2": 0.0005, "q3": 0.0005,
    "omega1": 1.0, "omega2": 1.0, "omega3": 1.0, "omega4": 1.0,
    "battery_soc": 0.001, "battery_voltage": 0.01,
    "groundspeed": 0.01, "altitude_agl": 0.005,
    "roll_rate": 0.005, "pitch_rate": 0.005, "yaw_rate": 0.005,
}


@dataclass
class SerialiseConfig:
    """Configuration for StateSerializer."""
    profile:     SerialiseProfile = SerialiseProfile.REALTIME
    include_raw: bool             = False   # Include raw rad values alongside degrees
    diff_mode:   bool             = True    # Only send changed fields
    rad_to_deg:  bool             = True    # Convert angular fields to degrees
    include_ts:  bool             = True    # Include serialise timestamp


class StateSerializer:
    """Converts DroneStateVector to optimised JSON for WebSocket streaming.

    Args:
        config: SerialiseConfig with profile and options.

    Example::

        ser = StateSerializer()
        json_str = ser.to_json(state)
        delta    = ser.to_diff_json(state)   # Only changed fields
    """

    # Angular fields that get converted to degrees when rad_to_deg=True
    _RAD_FIELDS = {
        "roll", "pitch", "yaw",
        "roll_rate", "pitch_rate", "yaw_rate",
    }

    def __init__(self, config: Optional[SerialiseConfig] = None) -> None:
        self._cfg     = config or SerialiseConfig()
        self._prev:   Optional[Dict[str, Any]] = None
        self._prev_seq: int = -1

    # ── Public API ────────────────────────────────────────────────────────────

    def to_json(
        self,
        state: DroneStateVector,
        profile: Optional[SerialiseProfile] = None,
    ) -> str:
        """Serialise a complete state snapshot to JSON.

        Args:
            state:   DroneStateVector to serialise.
            profile: Override the instance profile for this call.

        Returns:
            JSON string.
        """
        d = self._build_dict(state, profile or self._cfg.profile)
        return json.dumps(d, separators=(",", ":"))

    def to_diff_json(self, state: DroneStateVector) -> str:
        """Serialise only fields that changed since the last call.

        Always includes vehicle_id, sequence, and timestamp_wall
        for frame synchronisation.

        Returns:
            JSON string (may be very short if few fields changed).
        """
        d   = self._build_dict(state, SerialiseProfile.REALTIME)
        out = self._compute_diff(d, state.sequence)
        return json.dumps(out, separators=(",", ":"))

    def reset(self) -> None:
        """Clear differential state (force next call to send full payload)."""
        self._prev     = None
        self._prev_seq = -1

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_dict(
        self,
        state: DroneStateVector,
        profile: SerialiseProfile,
    ) -> Dict[str, Any]:
        """Build the base dictionary for a state."""
        fields_to_include: Set[str]
        if profile == SerialiseProfile.FULL:
            fields_to_include = set()   # All fields
        elif profile == SerialiseProfile.REALTIME:
            fields_to_include = _REALTIME_FIELDS
        else:
            fields_to_include = _TELEMETRY_FIELDS

        d: Dict[str, Any] = {}

        raw_d = state.to_dict()   # gets enum→primitive conversion

        for key, val in raw_d.items():
            if fields_to_include and key not in fields_to_include:
                continue

            # Angular conversion
            if self._cfg.rad_to_deg and key in self._RAD_FIELDS:
                if self._cfg.include_raw:
                    d[f"{key}_rad"] = round(val, 6)
                d[key] = round(math.degrees(val), 4)
            elif isinstance(val, float):
                d[key] = round(val, 6)
            else:
                d[key] = val

        # Add computed display-friendly fields
        d["altitude_agl_ft"] = round(state.altitude_agl * 3.28084, 2)
        d["groundspeed_kph"]  = round(state.groundspeed * 3.6, 2)
        d["armed"]            = state.is_armed
        d["airborne"]         = state.is_airborne

        if self._cfg.include_ts:
            d["_serialised_at"] = round(time.time(), 3)

        return d

    def _compute_diff(
        self,
        current: Dict[str, Any],
        sequence: int,
    ) -> Dict[str, Any]:
        """Return only changed fields from current vs previous snapshot."""
        # Always include frame identity fields
        diff: Dict[str, Any] = {
            "vehicle_id":     current["vehicle_id"],
            "sequence":       current["sequence"],
            "timestamp_wall": current["timestamp_wall"],
        }

        if self._prev is None or sequence != self._prev_seq + 1:
            # No previous or sequence gap — send full payload
            diff.update(current)
        else:
            for key, val in current.items():
                if key in diff:
                    continue
                prev_val = self._prev.get(key)
                if prev_val is None:
                    diff[key] = val
                    continue
                threshold = _DIFF_THRESHOLDS.get(key)
                if threshold is not None:
                    if isinstance(val, (int, float)) and isinstance(prev_val, (int, float)):
                        if abs(val - prev_val) >= threshold:
                            diff[key] = val
                elif val != prev_val:
                    diff[key] = val

        self._prev     = current
        self._prev_seq = sequence
        return diff

    def get_profile_fields(self, profile: SerialiseProfile) -> List[str]:
        """Return the list of field names for a given profile."""
        if profile == SerialiseProfile.REALTIME:
            return sorted(_REALTIME_FIELDS)
        elif profile == SerialiseProfile.TELEMETRY:
            return sorted(_TELEMETRY_FIELDS)
        else:
            return []   # FULL = all fields from DroneStateVector


# ── Convenience functions ─────────────────────────────────────────────────────

def state_to_json(
    state: DroneStateVector,
    profile: SerialiseProfile = SerialiseProfile.REALTIME,
) -> str:
    """Module-level convenience: serialise once without creating a StateSerializer."""
    return StateSerializer(SerialiseConfig(profile=profile)).to_json(state)


def states_to_json_array(
    states: List[DroneStateVector],
    profile: SerialiseProfile = SerialiseProfile.TELEMETRY,
) -> str:
    """Serialise a list of states as a JSON array (for history export)."""
    ser = StateSerializer(SerialiseConfig(profile=profile, diff_mode=False))
    items = [json.loads(ser.to_json(s)) for s in states]
    return json.dumps(items, separators=(",", ":"))
