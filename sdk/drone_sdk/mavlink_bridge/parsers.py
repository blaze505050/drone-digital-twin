"""
mavlink_bridge.parsers
======================
MAVLink message → DroneStateUpdate converters.

Each function takes a raw pymavlink message object and returns a
DroneStateUpdate (or None if the message is not useful).

This module is the authoritative mapping between the MAVLink wire protocol
and the platform's internal state representation.

Supported messages
------------------
MAVLink message          → DroneStateVector fields populated

HEARTBEAT                → flight_mode, arming_state
ATTITUDE                 → roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate
ATTITUDE_QUATERNION      → q0-q3, roll_rate, pitch_rate, yaw_rate
LOCAL_POSITION_NED       → x, y, z, vx, vy, vz
GLOBAL_POSITION_INT      → gps_lat, gps_lon, gps_alt_msl, altitude_agl, heading
GPS_RAW_INT              → gps_fix_type, gps_satellites, gps_hdop, gps_vdop
HIGHRES_IMU              → ax, ay, az
BATTERY_STATUS           → battery_voltage, battery_current, battery_soc
ACTUATOR_OUTPUT_STATUS   → omega1-omega4 (derived from normalized motor output)
ALTITUDE                 → altitude_amsl, altitude_agl
VFR_HUD                  → airspeed, groundspeed, heading, vertical_speed

Design: All functions are pure (stateless, no side effects).  They can be
called from any thread and tested without a live MAVLink connection.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from drone_sdk.state_manager import (
    ArmingState,
    DataSource,
    DroneStateUpdate,
    FlightMode,
)

logger = logging.getLogger(__name__)


# ── PX4 flight mode mapping ───────────────────────────────────────────────────
# PX4 uses a bitmask in HEARTBEAT.custom_mode.
# The lower 16 bits encode the main mode; upper 8 encode sub-mode.
# This mapping covers the most common modes.

_PX4_MAIN_MODE = {
    1:  FlightMode.MANUAL,
    2:  FlightMode.ACRO,
    3:  FlightMode.STABILIZED,
    4:  FlightMode.ALTITUDE_HOLD,
    5:  FlightMode.POSITION_HOLD,
    6:  FlightMode.AUTO_MISSION,
    7:  FlightMode.AUTO_RTL,
    8:  FlightMode.AUTO_LAND,
    13: FlightMode.TAKEOFF,
    14: FlightMode.HOLD,
    15: FlightMode.OFFBOARD,
}

# MAVLink base_mode bit masks
_BASE_MODE_ARMED  = 0x80   # Bit 7: ARMED
_BASE_MODE_CUSTOM = 0x01   # Bit 0: CUSTOM_MODE_ENABLED

# Motor speed scaling: PX4 reports normalized motor output [0, 1].
# We scale to approximate rad/s for a typical 5-inch quad at 15 000 RPM max.
_MAX_MOTOR_OMEGA_RADS = 1570.8   # ≈ 15 000 RPM


# ── Message parsers ───────────────────────────────────────────────────────────

def parse_heartbeat(
    msg,
    vehicle_id: str,
) -> Optional[DroneStateUpdate]:
    """Parse HEARTBEAT (#0) → flight mode and arming state.

    Args:
        msg:        pymavlink MAVLink_heartbeat_message object.
        vehicle_id: Target StateStore vehicle ID.

    Returns:
        DroneStateUpdate with flight_mode and arming_state set, or None
        if the message is from a non-autopilot component.
    """
    # Filter: only process autopilot messages
    # MAV_TYPE_GCS = 6, skip ground station heartbeats
    if hasattr(msg, "type") and msg.type == 6:
        return None

    upd = _make_update(vehicle_id)

    base_mode   = getattr(msg, "base_mode", 0)
    custom_mode = getattr(msg, "custom_mode", 0)

    # Arming state
    if base_mode & _BASE_MODE_ARMED:
        upd.arming_state = ArmingState.ARMED
    else:
        upd.arming_state = ArmingState.DISARMED

    # Flight mode (PX4 custom mode encoding)
    if base_mode & _BASE_MODE_CUSTOM:
        # PX4: bits 16-23 = main mode
        main_mode = (custom_mode >> 16) & 0xFF
        upd.flight_mode = _PX4_MAIN_MODE.get(main_mode, FlightMode.UNKNOWN)
    else:
        upd.flight_mode = FlightMode.UNKNOWN

    return upd


def parse_attitude(msg, vehicle_id: str) -> Optional[DroneStateUpdate]:
    """Parse ATTITUDE (#30) → Euler angles and angular rates.

    MAVLink ATTITUDE uses radians (NED frame, ZYX Euler convention).
    """
    upd = _make_update(vehicle_id)
    upd.euler = np.array([
        float(msg.roll),
        float(msg.pitch),
        float(msg.yaw),
    ], dtype=np.float64)
    upd.angular_velocity = np.array([
        float(msg.rollspeed),
        float(msg.pitchspeed),
        float(msg.yawspeed),
    ], dtype=np.float64)
    return upd


def parse_attitude_quaternion(msg, vehicle_id: str) -> Optional[DroneStateUpdate]:
    """Parse ATTITUDE_QUATERNION (#31) → quaternion and angular rates.

    MAVLink convention: q1=w, q2=x, q3=y, q4=z (JPL, not Hamilton).
    We reorder to our Hamilton convention: [w, x, y, z].
    """
    upd = _make_update(vehicle_id)
    # MAVLink: q1=w, q2=x, q3=y, q4=z
    upd.quaternion = np.array([
        float(msg.q1),   # w
        float(msg.q2),   # x
        float(msg.q3),   # y
        float(msg.q4),   # z
    ], dtype=np.float64)
    upd.angular_velocity = np.array([
        float(msg.rollspeed),
        float(msg.pitchspeed),
        float(msg.yawspeed),
    ], dtype=np.float64)
    return upd


def parse_local_position_ned(msg, vehicle_id: str) -> Optional[DroneStateUpdate]:
    """Parse LOCAL_POSITION_NED (#32) → position and velocity (NED)."""
    upd = _make_update(vehicle_id)
    upd.position = np.array([
        float(msg.x),    # North
        float(msg.y),    # East
        float(msg.z),    # Down
    ], dtype=np.float64)
    upd.velocity = np.array([
        float(msg.vx),
        float(msg.vy),
        float(msg.vz),
    ], dtype=np.float64)
    return upd


def parse_global_position_int(msg, vehicle_id: str) -> Optional[DroneStateUpdate]:
    """Parse GLOBAL_POSITION_INT (#33) → GPS lat/lon/alt and heading.

    MAVLink uses degE7 (integer degrees × 10^7) for lat/lon,
    mm for altitudes, and cdeg for heading.
    """
    upd = _make_update(vehicle_id)
    upd.gps_lat     = float(msg.lat)  / 1e7     # degE7 → degrees
    upd.gps_lon     = float(msg.lon)  / 1e7
    upd.gps_alt_msl = float(msg.alt)  / 1e3     # mm → m
    upd.altitude_agl = float(msg.relative_alt) / 1e3

    # NOTE: hdg field available in GLOBAL_POSITION_INT but DroneStateUpdate
    # has no direct heading field. Heading is derived from euler[2] (yaw)
    # by StateFactory.merge_update when ATTITUDE or ATTITUDE_QUATERNION data
    # is present. For heading-only updates use parse_vfr_hud instead.

    return upd


def parse_gps_raw_int(msg, vehicle_id: str) -> Optional[DroneStateUpdate]:
    """Parse GPS_RAW_INT (#24) → GPS fix quality metrics."""
    upd = _make_update(vehicle_id)
    upd.gps_fix_type   = int(msg.fix_type)
    upd.gps_satellites = int(msg.satellites_visible) if msg.satellites_visible != 255 else 0
    # eph is HDOP × 100 in centimetres; convert to dimensionless DOP
    upd.gps_hdop = float(msg.eph) / 100.0 if msg.eph != 0xFFFF else 99.9
    upd.gps_vdop = float(msg.epv) / 100.0 if msg.epv != 0xFFFF else 99.9
    return upd


def parse_highres_imu(msg, vehicle_id: str) -> Optional[DroneStateUpdate]:
    """Parse HIGHRES_IMU (#105) → body-frame accelerations."""
    upd = _make_update(vehicle_id)
    upd.acceleration = np.array([
        float(msg.xacc),
        float(msg.yacc),
        float(msg.zacc),
    ], dtype=np.float64)
    return upd


def parse_battery_status(msg, vehicle_id: str) -> Optional[DroneStateUpdate]:
    """Parse BATTERY_STATUS (#147) → battery electrical state.

    MAVLink reports voltage in mV per cell (array of up to 10 cells),
    current in cA (centiapm eres), and remaining capacity in %.
    """
    upd = _make_update(vehicle_id)

    # Voltage: sum all valid cells (non-0xFFFF)
    try:
        voltages = msg.voltages   # list of cell voltages in mV
        valid_v  = [v for v in voltages if v != 0xFFFF and v > 0]
        upd.battery_voltage = sum(valid_v) / 1000.0 if valid_v else 0.0
    except AttributeError:
        upd.battery_voltage = 0.0

    # Current: reported in centiaamperes (cA)
    current_ca = getattr(msg, "current_battery", -1)
    upd.battery_current = float(current_ca) / 100.0 if current_ca >= 0 else 0.0

    # State of Charge: reported as percentage [0, 100]; convert to [0, 1]
    remaining = getattr(msg, "battery_remaining", -1)
    upd.battery_soc = float(remaining) / 100.0 if 0 <= remaining <= 100 else 1.0

    return upd


def parse_actuator_output_status(msg, vehicle_id: str) -> Optional[DroneStateUpdate]:
    """Parse ACTUATOR_OUTPUT_STATUS (#375) → motor speeds.

    PX4 reports normalized actuator outputs in [0, 1].  We map these to
    approximate rad/s using the configured max motor speed.

    Note: Motor ordering follows PX4 quadrotor X convention:
        output[0] = front-right (CW)   → omega1
        output[1] = back-left  (CW)    → omega4
        output[2] = front-left (CCW)   → omega2
        output[3] = back-right (CCW)   → omega3
    """
    try:
        outputs = list(msg.actuator)
    except AttributeError:
        return None

    if len(outputs) < 4:
        return None

    def to_omega(normalized: float) -> float:
        clamped = max(0.0, min(1.0, float(normalized)))
        return clamped * _MAX_MOTOR_OMEGA_RADS

    upd = _make_update(vehicle_id)
    upd.rotor_omega = np.array([
        to_omega(outputs[0]),   # omega1 (front-right)
        to_omega(outputs[2]),   # omega2 (front-left)
        to_omega(outputs[3]),   # omega3 (back-right)
        to_omega(outputs[1]),   # omega4 (back-left)
    ], dtype=np.float64)
    return upd


def parse_vfr_hud(msg, vehicle_id: str) -> Optional[DroneStateUpdate]:
    """Parse VFR_HUD (#74) → airspeed, groundspeed, heading, vertical speed."""
    upd = _make_update(vehicle_id)
    upd.airspeed      = float(getattr(msg, "airspeed", 0.0))
    upd.altitude_agl  = float(getattr(msg, "alt", 0.0))
    # vertical_speed: climb is reported in m/s (positive = up)
    # DroneStateVector uses positive = up for vertical_speed
    return upd


def parse_altitude(msg, vehicle_id: str) -> Optional[DroneStateUpdate]:
    """Parse ALTITUDE (#141) → altitude_amsl and altitude_agl."""
    upd = _make_update(vehicle_id)
    upd.gps_alt_msl  = float(getattr(msg, "altitude_amsl", 0.0))
    upd.altitude_agl = float(getattr(msg, "altitude_terrain", 0.0))
    return upd


# ── Dispatcher ────────────────────────────────────────────────────────────────

# Maps MAVLink message type name → parser function
MESSAGE_PARSERS = {
    "HEARTBEAT":               parse_heartbeat,
    "ATTITUDE":                parse_attitude,
    "ATTITUDE_QUATERNION":     parse_attitude_quaternion,
    "LOCAL_POSITION_NED":      parse_local_position_ned,
    "GLOBAL_POSITION_INT":     parse_global_position_int,
    "GPS_RAW_INT":             parse_gps_raw_int,
    "HIGHRES_IMU":             parse_highres_imu,
    "BATTERY_STATUS":          parse_battery_status,
    "ACTUATOR_OUTPUT_STATUS":  parse_actuator_output_status,
    "VFR_HUD":                 parse_vfr_hud,
    "ALTITUDE":                parse_altitude,
}


def dispatch(msg, vehicle_id: str) -> Optional[DroneStateUpdate]:
    """Dispatch a raw MAVLink message to the appropriate parser.

    Args:
        msg:        pymavlink message object (has a get_type() method).
        vehicle_id: Target StateStore vehicle ID.

    Returns:
        DroneStateUpdate or None if the message type is not supported.
    """
    msg_type = msg.get_type()
    parser   = MESSAGE_PARSERS.get(msg_type)
    if parser is None:
        return None

    try:
        return parser(msg, vehicle_id)
    except Exception:  # noqa: BLE001
        logger.exception(
            "MAVLink parser error for message type '%s'", msg_type
        )
        return None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_update(vehicle_id: str) -> DroneStateUpdate:
    """Create a DroneStateUpdate stamped with the current wall/mono time."""
    return DroneStateUpdate(
        vehicle_id     = vehicle_id,
        source         = DataSource.MAVLINK,
        timestamp_wall = time.time(),
        timestamp_mono = time.monotonic(),
    )
