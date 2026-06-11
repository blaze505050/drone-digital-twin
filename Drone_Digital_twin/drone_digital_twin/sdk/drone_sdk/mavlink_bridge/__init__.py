"""
drone_sdk.mavlink_bridge
========================
MAVLink / PX4 Integration — Module 3 of the UAV Digital Twin Platform.

This module connects the Digital Twin to any MAVLink-speaking autopilot:
PX4 SITL (software simulation), PX4 on real hardware (Pixhawk, Cube,
Kakute), ArduPilot, or any other MAVLink-compliant flight controller.

Architecture::

    PX4 SITL / Pixhawk
          │ MAVLink UDP / Serial
          ▼
    MAVLinkSource.poll()
          │ DroneStateUpdate (partial)
          ▼
    TelemetryEngine → SourceArbiter
          │
          ▼
    StateStore.update_partial()
          │
          ▼
    Dashboard / AI / ROS2 Bridge

Public API
----------
Source::

    MAVLinkSource   # TelemetrySource for live MAVLink streams

Parsers::

    dispatch            # Route a raw MAVLink message to the correct parser
    parse_heartbeat     # → flight_mode, arming_state
    parse_attitude      # → roll, pitch, yaw, angular rates
    parse_attitude_quaternion   # → quaternion, angular rates
    parse_local_position_ned    # → position, velocity NED
    parse_global_position_int   # → GPS lat/lon/alt
    parse_gps_raw_int           # → GPS fix quality
    parse_highres_imu           # → body accelerations
    parse_battery_status        # → voltage, current, SoC
    parse_actuator_output_status # → rotor speeds
    parse_vfr_hud               # → airspeed, groundspeed
    parse_altitude              # → altitude_amsl, altitude_agl
    MESSAGE_PARSERS             # Dict of all registered parsers

Utilities::

    SITLLauncher    # Start / stop PX4 SITL process (Docker or native)
    ConnectionPreset # Standard connection strings

Quick-start — PX4 SITL
-----------------------
::

    from drone_sdk.state_manager import StateStore
    from drone_sdk.telemetry_engine import TelemetryEngine
    from drone_sdk.mavlink_bridge import MAVLinkSource, SourcePriority

    # 1. State store
    StateStore.create("drone_0")

    # 2. MAVLink source pointing at PX4 SITL default port
    source = MAVLinkSource(
        source_id      = "px4_sitl",
        vehicle_id     = "drone_0",
        connection_str = "udp:127.0.0.1:14550",
        priority       = SourcePriority.SITL,
    )
    source.connect()
    source.configure_default_streams(rate_hz=50)

    # 3. Telemetry engine
    engine = TelemetryEngine()
    engine.add_source(source)
    engine.start()

    # 4. Read state
    import time; time.sleep(2)
    state = StateStore.get_instance("drone_0").get_latest()
    print("Flight mode:", state.flight_mode.name)
    print("Armed:", state.is_armed)
    print("Position NED:", state.position_ned())

    engine.stop()
"""

from .mavlink_source import MAVLinkSource
from .parsers import (
    MESSAGE_PARSERS,
    dispatch,
    parse_actuator_output_status,
    parse_altitude,
    parse_attitude,
    parse_attitude_quaternion,
    parse_battery_status,
    parse_global_position_int,
    parse_gps_raw_int,
    parse_heartbeat,
    parse_highres_imu,
    parse_local_position_ned,
    parse_vfr_hud,
)
from drone_sdk.telemetry_engine.source import SourcePriority

# Convenience connection strings for common setups
class ConnectionPreset:
    """Standard MAVLink connection strings."""
    PX4_SITL_UDP    = "udp:127.0.0.1:14550"   # PX4 SITL default
    PX4_SITL_UDP_2  = "udp:127.0.0.1:14560"   # Second SITL instance
    MAVPROXY_TCP    = "tcp:127.0.0.1:5760"     # MAVProxy TCP bridge
    PIXHAWK_USB     = "/dev/ttyACM0:57600"     # Pixhawk USB (Linux)
    PIXHAWK_USB_MAC = "/dev/tty.usbmodem*"     # Pixhawk USB (macOS)
    TELEMETRY_RADIO = "/dev/ttyUSB0:57600"     # SiK telemetry radio
    COMPANION_UDP   = "udp:192.168.1.1:14550"  # Companion computer LAN


__version__ = "1.0.0"

__all__ = [
    "MAVLinkSource",
    "ConnectionPreset",
    "SourcePriority",
    "dispatch",
    "MESSAGE_PARSERS",
    "parse_heartbeat",
    "parse_attitude",
    "parse_attitude_quaternion",
    "parse_local_position_ned",
    "parse_global_position_int",
    "parse_gps_raw_int",
    "parse_highres_imu",
    "parse_battery_status",
    "parse_actuator_output_status",
    "parse_vfr_hud",
    "parse_altitude",
]
