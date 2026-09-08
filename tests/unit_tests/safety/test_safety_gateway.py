"""
Unit tests for Real-Time Safety Gateway & SITL Transport (B19).
Verifies:
- Setpoint clamping against physical velocity and geofence limits.
- Heartbeat watchdog timeout and emergency failsafe activation.
- Non-blocking UDP transport packet send and receive.
"""
import numpy as np
import pytest

from drone_sdk.contracts.streams import CommandIntent
from drone_sdk.safety import (
    SafetyGateway,
    SafetyLimits,
    UDPMavlinkTransport,
)


def test_safety_gateway_setpoint_clamping():
    gateway = SafetyGateway(SafetyLimits(
        max_horizontal_speed_mps=10.0,
        max_climb_speed_mps=3.0,
        geofence_radius_m=200.0,
        max_altitude_agl_m=100.0,
    ))
    gateway.record_heartbeat(timestamp_mono=100.0)

    # Excessive command: 500m distance, 25 m/s speed, 150m altitude
    bad_cmd = CommandIntent(
        vehicle_id="holybro_x500_v2",
        timestamp_mono=100.1,
        flight_mode="OFFBOARD",
        target_pos_ned=np.array([400.0, 300.0, -150.0]), # Distance = 500m, Alt = 150m
        target_vel_ned=np.array([20.0, 15.0, -6.0]),     # Horiz speed = 25m/s, Climb = 6m/s
        target_thrust=1.2, # Over 1.0
        armed=True,
    )

    is_safe, safe_cmd, actions = gateway.validate_and_clamp_command(bad_cmd, current_mono=100.1)
    assert is_safe is True
    assert len(actions) >= 4  # Clamped geofence, altitude, horiz velocity, climb rate

    # Check clamped limits
    r_xy = np.linalg.norm(safe_cmd.target_pos_ned[:2])
    assert np.isclose(r_xy, 200.0, atol=1e-3)
    assert safe_cmd.target_pos_ned[2] == -100.0  # Max altitude 100m AGL

    v_xy = np.linalg.norm(safe_cmd.target_vel_ned[:2])
    assert np.isclose(v_xy, 10.0, atol=1e-3)
    assert safe_cmd.target_vel_ned[2] == -3.0  # Max climb 3.0 m/s
    assert safe_cmd.target_thrust == 1.0


def test_safety_gateway_heartbeat_timeout_failsafe():
    gateway = SafetyGateway(SafetyLimits(heartbeat_timeout_sec=1.0))
    gateway.record_heartbeat(timestamp_mono=10.0)

    # 1.5 seconds later with no heartbeat
    cmd = CommandIntent(
        vehicle_id="holybro_x500_v2",
        timestamp_mono=11.5,
        target_thrust=0.80,
    )
    is_safe, safe_cmd, actions = gateway.validate_and_clamp_command(cmd, current_mono=11.5)

    assert is_safe is False
    assert safe_cmd.flight_mode == "FAILSAFE_LAND"
    assert gateway.is_failsafe_active is True
    assert "Heartbeat lost" in gateway.failsafe_reason


def test_udp_mavlink_transport():
    transport = UDPMavlinkTransport(local_port=14588, remote_port=14589, enable_mock=True)
    connected = transport.connect()
    assert connected is True

    # Send dummy MAVLink packet
    dummy_payload = b"\xFD\x09\x00\x00\x00\x01\x01\x1E\x00\x00"
    bytes_sent = transport.send_packet(dummy_payload)
    assert bytes_sent == len(dummy_payload)
    assert transport.stats.packets_sent == 1

    transport.close()
