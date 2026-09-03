"""Tests for NASA F Prime (F') SITL flight computer bridge and wire protocol."""
import struct
import numpy as np
import pytest

from drone_sdk.fprime_bridge import (
    FPrimePacket,
    FPrimePacketType,
    FPrimeChannelId,
    FPrimeCommandOpcode,
    FPrimeTelemetrySerializer,
    FPrimeCommandSerializer,
    FPrimeSITLBridge,
    FPrimeTelemetrySource,
    SYNC_WORD,
)
from drone_sdk.mission_planner.sinks import ControlCommand
from drone_sdk.state_manager import DataSource, StateFactory
from drone_sdk.telemetry_engine.source import SourcePriority, SourceStatus


def test_fprime_packet_serialization_and_crc():
    payload = b"NASA_JPL_FPRIME_TELEMETRY"
    pkt = FPrimePacket(packet_type=FPrimePacketType.TELEMETRY, payload=payload)
    serialized = pkt.serialize()

    # Sync word
    assert serialized[:4] == SYNC_WORD
    # Minimum length: header (8) + payload (25) + crc (4) = 37 bytes
    assert len(serialized) == 8 + len(payload) + 4

    # Valid deserialization
    decoded, remaining = FPrimePacket.deserialize(serialized)
    assert decoded is not None
    assert decoded.packet_type == FPrimePacketType.TELEMETRY
    assert decoded.payload == payload
    assert len(remaining) == 0


def test_fprime_packet_corrupted_crc():
    payload = b"TEST_CORRUPTION"
    pkt = FPrimePacket(packet_type=FPrimePacketType.COMMAND, payload=payload)
    raw = bytearray(pkt.serialize())

    # Corrupt a byte in payload
    raw[10] ^= 0xFF

    decoded, remaining = FPrimePacket.deserialize(bytes(raw))
    assert decoded is None  # Should reject corrupted frame


def test_fprime_telemetry_channel_encoding():
    channels = {
        FPrimeChannelId.POS_X:        12.5,
        FPrimeChannelId.POS_Y:        -4.2,
        FPrimeChannelId.POS_Z:        -15.0,
        FPrimeChannelId.BATTERY_SOC:  0.88,
        FPrimeChannelId.HEALTH_STATUS: 1,
    }
    encoded = FPrimeTelemetrySerializer.encode_channels(channels)
    decoded = FPrimeTelemetrySerializer.decode_channels(encoded)

    assert len(decoded) == 5
    assert abs(decoded[FPrimeChannelId.POS_X] - 12.5) < 1e-4
    assert abs(decoded[FPrimeChannelId.POS_Y] - (-4.2)) < 1e-4
    assert abs(decoded[FPrimeChannelId.POS_Z] - (-15.0)) < 1e-4
    assert abs(decoded[FPrimeChannelId.BATTERY_SOC] - 0.88) < 1e-4
    assert decoded[FPrimeChannelId.HEALTH_STATUS] == 1


def test_fprime_command_encoding():
    encoded = FPrimeCommandSerializer.encode_command(
        FPrimeCommandOpcode.SET_POSITION_NED, 10.0, 5.0, -20.0, 1.57
    )
    opcode, args = FPrimeCommandSerializer.decode_command(encoded)

    assert opcode == FPrimeCommandOpcode.SET_POSITION_NED
    assert len(args) == 4
    assert abs(args[0] - 10.0) < 1e-4
    assert abs(args[1] - 5.0) < 1e-4
    assert abs(args[2] - (-20.0)) < 1e-4
    assert abs(args[3] - 1.57) < 1e-4


def test_fprime_sitl_bridge_telemetry():
    bridge = FPrimeSITLBridge(vehicle_id="fprime_uav", use_loopback=True)
    bridge.start()

    state = StateFactory.create_initial("fprime_uav").copy_with(x=5.0, y=2.0, z=-12.0)
    assert bridge.send_telemetry(state)
    assert bridge.packets_sent == 1
    assert bridge.bytes_sent > 30

    bridge.stop()


def test_fprime_sitl_bridge_command_loopback():
    bridge = FPrimeSITLBridge(vehicle_id="fprime_uav", use_loopback=True)
    bridge.start()

    received_commands = []
    bridge.register_command_callback(lambda cmd: received_commands.append(cmd))

    # Simulate F' flight computer issuing position setpoint
    bridge.send_simulated_fprime_command(
        FPrimeCommandOpcode.SET_POSITION_NED, 15.0, 8.0, -25.0, 0.5
    )

    assert bridge.packets_received == 1
    assert len(received_commands) == 1
    cmd = received_commands[0]
    assert isinstance(cmd, ControlCommand)
    assert cmd.pos_target_ned[0] == 15.0
    assert cmd.pos_target_ned[1] == 8.0
    assert cmd.pos_target_ned[2] == -25.0
    assert abs(cmd.yaw_target_rad - 0.5) < 1e-4

    polled = bridge.poll_commands()
    assert len(polled) == 1
    # Second poll should be empty (drained)
    assert len(bridge.poll_commands()) == 0

    bridge.stop()


def test_fprime_telemetry_source():
    source = FPrimeTelemetrySource(vehicle_id="fprime_uav", priority=SourcePriority.HARDWARE)
    assert source.priority == SourcePriority.HARDWARE
    assert source.data_source == DataSource.HIL

    source.connect()
    assert source.status == SourceStatus.CONNECTED

    # Inject command from simulated F' flight computer
    source.bridge.send_simulated_fprime_command(
        FPrimeCommandOpcode.SET_VELOCITY_NED, 3.0, 0.0, 0.0, 0.0
    )

    updates = source.poll()
    assert len(updates) == 1
    upd = updates[0]
    assert upd.source == DataSource.HIL
    assert upd.velocity is not None
    assert upd.velocity[0] == 3.0

    source.disconnect()
    assert source.status == SourceStatus.DISCONNECTED
