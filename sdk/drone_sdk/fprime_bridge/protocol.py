"""
drone_sdk.fprime_bridge.protocol
================================
NASA F Prime (F') Flight Software Wire Protocol & Framing.

Implements the standard NASA JPL F Prime framing protocol and channelized
telemetry/command serialization for Software-In-The-Loop (SITL) and
Hardware-In-The-Loop (HIL) flight computer integration.

Frame Layout:
  ┌───────────────┬─────────────┬─────────────┬─────────────────┬──────────┐
  │ Sync Word (4) │ Pkt Type (2)│ Length (2)  │  Payload (N)    │ CRC32 (4)│
  │  0x5A5A5A5A   │  Big-Endian │ Big-Endian  │  Bytes          │ Big-End. │
  └───────────────┴─────────────┴─────────────┴─────────────────┴──────────┘

Packet Types:
  - 0: Fw::ComPacket (Uplink commands to flight computer)
  - 1: Fw::TlmPacket (Downlink channelized telemetry from flight computer)
  - 2: Fw::LogPacket (Event logs from flight software components)
  - 3: Fw::FilePacket (File uplink/downlink)

Python version: 3.9+
"""
from __future__ import annotations

import binascii
import enum
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FPrimePacketType(enum.IntEnum):
    """F Prime packet descriptor types."""
    COMMAND   = 0   # Fw::ComPacket
    TELEMETRY = 1   # Fw::TlmPacket
    LOG_EVENT = 2   # Fw::LogPacket
    FILE      = 3   # Fw::FilePacket


class FPrimeChannelId(enum.IntEnum):
    """Standard NASA F' UAV telemetry channel identifiers."""
    POS_X          = 1001   # NED X position (m) - float32
    POS_Y          = 1002   # NED Y position (m) - float32
    POS_Z          = 1003   # NED Z position (m) - float32
    VEL_X          = 1004   # NED X velocity (m/s) - float32
    VEL_Y          = 1005   # NED Y velocity (m/s) - float32
    VEL_Z          = 1006   # NED Z velocity (m/s) - float32
    ATT_ROLL       = 1007   # Roll angle (rad) - float32
    ATT_PITCH      = 1008   # Pitch angle (rad) - float32
    ATT_YAW        = 1009   # Yaw angle (rad) - float32
    BATTERY_SOC    = 1010   # State of Charge [0..1] - float32
    BATTERY_VOLT   = 1011   # Battery terminal voltage (V) - float32
    HEALTH_STATUS  = 1012   # System health code (uint8)
    ALTITUDE_AGL   = 1013   # Above Ground Level altitude (m) - float32


class FPrimeCommandOpcode(enum.IntEnum):
    """Standard NASA F' UAV command opcodes."""
    NOOP               = 0
    ARM_VEHICLE        = 101
    DISARM_VEHICLE     = 102
    SET_VELOCITY_NED   = 201
    SET_POSITION_NED   = 202
    NAV_WAYPOINT       = 203
    EMERGENCY_LAND     = 301


SYNC_WORD: bytes = b"\x5A\x5A\x5A\x5A"


@dataclass
class FPrimePacket:
    """Decoded NASA F Prime framed packet."""
    packet_type: FPrimePacketType
    payload:     bytes
    timestamp:   float = field(default_factory=time.time)

    def serialize(self) -> bytes:
        """Encode packet into framed binary representation with CRC32."""
        header = struct.pack(">4sHH", SYNC_WORD, int(self.packet_type), len(self.payload))
        data_to_crc = struct.pack(">H", int(self.packet_type)) + struct.pack(">H", len(self.payload)) + self.payload
        crc = binascii.crc32(data_to_crc) & 0xFFFFFFFF
        return header + self.payload + struct.pack(">I", crc)

    @classmethod
    def deserialize(cls, data: bytes) -> Tuple[Optional["FPrimePacket"], bytes]:
        """Attempt to extract one valid framed packet from incoming byte stream.

        Returns (packet, remaining_unconsumed_bytes).
        """
        if len(data) < 12:  # Min header (8) + min CRC (4)
            return None, data

        # Find sync word
        sync_idx = data.find(SYNC_WORD)
        if sync_idx == -1:
            return None, b""

        # Discard leading junk before sync word
        data = data[sync_idx:]
        if len(data) < 12:
            return None, data

        _, pkt_type_raw, length = struct.unpack(">4sHH", data[:8])

        total_frame_len = 8 + length + 4
        if len(data) < total_frame_len:
            # Need more bytes from network
            return None, data

        payload = data[8: 8 + length]
        received_crc = struct.unpack(">I", data[8 + length: total_frame_len])[0]

        # Verify CRC
        data_to_crc = struct.pack(">H", pkt_type_raw) + struct.pack(">H", length) + payload
        expected_crc = binascii.crc32(data_to_crc) & 0xFFFFFFFF

        if received_crc != expected_crc:
            # Bad frame, skip past sync word and continue searching
            return None, data[4:]

        pkt = cls(
            packet_type=FPrimePacketType(pkt_type_raw) if pkt_type_raw in [e.value for e in FPrimePacketType] else FPrimePacketType.TELEMETRY,
            payload=payload,
        )
        remaining = data[total_frame_len:]
        return pkt, remaining


class FPrimeTelemetrySerializer:
    """Encodes and decodes channelized NASA F' telemetry packets."""

    @staticmethod
    def encode_channels(channels: Dict[FPrimeChannelId, Any]) -> bytes:
        """Pack telemetry channels into an F' TlmPacket payload."""
        # Format: Count (uint16), then for each: ChannelId (uint16), Type (uint8: 1=float, 2=uint8), Value
        buf = bytearray()
        buf.extend(struct.pack(">H", len(channels)))
        for ch_id, val in channels.items():
            if isinstance(val, (float, int, np.floating)) and not isinstance(val, bool):
                buf.extend(struct.pack(">HBf", int(ch_id), 1, float(val)))
            elif isinstance(val, int) or isinstance(val, (np.integer, bool)):
                buf.extend(struct.pack(">HBB", int(ch_id), 2, int(val)))
            else:
                buf.extend(struct.pack(">HBf", int(ch_id), 1, float(val)))
        return bytes(buf)

    @staticmethod
    def decode_channels(payload: bytes) -> Dict[FPrimeChannelId, Any]:
        """Unpack an F' TlmPacket payload into channel key-value pairs."""
        if len(payload) < 2:
            return {}
        n_channels = struct.unpack(">H", payload[:2])[0]
        offset = 2
        result: Dict[FPrimeChannelId, Any] = {}

        for _ in range(n_channels):
            if offset + 3 > len(payload):
                break
            ch_id_raw, val_type = struct.unpack(">HB", payload[offset: offset + 3])
            offset += 3
            try:
                ch_id = FPrimeChannelId(ch_id_raw)
            except ValueError:
                ch_id = ch_id_raw  # type: ignore

            if val_type == 1:  # float32
                if offset + 4 > len(payload):
                    break
                val = struct.unpack(">f", payload[offset: offset + 4])[0]
                offset += 4
                result[ch_id] = val
            elif val_type == 2:  # uint8
                if offset + 1 > len(payload):
                    break
                val = struct.unpack(">B", payload[offset: offset + 1])[0]
                offset += 1
                result[ch_id] = val

        return result


class FPrimeCommandSerializer:
    """Encodes and decodes NASA F' uplink command packets."""

    @staticmethod
    def encode_command(opcode: FPrimeCommandOpcode, *args: float) -> bytes:
        """Pack an opcode and float arguments into an F' ComPacket payload."""
        buf = bytearray()
        buf.extend(struct.pack(">HH", int(opcode), len(args)))
        for a in args:
            buf.extend(struct.pack(">f", float(a)))
        return bytes(buf)

    @staticmethod
    def decode_command(payload: bytes) -> Tuple[Optional[FPrimeCommandOpcode], List[float]]:
        """Unpack an F' ComPacket payload into opcode and arguments."""
        if len(payload) < 4:
            return None, []
        opcode_raw, n_args = struct.unpack(">HH", payload[:4])
        offset = 4
        args = []
        for _ in range(n_args):
            if offset + 4 > len(payload):
                break
            val = struct.unpack(">f", payload[offset: offset + 4])[0]
            args.append(val)
            offset += 4

        try:
            opcode = FPrimeCommandOpcode(opcode_raw)
        except ValueError:
            opcode = None

        return opcode, args
