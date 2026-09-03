"""
drone_sdk.fprime_bridge
=======================
NASA F Prime (F') Inspired Flight Software Protocol & SITL Bridge.

Note: This module provides a lightweight, clean-room protocol and SITL/HIL
bridge inspired by NASA JPL's F Prime (F') component-based flight software
architecture (using F'-style packet type identifiers, channelized telemetry,
and command opcodes with 0x5A5A5A5A sync framing and CRC32 verification).
It is designed for rapid digital twin validation and software-in-the-loop
simulation; it is not a direct wire-compatible drop-in for compiled C++ FPP /
fprime-gds deployments.

Architecture:
  - protocol: F'-inspired framing (0x5A5A5A5A, packet types, CRC32, channel serialization).
  - bridge: Bidirectional TCP/UDP and loopback socket stream engine (FPrimeSITLBridge).
  - source: TelemetryEngine / SourceArbiter adapter (FPrimeTelemetrySource).

Python version: 3.9+
"""
from __future__ import annotations

from .protocol import (
    FPrimePacket,
    FPrimePacketType,
    FPrimeChannelId,
    FPrimeCommandOpcode,
    FPrimeTelemetrySerializer,
    FPrimeCommandSerializer,
    SYNC_WORD,
)
from .bridge import FPrimeSITLBridge
from .source import FPrimeTelemetrySource

__all__ = [
    "FPrimePacket",
    "FPrimePacketType",
    "FPrimeChannelId",
    "FPrimeCommandOpcode",
    "FPrimeTelemetrySerializer",
    "FPrimeCommandSerializer",
    "FPrimeSITLBridge",
    "FPrimeTelemetrySource",
    "SYNC_WORD",
]
