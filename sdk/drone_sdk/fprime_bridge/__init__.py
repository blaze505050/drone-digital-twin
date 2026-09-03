"""
drone_sdk.fprime_bridge
=======================
NASA F Prime (F') Flight Software SITL & HIL Integration Bridge.

Architecture:
  - protocol: Wire framing (0x5A5A5A5A, packet types, CRC32, channel serialization).
  - bridge: Bidirectional TCP/UDP and loopback socket stream engine (FPrimeSITLBridge).
  - source: TelemetryEngine / SourceArbiter adapter (FPrimeTelemetrySource).

Enables seamless software-in-the-loop and hardware-in-the-loop flight control
using NASA JPL's component-based flight software framework.

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
