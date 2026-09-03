"""
drone_sdk.fprime_bridge.bridge
==============================
NASA F Prime (F') SITL Flight Computer Socket Bridge.

Provides a bidirectional bridge between the UAV Digital Twin and NASA JPL's
F Prime flight software running in Software-In-The-Loop (SITL) mode:
  1. Uplinks high-rate sensor telemetry from the Digital Twin into F' components
     (e.g., GncComponent, NavComponent) via framed Fw::TlmPacket streams.
  2. Downlinks flight computer commands from F' components (CmdDispatcher)
     into the Digital Twin's mission planner and actuator dynamics.
  3. Supports both live network sockets (UDP/TCP) and deterministic headless
     in-memory loopback for test suites and continuous integration.

Python version: 3.9+
"""
from __future__ import annotations

import collections
import math
import socket
import threading
import time
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from drone_sdk.fprime_bridge.protocol import (
    FPrimeChannelId,
    FPrimeCommandOpcode,
    FPrimeCommandSerializer,
    FPrimePacket,
    FPrimePacketType,
    FPrimeTelemetrySerializer,
)
from drone_sdk.mission_planner.sinks import ControlCommand
from drone_sdk.state_manager.schema import DroneStateVector


class FPrimeSITLBridge:
    """NASA F' SITL Flight Computer communication bridge."""

    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 50050

    def __init__(
        self,
        vehicle_id: str = "fprime_drone",
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        use_loopback: bool = True,
    ) -> None:
        self.vehicle_id = vehicle_id
        self.host = host
        self.port = port
        self.use_loopback = use_loopback

        self._rx_buffer = bytearray()
        self._pending_commands: Deque[ControlCommand] = collections.deque(maxlen=100)
        self._command_callbacks: List[Callable[[ControlCommand], None]] = []

        # Metrics
        self.packets_sent = 0
        self.packets_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0

        self._sock: Optional[socket.socket] = None
        self._running = False
        self._lock = threading.RLock()

    def start(self) -> bool:
        """Start socket listener if not operating in in-memory loopback mode."""
        self._running = True
        if not self.use_loopback:
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._sock.bind((self.host, self.port))
                self._sock.setblocking(False)
            except Exception:
                self._sock = None
                return False
        return True

    def stop(self) -> None:
        """Close socket and stop bridge."""
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def register_command_callback(self, callback: Callable[[ControlCommand], None]) -> None:
        """Register a callback fired whenever F' issues a flight control command."""
        self._command_callbacks.append(callback)

    # ── Telemetry Transmission ────────────────────────────────────────────────

    def pack_telemetry(self, state: DroneStateVector) -> bytes:
        """Serialize a DroneStateVector into a framed NASA F' TlmPacket."""
        channels: Dict[FPrimeChannelId, float] = {
            FPrimeChannelId.POS_X:         float(state.x),
            FPrimeChannelId.POS_Y:         float(state.y),
            FPrimeChannelId.POS_Z:         float(state.z),
            FPrimeChannelId.VEL_X:         float(state.vx),
            FPrimeChannelId.VEL_Y:         float(state.vy),
            FPrimeChannelId.VEL_Z:         float(state.vz),
            FPrimeChannelId.ATT_ROLL:      float(state.roll),
            FPrimeChannelId.ATT_PITCH:     float(state.pitch),
            FPrimeChannelId.ATT_YAW:       float(state.yaw),
            FPrimeChannelId.BATTERY_SOC:   float(state.battery_soc),
            FPrimeChannelId.BATTERY_VOLT:  float(state.battery_voltage),
            FPrimeChannelId.ALTITUDE_AGL:  float(state.altitude_agl),
        }
        payload = FPrimeTelemetrySerializer.encode_channels(channels)
        pkt = FPrimePacket(packet_type=FPrimePacketType.TELEMETRY, payload=payload)
        return pkt.serialize()

    def send_telemetry(self, state: DroneStateVector) -> bool:
        """Send framed telemetry to F' flight software."""
        framed_bytes = self.pack_telemetry(state)
        with self._lock:
            self.packets_sent += 1
            self.bytes_sent += len(framed_bytes)

        if not self.use_loopback and self._sock:
            try:
                self._sock.sendto(framed_bytes, (self.host, self.port))
                return True
            except Exception:
                return False
        return True

    # ── Command Ingestion & Processing ────────────────────────────────────────

    def inject_raw_bytes(self, data: bytes) -> None:
        """Feed incoming network bytes into the F' stream decoder."""
        with self._lock:
            self._rx_buffer.extend(data)
            self.bytes_received += len(data)

            while True:
                pkt, remaining = FPrimePacket.deserialize(bytes(self._rx_buffer))
                if pkt is None:
                    break
                self._rx_buffer = bytearray(remaining)
                self.packets_received += 1
                self._handle_packet(pkt)

    def _handle_packet(self, pkt: FPrimePacket) -> None:
        """Route parsed packet by type."""
        if pkt.packet_type == FPrimePacketType.COMMAND:
            opcode, args = FPrimeCommandSerializer.decode_command(pkt.payload)
            cmd = self._translate_fprime_command(opcode, args)
            if cmd is not None:
                self._pending_commands.append(cmd)
                for cb in self._command_callbacks:
                    try:
                        cb(cmd)
                    except Exception:
                        pass

    def _translate_fprime_command(
        self,
        opcode: Optional[FPrimeCommandOpcode],
        args: List[float],
    ) -> Optional[ControlCommand]:
        """Translate NASA F' command opcode into a standardized ControlCommand."""
        if opcode == FPrimeCommandOpcode.SET_POSITION_NED and len(args) >= 3:
            return ControlCommand(
                pos_target_ned=np.array([args[0], args[1], args[2]]),
                vel_target_ned=np.zeros(3),
                yaw_target_rad=args[3] if len(args) > 3 else 0.0,
                thrust_normalized=0.55,
            )
        elif opcode == FPrimeCommandOpcode.SET_VELOCITY_NED and len(args) >= 3:
            return ControlCommand(
                pos_target_ned=np.zeros(3),
                vel_target_ned=np.array([args[0], args[1], args[2]]),
                yaw_target_rad=args[3] if len(args) > 3 else 0.0,
                thrust_normalized=0.55,
            )
        elif opcode == FPrimeCommandOpcode.NAV_WAYPOINT and len(args) >= 3:
            return ControlCommand(
                pos_target_ned=np.array([args[0], args[1], args[2]]),
                vel_target_ned=np.array([2.0, 0.0, 0.0]),
                yaw_target_rad=args[3] if len(args) > 3 else 0.0,
                thrust_normalized=0.60,
            )
        elif opcode == FPrimeCommandOpcode.EMERGENCY_LAND:
            return ControlCommand(
                pos_target_ned=np.array([0.0, 0.0, 0.0]),
                vel_target_ned=np.array([0.0, 0.0, 1.0]),  # Descend
                yaw_target_rad=0.0,
                thrust_normalized=0.30,
            )
        return None

    def poll_commands(self) -> List[ControlCommand]:
        """Retrieve and drain all pending commands received from F'."""
        with self._lock:
            cmds = list(self._pending_commands)
            self._pending_commands.clear()
            return cmds

    def send_simulated_fprime_command(self, opcode: FPrimeCommandOpcode, *args: float) -> bytes:
        """Simulate F' flight computer issuing an uplink command to the vehicle."""
        payload = FPrimeCommandSerializer.encode_command(opcode, *args)
        pkt = FPrimePacket(packet_type=FPrimePacketType.COMMAND, payload=payload)
        framed = pkt.serialize()
        self.inject_raw_bytes(framed)
        return framed

    @property
    def is_active(self) -> bool:
        return self._running
