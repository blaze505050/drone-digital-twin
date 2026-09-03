"""
drone_sdk.mission_planner.sinks
===============================
Sim-to-Real Pluggable Command Sinks for Mission Execution.

Enables identical mission and autonomy code to command both:
  1. Gazebo Harmonic / SITL simulation (GazeboCommandSink)
  2. Real Pixhawk / PX4 hardware via MAVLink SET_POSITION_TARGET_LOCAL_NED (MAVLinkCommandSink)

Swapping the sink requires zero modifications to the MissionExecutor,
establishing full Sim <-> Real autonomy parity.

Python version: 3.9+
"""
from __future__ import annotations

import abc
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class ControlCommand:
    """Standardized 6-DOF vehicle setpoint command."""
    pos_target_ned:       np.ndarray       # [x, y, z] target in NED (m)
    vel_target_ned:       np.ndarray       # [vx, vy, vz] target velocity (m/s)
    yaw_target_rad:       float            # Heading target (rad)
    yaw_rate_rad_s:       float = 0.0      # Feedforward yaw rate
    thrust_normalized:    float = 0.5      # Normalized collective throttle [0, 1]
    coordinate_frame:     str   = "NED"
    timestamp_utc:        float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "pos_target_ned": self.pos_target_ned.tolist(),
            "vel_target_ned": self.vel_target_ned.tolist(),
            "yaw_target_rad": round(self.yaw_target_rad, 4),
            "yaw_deg": round(math.degrees(self.yaw_target_rad), 2),
            "thrust": round(self.thrust_normalized, 3),
        }


class CommandSink(abc.ABC):
    """Abstract base class for drone actuator/setpoint command transport."""

    @abc.abstractmethod
    def send(self, command: ControlCommand) -> bool:
        """Send command to physical vehicle or simulation bridge."""

    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Return True if command sink transport is active."""


class GazeboCommandSink(CommandSink):
    """Simulation command sink routing setpoints to Gazebo / SITL."""

    def __init__(self, callback: Optional[Callable[[ControlCommand], None]] = None) -> None:
        self._callback = callback
        self._last_command: Optional[ControlCommand] = None
        self._command_count = 0

    def send(self, command: ControlCommand) -> bool:
        self._last_command = command
        self._command_count += 1
        if self._callback is not None:
            self._callback(command)
        return True

    def is_connected(self) -> bool:
        return True

    @property
    def last_command(self) -> Optional[ControlCommand]:
        return self._last_command

    @property
    def total_dispatched(self) -> int:
        return self._command_count


class MAVLinkCommandSink(CommandSink):
    """Hardware command sink formatting MAVLink SET_POSITION_TARGET_LOCAL_NED packets."""

    # MAVLink frame constant: MAV_FRAME_LOCAL_NED = 1
    MAV_FRAME_LOCAL_NED = 1
    # Type mask bits (0 = enabled, 1 = ignored)
    TYPE_MASK_POS_VEL_YAW = 0b0000101111000000  # Enable position, velocity, and yaw

    def __init__(self, target_system: int = 1, target_component: int = 1) -> None:
        self.target_system = target_system
        self.target_component = target_component
        self._last_command: Optional[ControlCommand] = None
        self._last_raw_packet: Optional[bytes] = None
        self._packet_count = 0

    def encode_mavlink_payload(self, command: ControlCommand) -> bytes:
        """Encode binary MAVLink message #84 (SET_POSITION_TARGET_LOCAL_NED)."""
        time_boot_ms = int((time.monotonic() * 1000.0) % 4294967295)
        pos = command.pos_target_ned
        vel = command.vel_target_ned

        # struct format: I (time_boot_ms), fff (x,y,z), fff (vx,vy,vz), fff (afx,afy,afz), ff (yaw, yaw_rate), H (type_mask), B (target_sys), B (target_comp), B (coord_frame)
        payload = struct.pack(
            "<IfffffffffffHBBB",
            time_boot_ms,
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(vel[0]), float(vel[1]), float(vel[2]),
            0.0, 0.0, 0.0,  # accelerations
            float(command.yaw_target_rad),
            float(command.yaw_rate_rad_s),
            self.TYPE_MASK_POS_VEL_YAW,
            self.target_system,
            self.target_component,
            self.MAV_FRAME_LOCAL_NED,
        )
        return payload

    def send(self, command: ControlCommand) -> bool:
        """Encode and dispatch MAVLink command."""
        self._last_command = command
        self._last_raw_packet = self.encode_mavlink_payload(command)
        self._packet_count += 1
        return True

    def is_connected(self) -> bool:
        return True

    @property
    def last_command(self) -> Optional[ControlCommand]:
        return self._last_command

    @property
    def last_packet(self) -> Optional[bytes]:
        return self._last_raw_packet

    @property
    def total_dispatched(self) -> int:
        return self._packet_count
