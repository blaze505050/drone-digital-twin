"""
dronepy.inputs
==============
Laptop Controller and Human-Machine Interface for DronePy.
Maps keyboard, virtual joysticks, or gamepad inputs to CommandIntent objects,
routing all commands through the SafetyGateway before dispatch.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from drone_sdk.dronepy.safety import CommandIntent, SafetyGateway, ValidationReport


@dataclass
class ControllerState:
    """Normalized controller axis and button state in range [-1.0, 1.0]."""
    roll: float = 0.0       # Left (-1) to Right (+1)
    pitch: float = 0.0      # Forward (+1) to Backward (-1)
    yaw: float = 0.0        # CCW (-1) to CW (+1)
    throttle: float = 0.0   # 0.0 to 1.0 (climb / descent)
    arm: bool = False
    disarm: bool = False
    mode: str = "VELOCITY"  # VELOCITY, POSITION, HOVER, LAND, RTL


class LaptopController:
    """
    Translates local laptop inputs (keyboard, gamepad, or synthetic script)
    into validated CommandIntent objects.
    """

    def __init__(
        self,
        safety_gateway: Optional[SafetyGateway] = None,
        max_speed_m_s: float = 5.0,
        max_yaw_rate_rad_s: float = np.radians(60.0),
    ) -> None:
        self.safety_gateway = safety_gateway or SafetyGateway()
        self.max_speed_m_s = max_speed_m_s
        self.max_yaw_rate_rad_s = max_yaw_rate_rad_s
        self.state = ControllerState()

    def update_from_keyboard(self, key_states: Dict[str, bool]) -> None:
        """
        Updates controller state from boolean dictionary of pressed keys.
        Mapping:
          'w': Pitch forward (+X)
          's': Pitch backward (-X)
          'a': Roll left (-Y)
          'd': Roll right (+Y)
          'q': Yaw counter-clockwise
          'e': Yaw clockwise
          'shift': Climb (increase throttle / -Z vel)
          'ctrl': Descend (decrease throttle / +Z vel)
          'space': Immediate DISARM
          'h': Return to Launch / Hover
        """
        pitch = 0.0
        roll = 0.0
        yaw = 0.0
        climb = 0.0

        if key_states.get("w", False):
            pitch += 1.0
        if key_states.get("s", False):
            pitch -= 1.0
        if key_states.get("d", False):
            roll += 1.0
        if key_states.get("a", False):
            roll -= 1.0
        if key_states.get("e", False):
            yaw += 1.0
        if key_states.get("q", False):
            yaw -= 1.0
        if key_states.get("shift", False):
            climb += 1.0
        if key_states.get("ctrl", False):
            climb -= 1.0

        self.state.pitch = float(pitch)
        self.state.roll = float(roll)
        self.state.yaw = float(yaw)
        self.state.throttle = float(climb)

        if key_states.get("space", False):
            self.state.disarm = True
            self.state.arm = False
        else:
            self.state.disarm = False

    def generate_intent(
        self,
        current_pos_ned: np.ndarray,
        current_yaw_rad: float = 0.0,
    ) -> CommandIntent:
        """
        Converts internal controller stick states to an inertial NED CommandIntent.
        """
        now = time.time()

        if self.state.disarm:
            return CommandIntent(
                timestamp=now,
                mode="DISARM",
                disarm_request=True,
            )

        if self.state.arm:
            return CommandIntent(
                timestamp=now,
                mode="ARM",
                arm_request=True,
            )

        # Convert forward/right stick commands to North/East velocities based on yaw
        cos_y = np.cos(current_yaw_rad)
        sin_y = np.sin(current_yaw_rad)

        v_forward = self.state.pitch * self.max_speed_m_s
        v_right = self.state.roll * self.max_speed_m_s

        v_north = cos_y * v_forward - sin_y * v_right
        v_east = sin_y * v_forward + cos_y * v_right
        v_down = -self.state.throttle * (self.max_speed_m_s * 0.5)  # Climbing is -Z

        vel_ned = np.array([v_north, v_east, v_down])

        return CommandIntent(
            timestamp=now,
            mode=self.state.mode,
            target_velocity_ned=vel_ned,
            throttle_norm=0.5 + 0.5 * self.state.throttle,
        )

    def process_and_validate(
        self,
        current_pos_ned: np.ndarray,
        current_vel_ned: np.ndarray,
        current_euler_rad: np.ndarray,
    ) -> ValidationReport:
        """
        Generates command intent and immediately validates it through the safety gateway.
        """
        intent = self.generate_intent(current_pos_ned, current_yaw_rad=current_euler_rad[2])
        return self.safety_gateway.validate_command(
            intent=intent,
            current_pos_ned=current_pos_ned,
            current_vel_ned=current_vel_ned,
            current_euler_rad=current_euler_rad,
        )
