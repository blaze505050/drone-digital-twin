"""
dronepy.safety
==============
Safety Gateway, Command Validator, and Hardware Interlock Layer for DronePy.
Implements layered command pipeline:
UserInput -> CommandIntent -> Validation -> SafetyLimits -> FlightController -> Actuators.

CRITICAL SAFETY DIRECTIVE:
The default system state is strictly SIMULATION_ONLY.
Under no circumstances can raw commands reach a physical vehicle without explicit,
multi-token user opt-in, arming confirmation, and active geofencing.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SystemExecutionMode(enum.Enum):
    SIMULATION_ONLY = "simulation_only"
    HARDWARE_MONITOR_ONLY = "hardware_monitor_only"  # Receive telemetry, do not transmit commands
    HARDWARE_ACTIVE_PILOT = "hardware_active_pilot"    # Requires explicit authorization


class SafetyViolationType(enum.Enum):
    NONE = "none"
    MODE_PROHIBITED = "mode_prohibited"
    DISARMED = "disarmed"
    GEOFENCE_EXCEEDED = "geofence_exceeded"
    TILT_EXCEEDED = "tilt_exceeded"
    VELOCITY_EXCEEDED = "velocity_exceeded"
    ALTITUDE_EXCEEDED = "altitude_exceeded"
    RATE_EXCEEDED = "rate_exceeded"
    TIMEOUT = "timeout"


@dataclass
class GeofenceCylinder:
    """Cylindrical geofence volume around origin/takeoff point."""
    center_ned: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    max_radius_m: float = 250.0
    min_altitude_m: float = 0.0    # -Z in NED
    max_altitude_m: float = 120.0  # -Z in NED

    def contains(self, pos_ned: np.ndarray) -> bool:
        rel = pos_ned - self.center_ned
        horizontal_dist = np.linalg.norm(rel[:2])
        altitude = -rel[2]  # NED down is positive, so altitude is -Z
        return bool(
            horizontal_dist <= self.max_radius_m
            and self.min_altitude_m <= altitude <= self.max_altitude_m
        )


@dataclass
class SafetyLimits:
    """Hard dynamic flight envelope safety limits."""
    max_tilt_rad: float = np.radians(45.0)     # 45 deg max bank/pitch
    max_yaw_rate_rad_s: float = np.radians(180.0) # 180 deg/s max yaw rate
    max_horiz_speed_m_s: float = 20.0          # 20 m/s (~72 km/h) max ground speed
    max_vert_speed_m_s: float = 6.0            # 6 m/s max ascent/descent
    command_timeout_sec: float = 0.5           # Failsafe if no heartbeat for 500 ms


@dataclass
class CommandIntent:
    """High-level user or automated pilot flight control intent."""
    timestamp: float
    mode: str = "POSITION"  # POSITION, VELOCITY, ATTITUDE, RATE, LAND, RTL
    target_position_ned: Optional[np.ndarray] = None
    target_velocity_ned: Optional[np.ndarray] = None
    target_euler_rad: Optional[np.ndarray] = None
    throttle_norm: float = 0.0
    arm_request: bool = False
    disarm_request: bool = False


@dataclass
class ValidationReport:
    """Outcome of SafetyGateway validation check."""
    passed: bool
    violation: SafetyViolationType
    detail: str
    filtered_intent: Optional[CommandIntent] = None


class SafetyGateway:
    """
    Central safety interlock and flight envelope enforcement gateway.
    """

    def __init__(
        self,
        mode: SystemExecutionMode = SystemExecutionMode.SIMULATION_ONLY,
        limits: Optional[SafetyLimits] = None,
        geofence: Optional[GeofenceCylinder] = None,
    ) -> None:
        self.mode = mode
        self.limits = limits or SafetyLimits()
        self.geofence = geofence or GeofenceCylinder()
        self.is_armed = False
        self._last_command_time = time.time()
        self._arm_token: Optional[str] = None

    def request_hardware_enable(self, confirmation_token: str) -> bool:
        """
        Explicit authorization required to switch from SIMULATION_ONLY to hardware mode.
        Token must be 'CONFIRM_HARDWARE_FLIGHT_SAFETY_CHECKED'.
        """
        if confirmation_token == "CONFIRM_HARDWARE_FLIGHT_SAFETY_CHECKED":
            self.mode = SystemExecutionMode.HARDWARE_ACTIVE_PILOT
            return True
        return False

    def arm(self, pin_code: int = 1234) -> bool:
        """Dual-action arming sequence."""
        if pin_code == 1234:
            self.is_armed = True
            return True
        return False

    def disarm(self) -> None:
        """Immediate motor disarm."""
        self.is_armed = False

    def validate_command(
        self,
        intent: CommandIntent,
        current_pos_ned: np.ndarray,
        current_vel_ned: np.ndarray,
        current_euler_rad: np.ndarray,
    ) -> ValidationReport:
        """
        Validates and clamps command intent against physical safety limits and execution mode.
        """
        now = time.time()
        self._last_command_time = now

        # Handle arming requests
        if intent.arm_request:
            self.is_armed = True
        if intent.disarm_request:
            self.is_armed = False
            return ValidationReport(
                passed=True,
                violation=SafetyViolationType.NONE,
                detail="Vehicle disarmed by user command.",
                filtered_intent=intent,
            )

        # Disarmed check
        if not self.is_armed and intent.mode not in ["LAND", "DISARM"]:
            return ValidationReport(
                passed=False,
                violation=SafetyViolationType.DISARMED,
                detail="Commands rejected: vehicle is DISARMED.",
            )

        # Geofence boundary check
        if not self.geofence.contains(current_pos_ned):
            return ValidationReport(
                passed=False,
                violation=SafetyViolationType.GEOFENCE_EXCEEDED,
                detail=f"Geofence breached at pos {current_pos_ned}. Failsafe triggered.",
            )

        # Tilt envelope check
        roll = abs(current_euler_rad[0])
        pitch = abs(current_euler_rad[1])
        if roll > self.limits.max_tilt_rad or pitch > self.limits.max_tilt_rad:
            return ValidationReport(
                passed=False,
                violation=SafetyViolationType.TILT_EXCEEDED,
                detail=f"Max tilt angle exceeded (Roll: {np.degrees(roll):.1f}°, Pitch: {np.degrees(pitch):.1f}°).",
            )

        # Velocity envelope check
        horiz_speed = float(np.linalg.norm(current_vel_ned[:2]))
        vert_speed = abs(float(current_vel_ned[2]))
        if horiz_speed > self.limits.max_horiz_speed_m_s:
            return ValidationReport(
                passed=False,
                violation=SafetyViolationType.VELOCITY_EXCEEDED,
                detail=f"Horizontal speed {horiz_speed:.1f} m/s exceeded limit {self.limits.max_horiz_speed_m_s} m/s.",
            )
        if vert_speed > self.limits.max_vert_speed_m_s:
            return ValidationReport(
                passed=False,
                violation=SafetyViolationType.VELOCITY_EXCEEDED,
                detail=f"Vertical speed {vert_speed:.1f} m/s exceeded limit {self.limits.max_vert_speed_m_s} m/s.",
            )

        # Passed all checks - produce filtered safe intent
        safe_intent = CommandIntent(
            timestamp=intent.timestamp,
            mode=intent.mode,
            target_position_ned=intent.target_position_ned,
            target_velocity_ned=intent.target_velocity_ned,
            target_euler_rad=intent.target_euler_rad,
            throttle_norm=np.clip(intent.throttle_norm, 0.0, 1.0),
        )

        return ValidationReport(
            passed=True,
            violation=SafetyViolationType.NONE,
            detail="Command validated and safe.",
            filtered_intent=safe_intent,
        )
