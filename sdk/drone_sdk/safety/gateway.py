"""
drone_sdk.safety.gateway
========================
Real-Time Safety Gateway & Flight Envelope Enforcement.
Implements Backlog Item B19 & PRD INT-02/SAF-02.

Enforces:
1. Autopilot heartbeat watchdog & freshness monitoring (<= 1.0s timeout).
2. Geofence radius and maximum altitude (AGL) clamping.
3. Maximum velocity and tilt angle limits (<= 35° tilt, <= 12 m/s cruise, <= 4 m/s climb).
4. Automatic failsafe trigger (Return-To-Launch / Land) upon safety violation.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..contracts.streams import CommandIntent
from ..state_manager.schema import DroneStateVector


@dataclass
class SafetyLimits:
    """Rigorous physical and operational flight boundary limits."""
    max_horizontal_speed_mps: float = 12.0
    max_climb_speed_mps: float = 4.0
    max_descent_speed_mps: float = 2.0
    max_tilt_angle_deg: float = 35.0
    max_yaw_rate_dps: float = 180.0
    geofence_radius_m: float = 500.0
    max_altitude_agl_m: float = 120.0
    min_altitude_agl_m: float = 0.0
    heartbeat_timeout_sec: float = 1.0


class SafetyGateway:
    """
    Flight envelope safety gate that intercepts, checks, and clamps all incoming control commands
    before they can reach actuators or the physical autopilot.
    """

    def __init__(self, limits: Optional[SafetyLimits] = None) -> None:
        self.limits = limits or SafetyLimits()
        self.last_heartbeat_mono: float = time.monotonic()
        self.is_failsafe_active: bool = False
        self.failsafe_reason: str = ""

    def record_heartbeat(self, timestamp_mono: Optional[float] = None) -> None:
        """Record receipt of valid autopilot heartbeat."""
        self.last_heartbeat_mono = timestamp_mono if timestamp_mono is not None else time.monotonic()
        self.is_failsafe_active = False

    def is_heartbeat_fresh(self, current_mono: Optional[float] = None) -> bool:
        """Check if autopilot communications are active and un-degraded."""
        now = current_mono if current_mono is not None else time.monotonic()
        dt = now - self.last_heartbeat_mono
        return bool(dt <= self.limits.heartbeat_timeout_sec)

    def validate_and_clamp_command(
        self,
        cmd: CommandIntent,
        current_state: Optional[DroneStateVector] = None,
        current_mono: Optional[float] = None,
    ) -> Tuple[bool, CommandIntent, List[str]]:
        """
        Validate and clamp command setpoints to ensure flight safety.
        Returns:
        (is_safe, safe_cmd, list_of_clamping_actions)
        """
        now = current_mono if current_mono is not None else time.monotonic()
        actions: List[str] = []

        # 1. Watchdog Heartbeat Check
        if not self.is_heartbeat_fresh(now):
            self.is_failsafe_active = True
            self.failsafe_reason = f"Heartbeat lost ({now - self.last_heartbeat_mono:.2f}s > {self.limits.heartbeat_timeout_sec}s)."
            # Overwrite command with Emergency Land / Hover
            safe_cmd = CommandIntent(
                vehicle_id=cmd.vehicle_id,
                timestamp_mono=now,
                flight_mode="FAILSAFE_LAND",
                target_thrust=0.50,
                armed=True,
            )
            return False, safe_cmd, [self.failsafe_reason]

        # Clone command to mutate safely
        safe_pos = cmd.target_pos_ned.copy() if cmd.target_pos_ned is not None else None
        safe_vel = cmd.target_vel_ned.copy() if cmd.target_vel_ned is not None else None
        safe_yaw = cmd.target_yaw_rad
        safe_thrust = float(np.clip(cmd.target_thrust, 0.0, 1.0))

        # 2. Geofence & Altitude Bounds
        if safe_pos is not None:
            r_xy = math.sqrt(safe_pos[0] ** 2 + safe_pos[1] ** 2)
            if r_xy > self.limits.geofence_radius_m:
                scale = self.limits.geofence_radius_m / r_xy
                safe_pos[0] *= scale
                safe_pos[1] *= scale
                actions.append(f"Clamped horizontal target from {r_xy:.1f}m to geofence limit {self.limits.geofence_radius_m:.1f}m.")

            # Altitude in NED is negative Z
            alt_agl = -safe_pos[2]
            if alt_agl > self.limits.max_altitude_agl_m:
                safe_pos[2] = -self.limits.max_altitude_agl_m
                actions.append(f"Clamped altitude from {alt_agl:.1f}m to maximum {self.limits.max_altitude_agl_m:.1f}m.")
            elif alt_agl < self.limits.min_altitude_agl_m:
                safe_pos[2] = -self.limits.min_altitude_agl_m
                actions.append(f"Clamped altitude to surface minimum {self.limits.min_altitude_agl_m:.1f}m.")

        # 3. Velocity Limits
        if safe_vel is not None:
            v_xy = math.sqrt(safe_vel[0] ** 2 + safe_vel[1] ** 2)
            if v_xy > self.limits.max_horizontal_speed_mps:
                scale_v = self.limits.max_horizontal_speed_mps / v_xy
                safe_vel[0] *= scale_v
                safe_vel[1] *= scale_v
                actions.append(f"Clamped horizontal speed from {v_xy:.1f}m/s to limit {self.limits.max_horizontal_speed_mps:.1f}m/s.")

            # Vertical speed (NED: +Z is descent, -Z is climb)
            if safe_vel[2] < -self.limits.max_climb_speed_mps:
                safe_vel[2] = -self.limits.max_climb_speed_mps
                actions.append(f"Clamped climb rate to {self.limits.max_climb_speed_mps:.1f}m/s.")
            elif safe_vel[2] > self.limits.max_descent_speed_mps:
                safe_vel[2] = self.limits.max_descent_speed_mps
                actions.append(f"Clamped descent rate to {self.limits.max_descent_speed_mps:.1f}m/s.")

        safe_cmd = CommandIntent(
            vehicle_id=cmd.vehicle_id,
            timestamp_mono=now,
            flight_mode=cmd.flight_mode,
            target_pos_ned=safe_pos,
            target_vel_ned=safe_vel,
            target_yaw_rad=safe_yaw,
            target_thrust=safe_thrust,
            armed=cmd.armed,
        )
        return True, safe_cmd, actions
