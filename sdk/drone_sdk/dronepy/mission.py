"""
dronepy.mission
===============
High-Level Mission Planning Interface for DronePy.
Wraps and integrates with drone_sdk.mission_planner for seamless waypoint navigation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from drone_sdk.mission_planner import (
    GeoPoint,
    Mission as CoreMission,
    MissionItem as CoreMissionItem,
    MissionItemType,
)


@dataclass
class Waypoint:
    """Single flight phase target in local NED coordinates."""
    pos_ned: np.ndarray
    speed_mps: float = 8.0
    acceptance_radius_m: float = 1.5
    loiter_time_s: float = 0.0
    yaw_deg: Optional[float] = None
    action_type: str = "GOTO"  # TAKEOFF, GOTO, HOVER, LAND, RTL

    def __post_init__(self) -> None:
        self.pos_ned = np.asarray(self.pos_ned, dtype=np.float64)


class Mission:
    """Intuitive high-level mission trajectory planner for DronePy.

    Example
    -------
    >>> mission = Mission()
    >>> mission.takeoff(altitude=15.0)
    >>> mission.hover(10.0)
    >>> mission.goto(50.0, 0.0, -15.0)
    >>> mission.goto(50.0, 50.0, -15.0)
    >>> mission.land()
    """

    def __init__(self, name: str = "DronePyMission", home_ned: Optional[np.ndarray] = None) -> None:
        self.name = name
        self.home_ned = np.asarray(home_ned if home_ned is not None else np.zeros(3), dtype=np.float64)
        self.waypoints: List[Waypoint] = []

    def takeoff(self, altitude: float = 10.0, speed: float = 2.5) -> "Mission":
        """Add takeoff command to climb to given altitude above ground level."""
        wp = Waypoint(
            pos_ned=np.array([self.home_ned[0], self.home_ned[1], -abs(altitude)]),
            speed_mps=speed,
            acceptance_radius_m=1.0,
            action_type="TAKEOFF",
        )
        self.waypoints.append(wp)
        return self

    def hover(self, duration: float = 5.0) -> "Mission":
        """Add a hover/loiter command at current position for duration seconds."""
        last_pos = self.waypoints[-1].pos_ned.copy() if self.waypoints else np.array([0.0, 0.0, -5.0])
        wp = Waypoint(
            pos_ned=last_pos,
            loiter_time_s=duration,
            action_type="HOVER",
        )
        self.waypoints.append(wp)
        return self

    def goto(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        north: Optional[float] = None,
        east: Optional[float] = None,
        down: Optional[float] = None,
        altitude: Optional[float] = None,
        speed: float = 8.0,
        acceptance_radius: float = 1.5,
        yaw_deg: Optional[float] = None,
    ) -> "Mission":
        """Add a 3D waypoint. Supports (x, y, z) or (north, east, down) or altitude AGL."""
        target_x = float(north if north is not None else (x if x is not None else 0.0))
        target_y = float(east if east is not None else (y if y is not None else 0.0))
        if altitude is not None:
            target_z = -abs(float(altitude))
        elif down is not None:
            target_z = float(down)
        elif z is not None:
            target_z = float(z)
        else:
            target_z = -10.0

        wp = Waypoint(
            pos_ned=np.array([target_x, target_y, target_z]),
            speed_mps=speed,
            acceptance_radius_m=acceptance_radius,
            yaw_deg=yaw_deg,
            action_type="GOTO",
        )
        self.waypoints.append(wp)
        return self

    def land(self, speed: float = 1.5) -> "Mission":
        """Add a descent and landing command."""
        last_pos = self.waypoints[-1].pos_ned.copy() if self.waypoints else self.home_ned.copy()
        last_pos[2] = 0.0  # Ground level
        wp = Waypoint(
            pos_ned=last_pos,
            speed_mps=speed,
            acceptance_radius_m=0.5,
            action_type="LAND",
        )
        self.waypoints.append(wp)
        return self

    def rtl(self, return_altitude: float = 15.0, speed: float = 8.0) -> "Mission":
        """Return to launch position and land."""
        rtl_pos = self.home_ned.copy()
        rtl_pos[2] = -abs(return_altitude)
        self.waypoints.append(Waypoint(pos_ned=rtl_pos, speed_mps=speed, action_type="RTL"))
        self.land()
        return self

    def to_core_mission(self, home_geo: Optional[GeoPoint] = None) -> CoreMission:
        """Convert to drone_sdk.mission_planner CoreMission object."""
        home = home_geo or GeoPoint(lat=47.397742, lon=8.545594, alt=488.0)
        core = CoreMission(mission_id=self.name, home=home)
        for i, wp in enumerate(self.waypoints):
            geo = GeoPoint.from_ned(float(wp.pos_ned[0]), float(wp.pos_ned[1]), float(wp.pos_ned[2]), home)
            t_type = MissionItemType.WAYPOINT
            if wp.action_type == "TAKEOFF":
                t_type = MissionItemType.TAKEOFF
            elif wp.action_type == "LAND":
                t_type = MissionItemType.LAND
            elif wp.action_type == "RTL":
                t_type = MissionItemType.RTL

            item = CoreMissionItem(
                index=i,
                item_type=t_type,
                position=geo,
                altitude_agl=-float(wp.pos_ned[2]),
                speed_ms=wp.speed_mps,
                acceptance_radius=wp.acceptance_radius_m,
                loiter_time=wp.loiter_time_s,
                yaw=wp.yaw_deg,
            )
            core.add_item(item)
        return core

    def __len__(self) -> int:
        return len(self.waypoints)

    def __getitem__(self, idx: int) -> Waypoint:
        return self.waypoints[idx]
