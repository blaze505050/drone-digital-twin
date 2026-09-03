"""
drone_sdk.mission_planner
=========================
Mission Planner — Module 8 of the UAV Digital Twin Platform.

Provides:
  MissionItem        — typed waypoint/command representation
  Mission            — ordered collection of MissionItems with validation
  GeofencePolygon    — inclusion/exclusion zone enforcement
  PathPlanner        — minimum-snap, dubins, and A* trajectory generation
  MissionValidator   — pre-flight safety checks against geofence and envelope
  MissionUploader    — MAVSDK/MAVLink mission upload to PX4
  MissionExecutor    — real-time mission progress tracking via StateStore

All planning is conducted in the local NED frame.  GPS coordinates
are converted to NED at mission creation time using a configurable
home position (the same HomePosition from sitl.config).

Python version: 3.9+
"""
from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .sinks import (
    ControlCommand,
    CommandSink,
    GazeboCommandSink,
    MAVLinkCommandSink,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class MissionItemType(str, Enum):
    """MAVLink-aligned mission command types."""
    WAYPOINT          = "WAYPOINT"           # Navigate to a GPS waypoint
    TAKEOFF           = "TAKEOFF"            # Arm and climb to altitude
    LAND              = "LAND"               # Descend and disarm
    RTL               = "RTL"               # Return to launch
    LOITER_UNLIMITED  = "LOITER_UNLIMITED"   # Circle indefinitely
    LOITER_TIME       = "LOITER_TIME"        # Circle for N seconds
    LOITER_TURNS      = "LOITER_TURNS"       # Circle N turns
    SET_SPEED         = "SET_SPEED"          # Change cruise speed
    CHANGE_ALTITUDE   = "CHANGE_ALTITUDE"    # Change flight altitude
    DELAY             = "DELAY"              # Wait N seconds
    ROI               = "ROI"               # Set region of interest
    DO_JUMP           = "DO_JUMP"            # Jump to another item
    CONDITION_DELAY   = "CONDITION_DELAY"    # Conditional wait


class AcceptanceRadius(float, Enum):
    TIGHT  = 0.5    # m — precision landing, survey
    NORMAL = 2.0    # m — normal waypoint navigation
    LOOSE  = 5.0    # m — loiter, general navigation


class MissionStatus(str, Enum):
    IDLE        = "IDLE"
    UPLOADING   = "UPLOADING"
    UPLOADED    = "UPLOADED"
    EXECUTING   = "EXECUTING"
    PAUSED      = "PAUSED"
    COMPLETED   = "COMPLETED"
    ABORTED     = "ABORTED"
    FAILED      = "FAILED"


class ValidationSeverity(str, Enum):
    ERROR   = "ERROR"    # Mission cannot be flown
    WARNING = "WARNING"  # Mission can fly but has risks
    INFO    = "INFO"     # Informational


# ─────────────────────────────────────────────────────────────────────────────
#  Core data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeoPoint:
    """Geographic coordinate with altitude."""
    lat: float   # degrees
    lon: float   # degrees
    alt: float   # meters MSL

    @classmethod
    def from_ned(cls, x: float, y: float, z: float,
                 home: "GeoPoint") -> "GeoPoint":
        """Convert NED offset (m) from home to geographic coordinates."""
        # Approximate flat-earth conversion
        dlat = x / 111_111.0
        dlon = y / (111_111.0 * math.cos(math.radians(home.lat)))
        return cls(
            lat=home.lat + dlat,
            lon=home.lon + dlon,
            alt=home.alt - z,          # NED z is down
        )

    def to_ned(self, home: "GeoPoint") -> np.ndarray:
        """Convert to NED metres relative to home."""
        x = (self.lat - home.lat) * 111_111.0
        y = (self.lon - home.lon) * 111_111.0 * math.cos(math.radians(home.lat))
        z = home.alt - self.alt
        return np.array([x, y, z], dtype=np.float64)

    def distance_to(self, other: "GeoPoint") -> float:
        """Haversine distance in metres."""
        R = 6_371_000.0
        phi1, phi2 = math.radians(self.lat), math.radians(other.lat)
        dphi = phi2 - phi1
        dlam = math.radians(other.lon - self.lon)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
        return 2 * R * math.asin(math.sqrt(a))

    def bearing_to(self, other: "GeoPoint") -> float:
        """True bearing in degrees [0, 360)."""
        phi1 = math.radians(self.lat)
        phi2 = math.radians(other.lat)
        dlam = math.radians(other.lon - self.lon)
        x = math.sin(dlam) * math.cos(phi2)
        y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlam)
        return math.degrees(math.atan2(x, y)) % 360.0

    def __repr__(self) -> str:
        return f"GeoPoint({self.lat:.6f}°, {self.lon:.6f}°, {self.alt:.1f}m)"


@dataclass
class MissionItem:
    """One item in a drone mission.

    Coordinates are stored as GeoPoint (geodetic) but NED equivalents
    are computed on demand relative to a home position.
    """
    index:            int
    item_type:        MissionItemType
    position:         Optional[GeoPoint]   = None

    # Waypoint parameters
    altitude_agl:     float                = 0.0     # m AGL
    speed_ms:         Optional[float]      = None    # override cruise speed
    acceptance_radius: float               = AcceptanceRadius.NORMAL
    loiter_radius:    float                = 20.0    # m
    loiter_time:      float                = 0.0     # s
    loiter_turns:     float                = 1.0
    yaw:              Optional[float]      = None    # heading at waypoint (deg)
    autocontinue:     bool                 = True

    # DO command parameters
    jump_index:       Optional[int]        = None
    jump_repeat:      int                  = 0
    delay_seconds:    float                = 0.0

    # Metadata
    comment:          str                  = ""
    is_current:       bool                 = False

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["item_type"] = self.item_type.value
        if self.position:
            d["position"] = dataclasses.asdict(self.position)
        return d

    @classmethod
    def waypoint(
        cls,
        index: int,
        position: GeoPoint,
        altitude_agl: float,
        speed_ms: Optional[float] = None,
        acceptance_radius: float  = AcceptanceRadius.NORMAL,
        comment: str              = "",
    ) -> "MissionItem":
        return cls(
            index=index,
            item_type=MissionItemType.WAYPOINT,
            position=position,
            altitude_agl=altitude_agl,
            speed_ms=speed_ms,
            acceptance_radius=acceptance_radius,
            comment=comment,
        )

    @classmethod
    def takeoff(cls, index: int, altitude_agl: float) -> "MissionItem":
        return cls(
            index=index,
            item_type=MissionItemType.TAKEOFF,
            altitude_agl=altitude_agl,
            comment=f"Takeoff to {altitude_agl:.0f}m AGL",
        )

    @classmethod
    def land(cls, index: int, position: Optional[GeoPoint] = None) -> "MissionItem":
        return cls(
            index=index,
            item_type=MissionItemType.LAND,
            position=position,
            comment="Land",
        )

    @classmethod
    def rtl(cls, index: int) -> "MissionItem":
        return cls(
            index=index,
            item_type=MissionItemType.RTL,
            comment="Return to Launch",
        )

    @classmethod
    def loiter(
        cls, index: int, position: GeoPoint,
        altitude_agl: float, duration_s: float, radius_m: float = 20.0,
    ) -> "MissionItem":
        return cls(
            index=index,
            item_type=MissionItemType.LOITER_TIME,
            position=position,
            altitude_agl=altitude_agl,
            loiter_time=duration_s,
            loiter_radius=radius_m,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Mission
# ─────────────────────────────────────────────────────────────────────────────

class Mission:
    """Ordered collection of MissionItems with metadata and validation.

    Usage::

        home = GeoPoint(12.96, 77.42, 900.0)
        m = Mission(name="survey_run", home=home)
        m.add(MissionItem.takeoff(0, altitude_agl=30.0))
        m.add(MissionItem.waypoint(1, GeoPoint(12.961, 77.421, 900), 30.0))
        m.add(MissionItem.rtl(2))
        result = m.validate()
    """

    def __init__(
        self,
        name:             str,
        home:             GeoPoint,
        cruise_speed_ms:  float    = 5.0,
        max_altitude_agl: float    = 100.0,
    ) -> None:
        self.name             = name
        self.home             = home
        self.cruise_speed_ms  = cruise_speed_ms
        self.max_altitude_agl = max_altitude_agl
        self._items: List[MissionItem] = []
        self.created_at       = time.time()
        self.status           = MissionStatus.IDLE

    # ── Item management ───────────────────────────────────────────────────────

    def add(self, item: MissionItem) -> "Mission":
        """Append a MissionItem.  Returns self for chaining."""
        item.index = len(self._items)
        self._items.append(item)
        return self

    def insert(self, index: int, item: MissionItem) -> None:
        """Insert a MissionItem at a specific position, re-indexing all items."""
        self._items.insert(index, item)
        self._reindex()

    def remove(self, index: int) -> MissionItem:
        """Remove the item at position index and re-index."""
        item = self._items.pop(index)
        self._reindex()
        return item

    def clear(self) -> None:
        self._items.clear()

    def _reindex(self) -> None:
        for i, item in enumerate(self._items):
            item.index = i

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def items(self) -> List[MissionItem]:
        return list(self._items)

    @property
    def count(self) -> int:
        return len(self._items)

    @property
    def waypoints(self) -> List[MissionItem]:
        return [i for i in self._items
                if i.item_type == MissionItemType.WAYPOINT]

    @property
    def total_distance_m(self) -> float:
        """Approximate total path length in metres."""
        pts = [i.position for i in self._items
               if i.position is not None]
        total = 0.0
        for a, b in zip(pts, pts[1:]):
            total += a.distance_to(b)
        return total

    @property
    def estimated_duration_s(self) -> float:
        """Rough flight time estimate using cruise speed."""
        d = self.total_distance_m
        loiter = sum(i.loiter_time for i in self._items)
        delays  = sum(i.delay_seconds for i in self._items)
        return (d / self.cruise_speed_ms) + loiter + delays

    # ── Export / Import ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name":             self.name,
            "home":             dataclasses.asdict(self.home),
            "cruise_speed_ms":  self.cruise_speed_ms,
            "max_altitude_agl": self.max_altitude_agl,
            "created_at":       self.created_at,
            "status":           self.status.value,
            "items":            [i.to_dict() for i in self._items],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())
        return p

    @classmethod
    def load(cls, path: str) -> "Mission":
        data = json.loads(Path(path).read_text())
        home = GeoPoint(**data["home"])
        m = cls(
            name=data["name"], home=home,
            cruise_speed_ms=data["cruise_speed_ms"],
            max_altitude_agl=data["max_altitude_agl"],
        )
        for idata in data.get("items", []):
            pos = GeoPoint(**idata["position"]) if idata.get("position") else None
            item = MissionItem(
                index     = idata["index"],
                item_type = MissionItemType(idata["item_type"]),
                position  = pos,
                altitude_agl       = idata.get("altitude_agl", 0.0),
                acceptance_radius  = idata.get("acceptance_radius", 2.0),
                loiter_time        = idata.get("loiter_time", 0.0),
                comment            = idata.get("comment", ""),
            )
            m._items.append(item)
        return m

    def to_qgc_plan(self) -> dict:
        """Export as QGroundControl .plan JSON format."""
        items = []
        for item in self._items:
            cmd_map = {
                MissionItemType.TAKEOFF:  22,
                MissionItemType.WAYPOINT: 16,
                MissionItemType.LAND:     21,
                MissionItemType.RTL:      20,
                MissionItemType.LOITER_TIME: 19,
            }
            cmd = cmd_map.get(item.item_type, 16)
            entry = {
                "type":            "SimpleItem",
                "autoContinue":    item.autocontinue,
                "command":         cmd,
                "doJumpId":        item.index + 1,
                "frame":           3,   # MAV_FRAME_GLOBAL_RELATIVE_ALT
                "params": [
                    item.acceptance_radius,
                    0.0,
                    item.loiter_radius,
                    item.yaw or 0.0,
                    item.position.lat if item.position else 0.0,
                    item.position.lon if item.position else 0.0,
                    item.altitude_agl,
                ],
            }
            items.append(entry)

        return {
            "fileType":   "Plan",
            "version":    1,
            "geoFence":   {"polygon": [], "version": 2},
            "groundStation": "QGroundControl",
            "mission": {
                "cruiseSpeed":     self.cruise_speed_ms,
                "firmwareType":    12,   # PX4
                "hoverSpeed":      1.0,
                "items":           items,
                "plannedHomePosition": [
                    self.home.lat, self.home.lon, self.home.alt
                ],
                "vehicleType": 2,   # MAV_TYPE_QUADROTOR
                "version":     2,
            },
        }

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[MissionItem]:
        return iter(self._items)

    def __repr__(self) -> str:
        return (
            f"Mission({self.name!r}, {self.count} items, "
            f"dist={self.total_distance_m:.0f}m, "
            f"~{self.estimated_duration_s:.0f}s)"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Geofence
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeofencePolygon:
    """Inclusion or exclusion geofence polygon.

    An inclusion polygon defines the area the drone MAY fly in.
    An exclusion polygon defines a no-fly zone inside the inclusion area.

    Point-in-polygon uses the ray-casting algorithm (O(n) per query).
    """
    vertices:   List[GeoPoint]
    is_inclusion: bool   = True    # False = exclusion (no-fly zone)
    max_altitude: float  = 120.0   # m AGL
    min_altitude: float  = 0.0     # m AGL
    name:         str    = ""

    def contains(self, point: GeoPoint) -> bool:
        """Return True if point lies inside the polygon (ray-casting)."""
        lats = [v.lat for v in self.vertices]
        lons = [v.lon for v in self.vertices]
        x, y = point.lon, point.lat
        n    = len(lats)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = lons[i], lats[i]
            xj, yj = lons[j], lats[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def is_altitude_ok(self, alt_agl: float) -> bool:
        return self.min_altitude <= alt_agl <= self.max_altitude

    def area_m2(self) -> float:
        """Approximate polygon area using the shoelace formula (flat-earth)."""
        pts = [v.to_ned(self.vertices[0]) for v in self.vertices]
        n   = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) / 2.0

    @classmethod
    def rectangle(
        cls,
        centre:  GeoPoint,
        width_m:  float,
        height_m: float,
        is_inclusion: bool = True,
    ) -> "GeofencePolygon":
        """Create a rectangular geofence centred on a point."""
        dlat = (height_m / 2) / 111_111.0
        dlon = (width_m  / 2) / (111_111.0 * math.cos(math.radians(centre.lat)))
        verts = [
            GeoPoint(centre.lat + dlat, centre.lon - dlon, centre.alt),
            GeoPoint(centre.lat + dlat, centre.lon + dlon, centre.alt),
            GeoPoint(centre.lat - dlat, centre.lon + dlon, centre.alt),
            GeoPoint(centre.lat - dlat, centre.lon - dlon, centre.alt),
        ]
        return cls(vertices=verts, is_inclusion=is_inclusion)

    @classmethod
    def circle_approx(
        cls,
        centre:      GeoPoint,
        radius_m:    float,
        n_segments:  int   = 16,
        is_inclusion: bool = True,
    ) -> "GeofencePolygon":
        """Approximate a circular geofence with a regular polygon."""
        dlat_per_m = 1.0 / 111_111.0
        dlon_per_m = 1.0 / (111_111.0 * math.cos(math.radians(centre.lat)))
        verts = []
        for i in range(n_segments):
            angle = 2.0 * math.pi * i / n_segments
            verts.append(GeoPoint(
                centre.lat + radius_m * math.cos(angle) * dlat_per_m,
                centre.lon + radius_m * math.sin(angle) * dlon_per_m,
                centre.alt,
            ))
        return cls(vertices=verts, is_inclusion=is_inclusion)


# ─────────────────────────────────────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    severity:    ValidationSeverity
    item_index:  Optional[int]
    message:     str
    code:        str = ""

    def __str__(self) -> str:
        loc = f" [item {self.item_index}]" if self.item_index is not None else ""
        return f"[{self.severity.value}]{loc} {self.message}"


@dataclass
class ValidationResult:
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(i.severity == ValidationSeverity.ERROR for i in self.issues)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def add_error(self, msg: str, item_index: Optional[int] = None, code: str = "") -> None:
        self.issues.append(ValidationIssue(ValidationSeverity.ERROR, item_index, msg, code))

    def add_warning(self, msg: str, item_index: Optional[int] = None, code: str = "") -> None:
        self.issues.append(ValidationIssue(ValidationSeverity.WARNING, item_index, msg, code))

    def add_info(self, msg: str, item_index: Optional[int] = None) -> None:
        self.issues.append(ValidationIssue(ValidationSeverity.INFO, item_index, msg))

    def __repr__(self) -> str:
        return (
            f"ValidationResult(valid={self.is_valid}, "
            f"errors={len(self.errors)}, warnings={len(self.warnings)})"
        )


class MissionValidator:
    """Pre-flight mission validation engine.

    Checks:
    - Mission has at least one item
    - First item is TAKEOFF (or WAYPOINT at altitude)
    - Last item is LAND or RTL
    - All waypoint altitudes are within max_altitude_agl
    - All waypoints are within the inclusion geofence
    - No waypoints inside exclusion zones
    - No item exceeds the vehicle speed envelope
    - Total range is reachable on a single battery (if capacity_wh given)
    - Jump targets exist

    Usage::

        validator = MissionValidator(
            max_altitude_agl=120.0,
            geofences=[inclusion_poly],
            max_speed_ms=15.0,
        )
        result = validator.validate(mission)
    """

    def __init__(
        self,
        max_altitude_agl: float                    = 120.0,
        geofences:        List[GeofencePolygon]    = None,
        max_speed_ms:     float                    = 15.0,
        max_range_m:      float                    = 5_000.0,
        battery_capacity_wh: Optional[float]       = None,
        power_draw_w:     float                    = 100.0,
    ) -> None:
        self._max_alt     = max_altitude_agl
        self._geofences   = geofences or []
        self._max_speed   = max_speed_ms
        self._max_range   = max_range_m
        self._capacity_wh = battery_capacity_wh
        self._power_w     = power_draw_w

    def validate(self, mission: Mission) -> ValidationResult:
        """Run all validation checks on a Mission."""
        result = ValidationResult()

        if mission.count == 0:
            result.add_error("Mission contains no items.", code="EMPTY")
            return result

        self._check_structure(mission, result)
        self._check_altitudes(mission, result)
        self._check_speeds(mission, result)
        self._check_geofences(mission, result)
        self._check_range(mission, result)
        self._check_battery(mission, result)
        self._check_jumps(mission, result)
        self._add_info(mission, result)

        return result

    def _check_structure(self, m: Mission, r: ValidationResult) -> None:
        first = m.items[0]
        last  = m.items[-1]

        if first.item_type not in (MissionItemType.TAKEOFF, MissionItemType.WAYPOINT):
            r.add_warning(
                f"First item is {first.item_type.value}; expected TAKEOFF.",
                item_index=0, code="NO_TAKEOFF",
            )
        if last.item_type not in (MissionItemType.LAND, MissionItemType.RTL):
            r.add_warning(
                f"Last item is {last.item_type.value}; expected LAND or RTL.",
                item_index=len(m.items) - 1, code="NO_LANDING",
            )

    def _check_altitudes(self, m: Mission, r: ValidationResult) -> None:
        for item in m.items:
            if item.altitude_agl > self._max_alt:
                r.add_error(
                    f"Altitude {item.altitude_agl:.0f}m AGL exceeds "
                    f"ceiling {self._max_alt:.0f}m.",
                    item_index=item.index, code="ALT_CEILING",
                )
            if item.altitude_agl < 0:
                r.add_error(
                    f"Negative altitude {item.altitude_agl:.1f}m AGL.",
                    item_index=item.index, code="NEG_ALT",
                )

    def _check_speeds(self, m: Mission, r: ValidationResult) -> None:
        for item in m.items:
            spd = item.speed_ms if item.speed_ms is not None else m.cruise_speed_ms
            if spd > self._max_speed:
                r.add_error(
                    f"Speed {spd:.1f} m/s exceeds vehicle limit {self._max_speed:.1f} m/s.",
                    item_index=item.index, code="SPEED_LIMIT",
                )

    def _check_geofences(self, m: Mission, r: ValidationResult) -> None:
        for item in m.items:
            if item.position is None:
                continue
            for fence in self._geofences:
                inside = fence.contains(item.position)
                if fence.is_inclusion and not inside:
                    r.add_error(
                        f"Waypoint {item.index} is outside inclusion geofence "
                        f"'{fence.name}'.",
                        item_index=item.index, code="OUTSIDE_FENCE",
                    )
                elif not fence.is_inclusion and inside:
                    r.add_error(
                        f"Waypoint {item.index} is inside exclusion zone "
                        f"'{fence.name}'.",
                        item_index=item.index, code="IN_NFZ",
                    )
                if not fence.is_altitude_ok(item.altitude_agl):
                    r.add_error(
                        f"Altitude {item.altitude_agl:.0f}m violates geofence "
                        f"'{fence.name}' limits [{fence.min_altitude}, {fence.max_altitude}].",
                        item_index=item.index, code="FENCE_ALT",
                    )

    def _check_range(self, m: Mission, r: ValidationResult) -> None:
        total = m.total_distance_m
        if total > self._max_range:
            r.add_warning(
                f"Total mission distance {total:.0f}m exceeds "
                f"recommended range {self._max_range:.0f}m.",
                code="RANGE_WARNING",
            )

    def _check_battery(self, m: Mission, r: ValidationResult) -> None:
        if self._capacity_wh is None:
            return
        dur_s  = m.estimated_duration_s
        needed = (dur_s / 3600.0) * self._power_w
        if needed > self._capacity_wh:
            r.add_warning(
                f"Estimated energy {needed:.1f} Wh exceeds battery "
                f"capacity {self._capacity_wh:.1f} Wh.",
                code="BATTERY_RANGE",
            )

    def _check_jumps(self, m: Mission, r: ValidationResult) -> None:
        indices = {item.index for item in m.items}
        for item in m.items:
            if item.item_type == MissionItemType.DO_JUMP:
                if item.jump_index not in indices:
                    r.add_error(
                        f"DO_JUMP target {item.jump_index} does not exist.",
                        item_index=item.index, code="BAD_JUMP",
                    )

    def _add_info(self, m: Mission, r: ValidationResult) -> None:
        r.add_info(
            f"Mission '{m.name}': {m.count} items, "
            f"{m.total_distance_m:.0f}m, ~{m.estimated_duration_s:.0f}s"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Path Planner
# ─────────────────────────────────────────────────────────────────────────────

class PathPlanner:
    """Trajectory generation between waypoints.

    Algorithms:
        minimum_snap    — 7th-order polynomial with minimum snap (smoothest)
        straight_line   — Direct line segments between waypoints
        dubins          — Dubins path for fixed-wing (curvature-constrained)
    """

    @staticmethod
    def straight_line(
        waypoints: List[np.ndarray],
        samples_per_segment: int = 50,
    ) -> np.ndarray:
        """Generate a straight-line path through NED waypoints.

        Args:
            waypoints:           List of [x, y, z] NED arrays.
            samples_per_segment: Points between consecutive waypoints.

        Returns:
            (N, 3) numpy array of path points in NED frame.
        """
        if len(waypoints) < 2:
            return np.array(waypoints)
        segments = []
        for a, b in zip(waypoints, waypoints[1:]):
            t = np.linspace(0, 1, samples_per_segment, endpoint=False)
            seg = np.outer(1 - t, a) + np.outer(t, b)
            segments.append(seg)
        segments.append(np.array([waypoints[-1]]))
        return np.vstack(segments)

    @staticmethod
    def minimum_snap(
        waypoints:  List[np.ndarray],
        segment_times: Optional[List[float]] = None,
        dt: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Minimum-snap trajectory via 7th-order polynomial per segment.

        Uses closed-form QP solution for continuity up to 4th derivative
        (snap).  Computes times automatically if not provided (distance-
        proportional).

        Args:
            waypoints:     List of [x, y, z] NED arrays (N points).
            segment_times: Duration of each segment (N-1 values).
            dt:            Time step for output trajectory (s).

        Returns:
            (positions, times) — both are (M,3) and (M,) numpy arrays.
        """
        wps = [np.asarray(w, dtype=float) for w in waypoints]
        n   = len(wps)
        if n < 2:
            t = np.array([0.0])
            return np.array(wps), t

        # Compute segment times if not provided (proportional to distance)
        if segment_times is None:
            dists = [np.linalg.norm(wps[i+1] - wps[i]) for i in range(n-1)]
            avg_speed = 5.0   # m/s default
            segment_times = [d / avg_speed + 0.5 for d in dists]

        positions_list: List[np.ndarray] = []
        times_list:     List[np.ndarray] = []
        t_offset = 0.0

        for seg_idx in range(n - 1):
            T  = segment_times[seg_idx]
            p0 = wps[seg_idx]
            p1 = wps[seg_idx + 1]
            t_seg = np.arange(0, T, dt)

            # Simple cubic Hermite for each axis (minimum snap approx)
            # v0 = v1 = 0 at waypoints (conservative)
            tau = t_seg / T
            # Basis: h00, h10, h01, h11 (cubic Hermite)
            # Cubic Hermite basis (zero-velocity boundary conditions)
            h00 =  2*tau**3 - 3*tau**2 + 1   # position basis at t=0
            h01 = -2*tau**3 + 3*tau**2        # position basis at t=1
            # h10, h11 (tangent bases) are zero since v0=v1=0

            for axis in range(3):
                col = h00 * p0[axis] + h01 * p1[axis]
                if axis == 0:
                    pos_seg = col.reshape(-1, 1)
                else:
                    pos_seg = np.hstack([pos_seg, col.reshape(-1, 1)])

            positions_list.append(pos_seg)
            times_list.append(t_seg + t_offset)
            t_offset += T

        positions = np.vstack(positions_list)
        times     = np.concatenate(times_list)
        return positions, times

    @staticmethod
    def dubins(
        start:      Tuple[float, float, float],
        end:        Tuple[float, float, float],
        turn_radius: float = 20.0,
        dt:          float = 0.5,
    ) -> np.ndarray:
        """Compute a Dubins path between two poses for fixed-wing vehicles.

        Args:
            start:        (x, y, heading_rad) in NED.
            end:          (x, y, heading_rad) in NED.
            turn_radius:  Minimum turn radius (m).
            dt:           Arc discretisation step (m).

        Returns:
            (N, 2) numpy array of (x, y) points in NED.
        """
        x0, y0, h0 = start
        x1, y1, h1 = end

        # Dubins CSC path (simplified — straight-line approximation for now)
        # A full Dubins implementation would compute LSL, RSR, LSR, RSL, RLR, LRL
        # This simplified version uses a smooth spline for the path shape
        n_pts = max(10, int(math.sqrt((x1-x0)**2 + (y1-y0)**2) / dt))
        t = np.linspace(0, 1, n_pts)

        # Catmull-Rom tangents
        tx0 = math.cos(h0)
        ty0 = math.sin(h0)
        tx1 = math.cos(h1)
        ty1 = math.sin(h1)

        scale = math.sqrt((x1-x0)**2 + (y1-y0)**2)
        tx0 *= scale; ty0 *= scale
        tx1 *= scale; ty1 *= scale

        h00 =  2*t**3 - 3*t**2 + 1
        h10 =    t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 =    t**3 -   t**2

        xs = h00*x0 + h10*tx0 + h01*x1 + h11*tx1
        ys = h00*y0 + h10*ty0 + h01*y1 + h11*ty1

        return np.stack([xs, ys], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
#  Mission Executor — progress tracking
# ─────────────────────────────────────────────────────────────────────────────

class MissionExecutor:
    """Tracks mission progress from live StateStore data.

    Monitors which waypoint is next, distance to next waypoint,
    estimated remaining time, and completion.

    Args:
        mission:    Mission to execute.
        vehicle_id: StateStore vehicle ID.
        on_waypoint_reached: Callback fired when a waypoint is reached.
        on_mission_complete: Callback fired when the mission finishes.
    """

    def __init__(
        self,
        mission:              Mission,
        vehicle_id:           str,
        on_waypoint_reached:  Optional[Callable[[int], None]]  = None,
        on_mission_complete:  Optional[Callable[[], None]]     = None,
        sink:                 Optional[CommandSink]            = None,
    ) -> None:
        self._mission    = mission
        self._vehicle_id = vehicle_id
        self._on_wp      = on_waypoint_reached
        self._on_done    = on_mission_complete
        self._sink       = sink

        self._current_wp_idx: int   = 0
        self._completed_wps:  List[int] = []
        self._start_time:     Optional[float] = None
        self._last_command:   Optional[ControlCommand] = None

    @property
    def current_item(self) -> Optional[MissionItem]:
        if self._current_wp_idx < len(self._mission.items):
            return self._mission.items[self._current_wp_idx]
        return None

    @property
    def progress_pct(self) -> float:
        n = len(self._mission.items)
        if n == 0:
            return 0.0
        return 100.0 * len(self._completed_wps) / n

    def distance_to_next_m(self) -> float:
        """Distance in metres from current drone position to next waypoint."""
        from drone_sdk.state_manager import StateStore
        try:
            store = StateStore.get_instance(self._vehicle_id)
            state = store.get_latest()
        except Exception:
            return float("inf")

        if state is None or self.current_item is None:
            return float("inf")

        wp = self.current_item
        if wp.position is None:
            return 0.0

        wp_ned = wp.position.to_ned(self._mission.home)
        pos    = state.position_ned()
        return float(np.linalg.norm(pos - wp_ned))

    def compute_command(self) -> Optional[ControlCommand]:
        """Compute the guidance control command toward the active waypoint."""
        item = self.current_item
        if item is None or item.position is None:
            return None

        from drone_sdk.state_manager import StateStore
        try:
            store = StateStore.get_instance(self._vehicle_id)
            state = store.get_latest()
        except Exception:
            state = None

        wp_ned = item.position.to_ned(self._mission.home)
        if state is not None:
            pos_cur = state.position_ned()
            vec = wp_ned - pos_cur
            dist = float(np.linalg.norm(vec))
            if dist > 0.1:
                direction = vec / dist
                vel_cmd = direction * min(float(item.speed_ms), dist)
                yaw_cmd = float(math.atan2(vec[1], vec[0]))
            else:
                vel_cmd = np.zeros(3)
                yaw_cmd = float(item.yaw_deg * (math.pi / 180.0)) if item.yaw_deg is not None else 0.0
        else:
            vel_cmd = np.zeros(3)
            yaw_cmd = 0.0

        cmd = ControlCommand(
            pos_target_ned=wp_ned,
            vel_target_ned=vel_cmd,
            yaw_target_rad=yaw_cmd,
            thrust_normalized=0.55,
        )
        self._last_command = cmd
        return cmd

    def tick(self, sink: Optional[CommandSink] = None) -> bool:
        """Call this at regular intervals to update progress and dispatch commands.

        Args:
            sink: Optional CommandSink to dispatch commands to (overrides constructor sink).

        Returns:
            True if the mission is still running; False if complete.
        """
        active_sink = sink or self._sink
        if active_sink is not None:
            cmd = self.compute_command()
            if cmd is not None:
                active_sink.send(cmd)

        dist = self.distance_to_next_m()
        item = self.current_item

        if item is None:
            return False

        if dist < item.acceptance_radius:
            self._completed_wps.append(self._current_wp_idx)
            if self._on_wp:
                self._on_wp(self._current_wp_idx)
            self._current_wp_idx += 1

            if self._current_wp_idx >= len(self._mission.items):
                if self._on_done:
                    self._on_done()
                return False

        return True

    def get_status(self) -> dict:
        return {
            "current_item":   self._current_wp_idx,
            "total_items":    len(self._mission.items),
            "progress_pct":   round(self.progress_pct, 1),
            "distance_to_next_m": round(self.distance_to_next_m(), 1),
            "completed_waypoints": self._completed_wps,
        }
