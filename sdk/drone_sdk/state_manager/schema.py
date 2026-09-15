"""
state_manager.schema
====================
Canonical data types for the Digital Twin State Manager.

All modules in the platform (MAVLink, ROS2, SITL, Dashboard, AI/ML) exchange
state through these types.  Nothing else is the source of truth.

Coordinate conventions
----------------------
* Position  : NED frame (North-East-Down), metres
* Velocity  : NED frame, m/s
* Attitude  : Euler angles (roll-pitch-yaw) in radians;
              quaternion in Hamilton form [w, x, y, z]
* Angular   : body frame (roll_rate, pitch_rate, yaw_rate), rad/s
              These are the aerospace p-q-r rates; named explicitly to avoid
              confusion with quaternion component names.
* Rotor     : angular speed magnitude, rad/s (always ≥ 0)
* Battery   : SI units (V, A, Wh, °C)
* GPS       : geodetic (degrees, degrees, metres MSL)

Python version: 3.9+
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum, IntEnum
from typing import Optional

import numpy as np


# ── Enumerations ──────────────────────────────────────────────────────────────

class FlightMode(IntEnum):
    """PX4-aligned flight mode identifiers.

    Integer values are intentionally compatible with MAVLink CUSTOM_MODE so
    that the MAVLink adapter (Module 3) can cast directly without a lookup.
    """
    UNKNOWN         = 0
    MANUAL          = 1
    STABILIZED      = 2
    ALTITUDE_HOLD   = 3
    POSITION_HOLD   = 4
    OFFBOARD        = 5
    AUTO_MISSION    = 6
    AUTO_RTL        = 7
    AUTO_LAND       = 8
    ACRO            = 9
    SPORT           = 10
    HOLD            = 11
    TAKEOFF         = 12

    @classmethod
    def _missing_(cls, value: object) -> "FlightMode":  # noqa: D105
        return cls.UNKNOWN


class ArmingState(IntEnum):
    """Arming state of the vehicle."""
    DISARMED         = 0
    ARMING           = 1
    ARMED            = 2
    DISARMING        = 3
    EMERGENCY_LAND   = 4

    @classmethod
    def _missing_(cls, value: object) -> "ArmingState":  # noqa: D105
        return cls.DISARMED


class HealthStatus(IntEnum):
    """Coarse health classification, suitable for dashboard display."""
    NOMINAL  = 0   # All subsystems within normal bounds
    DEGRADED = 1   # One or more subsystems out of spec but still flyable
    CRITICAL = 2   # Immediate attention required
    NO_DATA  = 3   # No state update has been received yet


class DataSource(str, Enum):
    """Identifies which module produced a state update.

    Inherits from str so values JSON-serialise transparently.
    """
    MAVLINK  = "mavlink"   # Module 3 — PX4 / real hardware
    ROS2     = "ros2"      # Module 4 — ROS2 bridge
    SITL     = "sitl"      # Module 6 — PX4 SITL
    MANUAL   = "manual"    # Direct Python API (tests, demos)
    REPLAY   = "replay"    # Offline log replay
    VISION   = "vision"    # Vision-based tracking / YOLO perception
    TWIN     = "twin"      # Parallel physics prediction digital twin
    HIL      = "hil"       # Hardware-In-The-Loop flight computer
    FPRIME   = "fprime"    # NASA F Prime flight software
    UNKNOWN  = "unknown"


# ── Core state vector ─────────────────────────────────────────────────────────

@dataclass
class DroneStateVector:
    """Complete, authoritative state for a single drone at one point in time.

    This is the **single source of truth** for all drone state.  Every module
    that reads or writes drone state does so through DroneStateVector instances
    stored in a StateStore.

    The dataclass is *mutable* so that StateStore can keep a single live object
    and update it atomically under its RLock, avoiding the cost of allocating a
    new instance on every 100 Hz telemetry tick.  External callers receive
    copies via ``get_latest()`` and should treat them as immutable snapshots.

    Example::

        state = DroneStateVector(vehicle_id="drone_0", sequence=0,
                                  timestamp_wall=time.time(),
                                  timestamp_mono=time.monotonic(),
                                  source=DataSource.MANUAL)
        pos = state.position_ned()   # np.array([0., 0., 0.])
    """

    # ── Metadata ─────────────────────────────────────────────────────────────
    vehicle_id:      str         # Unique vehicle identifier
    sequence:        int         # Monotonically increasing frame counter
    timestamp_wall:  float       # Unix time when state was written (time.time())
    timestamp_mono:  float       # Monotonic time when state was written
    timestamp_sim:   float       # Simulation clock (0.0 for real flights)
    source:          DataSource  # Which module produced this state

    # ── Position — NED, metres ────────────────────────────────────────────────
    x: float = 0.0   # North   (+North)
    y: float = 0.0   # East    (+East)
    z: float = 0.0   # Down    (+Down; altitude_amsl ≈ -z + home_alt)

    # ── Velocity — NED, m/s ───────────────────────────────────────────────────
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # ── Acceleration — body frame, m/s² ──────────────────────────────────────
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0

    # ── Attitude: Euler angles, radians ──────────────────────────────────────
    roll:  float = 0.0   # φ  ∈ (−π, π]
    pitch: float = 0.0   # θ  ∈ (−π/2, π/2]
    yaw:   float = 0.0   # ψ  ∈ (−π, π]

    # ── Attitude: Hamilton quaternion [w, x, y, z] ────────────────────────────
    q0: float = 1.0   # w (scalar part)
    q1: float = 0.0   # x
    q2: float = 0.0   # y
    q3: float = 0.0   # z

    # ── Angular velocity — body frame, rad/s ─────────────────────────────────
    # Named roll_rate / pitch_rate / yaw_rate to avoid confusion with
    # quaternion component names (q0-q3) and numpy variable names.
    roll_rate:  float = 0.0   # p  (positive = right-wing-down)
    pitch_rate: float = 0.0   # q  (positive = nose-up)
    yaw_rate:   float = 0.0   # r  (positive = nose-right / clockwise from above)

    # ── Rotor speeds — rad/s, always ≥ 0 ─────────────────────────────────────
    # X-configuration:   1(FR,CW)  2(FL,CCW)  3(BR,CCW)  4(BL,CW)
    omega1: float = 0.0
    omega2: float = 0.0
    omega3: float = 0.0
    omega4: float = 0.0

    # ── Battery ───────────────────────────────────────────────────────────────
    battery_voltage:      float = 0.0    # Total pack voltage, V
    battery_current:      float = 0.0    # Discharge current, A (positive = discharging)
    battery_soc:          float = 1.0    # State of Charge, 0.0–1.0
    battery_soh:          float = 1.0    # State of Health, 0.0–1.0
    battery_remaining_wh: float = 0.0    # Remaining energy, Wh
    battery_temperature:  float = 25.0   # Cell temperature, °C

    # ── GPS ───────────────────────────────────────────────────────────────────
    gps_lat:        float = 0.0    # Geodetic latitude,  degrees
    gps_lon:        float = 0.0    # Geodetic longitude, degrees
    gps_alt_msl:    float = 0.0    # Altitude above mean sea level, m
    gps_fix_type:   int   = 0      # 0=no fix, 2=2D, 3=3D, 4=DGPS, 5=RTK-float, 6=RTK-fixed
    gps_satellites: int   = 0
    gps_hdop:       float = 99.9   # Horizontal dilution of precision
    gps_vdop:       float = 99.9   # Vertical dilution of precision

    # ── Derived / cached quantities ───────────────────────────────────────────
    # Populated by the store or factory from primary fields
    altitude_agl:   float = 0.0    # Height above ground level, m
    altitude_amsl:  float = 0.0    # Altitude above MSL (== gps_alt_msl if GPS valid)
    airspeed:       float = 0.0    # Indicated airspeed, m/s
    groundspeed:    float = 0.0    # Ground speed magnitude, m/s
    vertical_speed: float = 0.0    # Positive = climbing, m/s
    heading:        float = 0.0    # True heading, degrees [0, 360)

    # ── Flight status ─────────────────────────────────────────────────────────
    flight_mode:  FlightMode  = FlightMode.UNKNOWN
    arming_state: ArmingState = ArmingState.DISARMED

    # ── Health and data quality ───────────────────────────────────────────────
    health_status: HealthStatus = HealthStatus.NO_DATA
    health_score:  float        = 0.0    # 0.0 (dead) → 1.0 (perfect)
    is_valid:      bool         = False  # Passed PhysicsValidator
    latency_ms:    float        = 0.0    # Time since data was acquired, ms

    # ── Convenience accessors ─────────────────────────────────────────────────

    def position_ned(self) -> np.ndarray:
        """Position vector [x, y, z] in NED frame, metres."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def velocity_ned(self) -> np.ndarray:
        """Velocity vector [vx, vy, vz] in NED frame, m/s."""
        return np.array([self.vx, self.vy, self.vz], dtype=np.float64)

    def acceleration_body(self) -> np.ndarray:
        """Acceleration vector [ax, ay, az] in body frame, m/s²."""
        return np.array([self.ax, self.ay, self.az], dtype=np.float64)

    def quaternion(self) -> np.ndarray:
        """Quaternion [w, x, y, z] (Hamilton convention)."""
        return np.array([self.q0, self.q1, self.q2, self.q3], dtype=np.float64)

    def euler_angles(self) -> np.ndarray:
        """Euler angles [roll, pitch, yaw] in radians."""
        return np.array([self.roll, self.pitch, self.yaw], dtype=np.float64)

    def angular_velocity_body(self) -> np.ndarray:
        """Angular velocity [roll_rate, pitch_rate, yaw_rate] in body frame, rad/s."""
        return np.array([self.roll_rate, self.pitch_rate, self.yaw_rate], dtype=np.float64)

    def rotor_speeds(self) -> np.ndarray:
        """Rotor angular velocities [ω1, ω2, ω3, ω4], rad/s."""
        return np.array([self.omega1, self.omega2, self.omega3, self.omega4], dtype=np.float64)

    # ── Derived boolean properties ────────────────────────────────────────────

    @property
    def is_armed(self) -> bool:
        """True when arming_state is ARMED."""
        return self.arming_state == ArmingState.ARMED

    @property
    def is_airborne(self) -> bool:
        """True when armed and more than 10 cm above ground."""
        return self.is_armed and self.altitude_agl > 0.10

    @property
    def has_gps_fix(self) -> bool:
        """True when GPS reports at least a 3-D fix."""
        return self.gps_fix_type >= 3

    @property
    def quaternion_norm(self) -> float:
        """Euclidean norm of the quaternion (should be ≈ 1.0)."""
        return math.sqrt(self.q0**2 + self.q1**2 + self.q2**2 + self.q3**2)

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def copy_with(self, **kwargs: object) -> "DroneStateVector":
        """Return a shallow copy with the specified fields replaced.

        Uses ``dataclasses.replace`` internally.  Does not mutate self.

        Example::

            updated = state.copy_with(x=10.0, vx=2.0)
        """
        # dataclasses.replace with dynamic kwargs passes type checking via ignore:
        return replace(self, **kwargs)  # type: ignore[misc]

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dictionary.

        Enum values are converted to their primitive representation:
        IntEnum → int, str-Enum → str.
        """
        d = asdict(self)
        # asdict converts enums to their values automatically in Python 3.11+
        # but not reliably in 3.9/3.10, so we be explicit:
        d["flight_mode"]  = int(self.flight_mode)
        d["arming_state"] = int(self.arming_state)
        d["health_status"] = int(self.health_status)
        d["source"]        = self.source.value
        return d

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"DroneStateVector(vehicle_id={self.vehicle_id!r}, seq={self.sequence}, "
            f"pos=[{self.x:.2f},{self.y:.2f},{self.z:.2f}] NED, "
            f"yaw={math.degrees(self.yaw):.1f}°, "
            f"mode={self.flight_mode.name}, armed={self.is_armed}, "
            f"valid={self.is_valid})"
        )


# ── Partial update type ───────────────────────────────────────────────────────

@dataclass(eq=False)
class DroneStateUpdate:
    """Partial state update produced by a single data-source adapter.

    Each field is Optional; ``None`` means "this source has no data for this
    field in this update cycle."  The StateStore merges partial updates into the
    canonical DroneStateVector via StateFactory.merge_update().

    Usage (Module 3 — MAVLink adapter, to be implemented later)::

        update = DroneStateUpdate(
            vehicle_id="drone_0",
            source=DataSource.MAVLINK,
        )
        update.position = np.array([10.0, 0.0, -5.0])   # NED
        update.battery_voltage = 15.8
        store.update_partial(update)
    """

    vehicle_id:     str        # Must match an existing StateStore
    source:         DataSource

    timestamp_wall: float = field(default_factory=time.time)
    timestamp_mono: float = field(default_factory=time.monotonic)
    timestamp_sim:  float = 0.0

    # Primary kinematic fields (None = not present in this update)
    position:        Optional[np.ndarray] = None   # [x, y, z] NED, m
    velocity:        Optional[np.ndarray] = None   # [vx, vy, vz] NED, m/s
    acceleration:    Optional[np.ndarray] = None   # [ax, ay, az] body, m/s²
    euler:           Optional[np.ndarray] = None   # [roll, pitch, yaw] rad
    quaternion:      Optional[np.ndarray] = None   # [q0, q1, q2, q3]
    angular_velocity: Optional[np.ndarray] = None  # [roll_rate, pitch_rate, yaw_rate] rad/s
    rotor_omega:     Optional[np.ndarray] = None   # [ω1, ω2, ω3, ω4] rad/s

    # Battery
    battery_voltage:      Optional[float] = None
    battery_current:      Optional[float] = None
    battery_soc:          Optional[float] = None
    battery_soh:          Optional[float] = None
    battery_temperature:  Optional[float] = None

    # GPS
    gps_lat:        Optional[float] = None
    gps_lon:        Optional[float] = None
    gps_alt_msl:    Optional[float] = None
    gps_fix_type:   Optional[int]   = None
    gps_satellites: Optional[int]   = None
    gps_hdop:       Optional[float] = None
    gps_vdop:       Optional[float] = None

    # Derived
    altitude_agl:   Optional[float] = None
    airspeed:       Optional[float] = None

    # Status
    flight_mode:  Optional[FlightMode]  = None
    arming_state: Optional[ArmingState] = None


# ── Vehicle configuration ─────────────────────────────────────────────────────

@dataclass
class VehicleConfig:
    """Per-vehicle configuration for validation bounds and monitoring parameters.

    Provides sensible defaults for a typical 5-inch racing quadrotor.
    Adjust for your specific airframe before creating a StateStore.

    Example (250 mm racing quad)::

        cfg = VehicleConfig(
            vehicle_id="racer_1",
            max_speed_ms=25.0,
            max_rotor_omega_rads=2500.0,
            battery_cells=4,
            expected_update_rate_hz=100.0,
        )
    """

    vehicle_id: str

    # ── Airframe classification ───────────────────────────────────────────────
    vehicle_type: str = "quadrotor_x"   # "quadrotor_x" | "hexarotor" | "fixed_wing"

    # ── Physical validation bounds ────────────────────────────────────────────
    max_altitude_m:           float = 500.0     # AGL, hard ceiling
    max_lateral_range_m:      float = 10_000.0  # from home position
    max_speed_ms:             float = 30.0      # groundspeed magnitude
    max_vertical_speed_ms:    float = 15.0
    max_roll_rad:             float = math.pi / 2      # ≈ 90°
    max_pitch_rad:            float = math.pi / 2
    max_angular_rate_rads:    float = 10.0              # per axis
    max_rotor_omega_rads:     float = 1570.8   # ≈ 15 000 RPM
    min_rotor_omega_rads:     float = 0.0
    quaternion_norm_tol:      float = 1e-3      # |q| must be within this of 1.0

    # ── Battery configuration ─────────────────────────────────────────────────
    battery_cells:         int   = 4     # Series cells (4S)
    cell_voltage_max:      float = 4.20  # V — fully charged
    cell_voltage_nominal:  float = 3.70  # V — storage
    cell_voltage_min:      float = 3.00  # V — critical low
    battery_capacity_wh:   float = 22.0  # Wh — nominal capacity

    # ── Health monitoring ─────────────────────────────────────────────────────
    expected_update_rate_hz: float = 50.0    # Expected telemetry rate
    max_state_age_ms:        float = 500.0   # Stale threshold
    min_health_score:        float = 0.60    # Below → DEGRADED
    critical_health_score:   float = 0.30    # Below → CRITICAL

    # ── History buffer ────────────────────────────────────────────────────────
    history_size: int = 5_000   # Ring buffer capacity (100 s @ 50 Hz)

    # ── Derived battery bounds ────────────────────────────────────────────────

    @property
    def battery_voltage_max(self) -> float:
        """Maximum valid pack voltage."""
        return self.battery_cells * self.cell_voltage_max

    @property
    def battery_voltage_min(self) -> float:
        """Minimum valid pack voltage (land immediately)."""
        return self.battery_cells * self.cell_voltage_min

    @property
    def battery_voltage_nominal(self) -> float:
        """Nominal pack voltage."""
        return self.battery_cells * self.cell_voltage_nominal
