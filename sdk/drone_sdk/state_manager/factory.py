"""
state_manager.factory
=====================
StateFactory — constructs and merges DroneStateVector instances.

Purpose
-------
Provide a clean, tested construction path for every data source that will
eventually feed into the StateStore:

* ``create_initial``   — default zero state (tests, cold-start)
* ``from_dict``        — deserialise from a plain dictionary
* ``from_6dof``        — adapt the existing rigid_body.py 13-state output
* ``merge_update``     — apply a DroneStateUpdate to a base state

The ``from_6dof`` method is the critical bridge to the *existing* codebase:
the ``RigidBodyDynamics.step_rk4()`` call in the current ``sdk/drone_sdk/
dynamics/rigid_body.py`` returns a 13-element state vector.  This factory
converts that into a DroneStateVector so the simulation loop can feed the
StateStore without any changes to existing modules.

The ``merge_update`` method is what Module 3 (MAVLink) will call for each
incoming MAVLink message: it returns a new DroneStateVector with only the
fields supplied in the update applied, leaving all others unchanged.

Python version: 3.9+
"""
from __future__ import annotations

import math
import time
import logging
from typing import Optional

import numpy as np

from .schema import (
    ArmingState,
    DataSource,
    DroneStateUpdate,
    DroneStateVector,
    FlightMode,
    HealthStatus,
    VehicleConfig,
)
from .exceptions import FactoryError

logger = logging.getLogger(__name__)


class StateFactory:
    """Static factory for DroneStateVector construction.

    All methods are ``@staticmethod``; no instance is needed.
    """

    # ── Create ────────────────────────────────────────────────────────────────

    @staticmethod
    def create_initial(
        vehicle_id: str,
        config: Optional[VehicleConfig] = None,
        source: DataSource = DataSource.MANUAL,
    ) -> DroneStateVector:
        """Return a valid ground-rest state with all fields at zero / identity.

        The quaternion is initialised to the identity rotation [1, 0, 0, 0]
        (level hover) so that ``quaternion_norm == 1.0`` and the state passes
        PhysicsValidator out of the box.

        Args:
            vehicle_id: Identifier that matches the target StateStore.
            config:     Optional VehicleConfig; only used for logging.
            source:     DataSource tag for the initial state.

        Returns:
            A DroneStateVector with ``is_valid=True``.
        """
        now_wall = time.time()
        now_mono = time.monotonic()

        state = DroneStateVector(
            vehicle_id     = vehicle_id,
            sequence       = 0,
            timestamp_wall = now_wall,
            timestamp_mono = now_mono,
            timestamp_sim  = 0.0,
            source         = source,
            # All floats default to 0.0 except:
            q0             = 1.0,        # Identity quaternion
            battery_soc    = 1.0,        # Full charge
            battery_soh    = 1.0,        # Perfect health
            battery_temperature = 25.0,  # Ambient
            gps_hdop       = 99.9,
            gps_vdop       = 99.9,
            flight_mode    = FlightMode.UNKNOWN,
            arming_state   = ArmingState.DISARMED,
            health_status  = HealthStatus.NO_DATA,
            health_score   = 0.0,
            is_valid       = True,
            latency_ms     = 0.0,
        )
        logger.debug("StateFactory: created initial state for '%s'", vehicle_id)
        return state

    # ── Adapt from existing rigid_body.py output ──────────────────────────────

    @staticmethod
    def from_6dof(
        vehicle_id: str,
        position: np.ndarray,
        velocity: np.ndarray,
        quaternion: np.ndarray,
        angular_velocity: np.ndarray,
        *,
        rotor_omega: Optional[np.ndarray] = None,
        timestamp_sim: float = 0.0,
        sequence: int = 0,
        source: DataSource = DataSource.SITL,
    ) -> DroneStateVector:
        """Construct a DroneStateVector from the 13-state 6-DOF output.

        This is the bridge to ``sdk/drone_sdk/dynamics/rigid_body.py``.
        The ``RigidBodyDynamics.step_rk4()`` call returns:
            state_13 = [x, y, z, vx, vy, vz, q0, q1, q2, q3, p, q, r]

        Usage::

            pos   = state_13[0:3]
            vel   = state_13[3:6]
            quat  = state_13[6:10]
            omega = state_13[10:13]
            state = StateFactory.from_6dof(
                "drone_0", pos, vel, quat, omega,
                timestamp_sim=t, sequence=k
            )

        Args:
            vehicle_id:       Vehicle identifier.
            position:         [x, y, z] NED, metres.
            velocity:         [vx, vy, vz] NED, m/s.
            quaternion:       [q0, q1, q2, q3] Hamilton (w, x, y, z).
            angular_velocity: [roll_rate, pitch_rate, yaw_rate] body frame, rad/s.
            rotor_omega:      [ω1, ω2, ω3, ω4] rad/s (optional).
            timestamp_sim:    Simulation clock, seconds.
            sequence:         Frame counter.
            source:           DataSource tag.

        Returns:
            DroneStateVector with kinematics populated and derived quantities
            computed (groundspeed, vertical_speed, Euler angles from quaternion).

        Raises:
            FactoryError: If array shapes are wrong.
        """
        StateFactory._assert_shape("position",         position,         (3,))
        StateFactory._assert_shape("velocity",         velocity,         (3,))
        StateFactory._assert_shape("quaternion",       quaternion,       (4,))
        StateFactory._assert_shape("angular_velocity", angular_velocity, (3,))

        q0, q1, q2, q3 = float(quaternion[0]), float(quaternion[1]), \
                          float(quaternion[2]), float(quaternion[3])

        roll, pitch, yaw = StateFactory._quat_to_euler(q0, q1, q2, q3)

        vx, vy, vz = float(velocity[0]), float(velocity[1]), float(velocity[2])
        groundspeed   = math.sqrt(vx**2 + vy**2)
        vertical_speed = -vz               # NED: negative vz = climbing

        # Heading from yaw (NED frame: 0° = North, clockwise positive)
        heading = math.degrees(yaw) % 360.0

        # Rotors
        r_omega = rotor_omega if rotor_omega is not None else np.zeros(4)
        if rotor_omega is not None:
            StateFactory._assert_shape("rotor_omega", rotor_omega, (4,))

        now_wall = time.time()
        now_mono = time.monotonic()

        return DroneStateVector(
            vehicle_id     = vehicle_id,
            sequence       = sequence,
            timestamp_wall = now_wall,
            timestamp_mono = now_mono,
            timestamp_sim  = timestamp_sim,
            source         = source,

            x = float(position[0]),
            y = float(position[1]),
            z = float(position[2]),

            vx = vx, vy = vy, vz = vz,

            q0 = q0, q1 = q1, q2 = q2, q3 = q3,
            roll = roll, pitch = pitch, yaw = yaw,

            roll_rate  = float(angular_velocity[0]),
            pitch_rate = float(angular_velocity[1]),
            yaw_rate   = float(angular_velocity[2]),

            omega1 = float(r_omega[0]),
            omega2 = float(r_omega[1]),
            omega3 = float(r_omega[2]),
            omega4 = float(r_omega[3]),

            groundspeed    = groundspeed,
            vertical_speed = vertical_speed,
            heading        = heading,

            # Battery / GPS are not known from rigid_body.py output
            battery_soc = 1.0,
            battery_soh = 1.0,
            battery_temperature = 25.0,
            gps_hdop = 99.9,
            gps_vdop = 99.9,

            flight_mode  = FlightMode.UNKNOWN,
            arming_state = ArmingState.DISARMED,
            health_status = HealthStatus.NO_DATA,
            is_valid = True,
        )

    # ── Deserialise ───────────────────────────────────────────────────────────

    @staticmethod
    def from_dict(d: dict) -> DroneStateVector:
        """Reconstruct a DroneStateVector from a dictionary (e.g. JSON load).

        The dict must have been produced by ``DroneStateVector.to_dict()``.
        Enum fields are accepted as either their int/str value or the enum itself.

        Args:
            d: Dictionary of field values.

        Returns:
            DroneStateVector instance.

        Raises:
            FactoryError: If mandatory fields are missing.
        """
        try:
            # Normalise enums
            raw = dict(d)  # copy to avoid mutating input
            fm = raw.get("flight_mode", 0)
            raw["flight_mode"] = FlightMode(fm) if isinstance(fm, int) else FlightMode[fm]

            arm = raw.get("arming_state", 0)
            raw["arming_state"] = ArmingState(arm) if isinstance(arm, int) else ArmingState[arm]

            hs = raw.get("health_status", 3)
            raw["health_status"] = HealthStatus(hs) if isinstance(hs, int) else HealthStatus[hs]

            src = raw.get("source", "unknown")
            raw["source"] = DataSource(src) if isinstance(src, str) else src

            return DroneStateVector(**raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise FactoryError(f"Cannot construct DroneStateVector from dict: {exc}") from exc

    # ── Merge partial update ──────────────────────────────────────────────────

    @staticmethod
    def merge_update(
        base: DroneStateVector,
        update: DroneStateUpdate,
    ) -> DroneStateVector:
        """Apply a partial DroneStateUpdate to a base DroneStateVector.

        Only fields that are *not None* in the update are applied.
        All other fields are carried over from base unchanged.

        This is the core operation that Module 3 (MAVLink) will use: each
        MAVLink message type (ATTITUDE, LOCAL_POSITION_NED, BATTERY_STATUS…)
        provides a DroneStateUpdate with only the relevant fields populated.

        Args:
            base:   The current canonical state.
            update: Partial update from a single data source.

        Returns:
            New DroneStateVector; base is not mutated.

        Example::

            update = DroneStateUpdate("drone_0", DataSource.MAVLINK)
            update.battery_voltage = 15.2
            update.battery_soc = 0.73
            new_state = StateFactory.merge_update(current_state, update)
        """
        if base.vehicle_id != update.vehicle_id:
            raise FactoryError(
                f"vehicle_id mismatch: base='{base.vehicle_id}' "
                f"update='{update.vehicle_id}'"
            )

        kwargs: dict = {
            "sequence":        base.sequence + 1,
            "timestamp_wall":  update.timestamp_wall,
            "timestamp_mono":  update.timestamp_mono,
            "timestamp_sim":   update.timestamp_sim if update.timestamp_sim else base.timestamp_sim,
            "source":          update.source,
        }

        # Position
        if update.position is not None:
            StateFactory._assert_shape("update.position", update.position, (3,))
            kwargs["x"] = float(update.position[0])
            kwargs["y"] = float(update.position[1])
            kwargs["z"] = float(update.position[2])

        # Velocity + derived
        if update.velocity is not None:
            StateFactory._assert_shape("update.velocity", update.velocity, (3,))
            vx, vy, vz = map(float, update.velocity)
            kwargs.update(vx=vx, vy=vy, vz=vz)
            kwargs["groundspeed"]    = math.sqrt(vx**2 + vy**2)
            kwargs["vertical_speed"] = -vz

        # Acceleration
        if update.acceleration is not None:
            StateFactory._assert_shape("update.acceleration", update.acceleration, (3,))
            kwargs["ax"] = float(update.acceleration[0])
            kwargs["ay"] = float(update.acceleration[1])
            kwargs["az"] = float(update.acceleration[2])

        # Euler angles
        if update.euler is not None:
            StateFactory._assert_shape("update.euler", update.euler, (3,))
            kwargs["roll"]  = float(update.euler[0])
            kwargs["pitch"] = float(update.euler[1])
            kwargs["yaw"]   = float(update.euler[2])
            kwargs["heading"] = math.degrees(float(update.euler[2])) % 360.0

        # Quaternion (recompute euler if euler not supplied)
        if update.quaternion is not None:
            StateFactory._assert_shape("update.quaternion", update.quaternion, (4,))
            q0, q1, q2, q3 = map(float, update.quaternion)
            kwargs.update(q0=q0, q1=q1, q2=q2, q3=q3)
            if update.euler is None:
                roll, pitch, yaw = StateFactory._quat_to_euler(q0, q1, q2, q3)
                kwargs.update(roll=roll, pitch=pitch, yaw=yaw)
                kwargs["heading"] = math.degrees(yaw) % 360.0

        # Angular velocity
        if update.angular_velocity is not None:
            StateFactory._assert_shape("update.angular_velocity", update.angular_velocity, (3,))
            kwargs["roll_rate"]  = float(update.angular_velocity[0])
            kwargs["pitch_rate"] = float(update.angular_velocity[1])
            kwargs["yaw_rate"]   = float(update.angular_velocity[2])

        # Rotor speeds
        if update.rotor_omega is not None:
            StateFactory._assert_shape("update.rotor_omega", update.rotor_omega, (4,))
            kwargs["omega1"] = float(update.rotor_omega[0])
            kwargs["omega2"] = float(update.rotor_omega[1])
            kwargs["omega3"] = float(update.rotor_omega[2])
            kwargs["omega4"] = float(update.rotor_omega[3])

        # Battery
        for attr in (
            "battery_voltage", "battery_current", "battery_soc",
            "battery_soh", "battery_temperature",
        ):
            val = getattr(update, attr)
            if val is not None:
                kwargs[attr] = val

        # battery_soc present: remaining Wh estimated by StateStore on merge

        # GPS
        for attr in (
            "gps_lat", "gps_lon", "gps_alt_msl", "gps_fix_type",
            "gps_satellites", "gps_hdop", "gps_vdop",
        ):
            val = getattr(update, attr)
            if val is not None:
                kwargs[attr] = val

        if "gps_alt_msl" in kwargs:
            kwargs["altitude_amsl"] = kwargs["gps_alt_msl"]

        # Derived
        if update.altitude_agl is not None:
            kwargs["altitude_agl"] = update.altitude_agl
        if update.airspeed is not None:
            kwargs["airspeed"] = update.airspeed

        # Status
        if update.flight_mode is not None:
            kwargs["flight_mode"] = update.flight_mode
        if update.arming_state is not None:
            kwargs["arming_state"] = update.arming_state

        return base.copy_with(**kwargs)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _assert_shape(name: str, arr: np.ndarray, expected: tuple) -> None:
        """Raise FactoryError if arr.shape != expected."""
        if np.asarray(arr).shape != expected:
            raise FactoryError(
                f"'{name}' has shape {np.asarray(arr).shape}, expected {expected}"
            )

    @staticmethod
    def _quat_to_euler(q0: float, q1: float, q2: float, q3: float) -> tuple[float, float, float]:
        """Convert Hamilton quaternion [w, x, y, z] to Euler angles (rad).

        Uses the ZYX (yaw-pitch-roll) intrinsic convention, which is the
        standard for NED-frame aerospace applications and consistent with both
        PX4 and the existing rigid_body.py.

        Returns:
            (roll, pitch, yaw) in radians.
        """
        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2.0 * (q0 * q2 - q3 * q1)
        sinp = max(-1.0, min(1.0, sinp))   # clamp for numerical safety
        pitch = math.asin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
