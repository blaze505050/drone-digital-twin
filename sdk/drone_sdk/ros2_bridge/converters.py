"""
ros2_bridge.converters
======================
Bidirectional converters between DroneStateVector and ROS2 message types.

This module is **import-safe without ROS2**: all converter functions receive
pre-created ROS2 message objects as arguments so they can be imported and
called without rclpy installed.  The ROS2Node class (ros2_node.py) is where
rclpy is actually imported.

Conversions provided
--------------------
DroneStateVector → ROS2

    to_odometry(state, msg)           nav_msgs/Odometry
    to_imu(state, msg)                sensor_msgs/Imu
    to_nav_sat_fix(state, msg)        sensor_msgs/NavSatFix
    to_battery_state(state, msg)      sensor_msgs/BatteryState
    to_pose_stamped(state, msg)       geometry_msgs/PoseStamped
    to_twist_stamped(state, msg)      geometry_msgs/TwistStamped

ROS2 → DroneStateUpdate

    from_pose_stamped(msg, vid)       → DroneStateUpdate (position + orientation)
    from_twist_stamped(msg, vid)      → DroneStateUpdate (velocity + angular rates)

Coordinate conventions
----------------------
ROS2 uses ENU (East-North-Up) while our platform uses NED.
All converters include the NED↔ENU rotation:

    x_ENU =  y_NED
    y_ENU =  x_NED
    z_ENU = -z_NED

Quaternion: ROS2 uses Hamilton [x, y, z, w].  Our schema uses [w, x, y, z].
The conversion adds a NED→ENU rotation quaternion.

Python version: 3.9+
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from drone_sdk.state_manager import DataSource, DroneStateUpdate, DroneStateVector


# ── NED ↔ ENU rotation helpers ────────────────────────────────────────────────

def _ned_pos_to_enu(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert NED position to ENU position."""
    return y, x, -z   # ENU: x=East=NED-y, y=North=NED-x, z=Up=-NED-z


def _enu_pos_to_ned(x_enu: float, y_enu: float, z_enu: float) -> tuple[float, float, float]:
    """Convert ENU position to NED position."""
    return y_enu, x_enu, -z_enu


def _ned_vel_to_enu(vx: float, vy: float, vz: float) -> tuple[float, float, float]:
    return vy, vx, -vz


def _enu_vel_to_ned(vx_enu: float, vy_enu: float, vz_enu: float) -> tuple[float, float, float]:
    return vy_enu, vx_enu, -vz_enu


def _ned_quat_to_enu(w: float, x: float, y: float, z: float) -> tuple[float, float, float, float]:
    """Rotate quaternion from NED to ENU frame.

    The NED→ENU rotation is a 180° rotation about the x+z axis:
        q_rot = [cos(90°), 0, 0, sin(90°)] * [cos(90°), sin(90°), 0, 0]
              = [0, 1/√2, 1/√2, 0]

    We apply: q_enu = q_rot ⊗ q_ned ⊗ q_rot*
    """
    # NED→ENU rotation quaternion: [w=0, x=1/√2, y=1/√2, z=0]
    s = 1.0 / math.sqrt(2.0)
    rw, rx, ry, rz = 0.0, s, s, 0.0

    # q_rot ⊗ q_ned
    w1 = rw*w - rx*x - ry*y - rz*z
    x1 = rw*x + rx*w + ry*z - rz*y
    y1 = rw*y - rx*z + ry*w + rz*x
    z1 = rw*z + rx*y - ry*x + rz*w

    # ⊗ q_rot_conj (= [rw, -rx, -ry, -rz])
    w2 = w1*rw + x1*rx + y1*ry + z1*rz
    x2 = -w1*rx + x1*rw - y1*rz + z1*ry
    y2 = -w1*ry + x1*rz + y1*rw - z1*rx
    z2 = -w1*rz - x1*ry + y1*rx + z1*rw

    # Normalise
    norm = math.sqrt(w2**2 + x2**2 + y2**2 + z2**2)
    if norm < 1e-10:
        return 1.0, 0.0, 0.0, 0.0
    return w2/norm, x2/norm, y2/norm, z2/norm


def _make_ros_stamp(msg: Any, t: float) -> None:
    """Set the stamp on a ROS2 Header from a Unix wall time."""
    msg.header.stamp.sec     = int(t)
    msg.header.stamp.nanosec = int((t % 1.0) * 1_000_000_000)


# ── DroneStateVector → ROS2 ───────────────────────────────────────────────────

def to_odometry(state: DroneStateVector, msg: Any, frame_id: str = "odom") -> Any:
    """Fill a nav_msgs/Odometry message from a DroneStateVector.

    Args:
        state:    DroneStateVector (NED frame).
        msg:      Pre-created nav_msgs.msg.Odometry instance.
        frame_id: ROS2 frame ID string.

    Returns:
        The same msg object with all fields populated.
    """
    _make_ros_stamp(msg, state.timestamp_wall)
    msg.header.frame_id   = frame_id
    msg.child_frame_id    = f"{state.vehicle_id}/base_link"

    # Position: NED → ENU
    ex, ey, ez = _ned_pos_to_enu(state.x, state.y, state.z)
    msg.pose.pose.position.x = ex
    msg.pose.pose.position.y = ey
    msg.pose.pose.position.z = ez

    # Orientation: NED quaternion → ENU quaternion, ROS2 order (x,y,z,w)
    ew, ex_q, ey_q, ez_q = _ned_quat_to_enu(state.q0, state.q1, state.q2, state.q3)
    msg.pose.pose.orientation.x = ex_q
    msg.pose.pose.orientation.y = ey_q
    msg.pose.pose.orientation.z = ez_q
    msg.pose.pose.orientation.w = ew

    # Velocity: NED → ENU
    tvx, tvy, tvz = _ned_vel_to_enu(state.vx, state.vy, state.vz)
    msg.twist.twist.linear.x = tvx
    msg.twist.twist.linear.y = tvy
    msg.twist.twist.linear.z = tvz

    # Angular velocity: NED → ENU (same rotation)
    tax, tay, taz = _ned_vel_to_enu(state.roll_rate, state.pitch_rate, state.yaw_rate)
    msg.twist.twist.angular.x = tax
    msg.twist.twist.angular.y = tay
    msg.twist.twist.angular.z = taz

    return msg


def to_imu(state: DroneStateVector, msg: Any, frame_id: str = "imu") -> Any:
    """Fill a sensor_msgs/Imu message.

    Covariance matrices are set to -1 (unknown) unless your sensor
    characterisation provides them.
    """
    _make_ros_stamp(msg, state.timestamp_wall)
    msg.header.frame_id = f"{state.vehicle_id}/{frame_id}"

    # Orientation: NED → ENU
    ew, ex_q, ey_q, ez_q = _ned_quat_to_enu(state.q0, state.q1, state.q2, state.q3)
    msg.orientation.x = ex_q
    msg.orientation.y = ey_q
    msg.orientation.z = ez_q
    msg.orientation.w = ew

    # Angular velocity: NED body → ENU body
    tax, tay, taz = _ned_vel_to_enu(state.roll_rate, state.pitch_rate, state.yaw_rate)
    msg.angular_velocity.x = tax
    msg.angular_velocity.y = tay
    msg.angular_velocity.z = taz

    # Linear acceleration: NED body → ENU body
    aax, aay, aaz = _ned_vel_to_enu(state.ax, state.ay, state.az)
    msg.linear_acceleration.x = aax
    msg.linear_acceleration.y = aay
    msg.linear_acceleration.z = aaz

    # Unknown covariances: -1 in [0]
    for cov in (msg.orientation_covariance,
                msg.angular_velocity_covariance,
                msg.linear_acceleration_covariance):
        cov[0] = -1.0

    return msg


def to_nav_sat_fix(state: DroneStateVector, msg: Any) -> Any:
    """Fill a sensor_msgs/NavSatFix from GPS fields."""
    _make_ros_stamp(msg, state.timestamp_wall)
    msg.header.frame_id = f"{state.vehicle_id}/gps"

    msg.latitude  = state.gps_lat
    msg.longitude = state.gps_lon
    msg.altitude  = state.gps_alt_msl

    # NavSatStatus: 0=no fix, 1=fix, 2=SBAS, 3=GBAS
    msg.status.status  = max(-1, state.gps_fix_type - 1)
    msg.status.service = 1   # GPS

    # Covariance type: 1=approximate (from HDOP/VDOP)
    if state.gps_hdop < 99.0:
        h_std = state.gps_hdop * 5.0   # rough approximation: HDOP × 5 m
        v_std = state.gps_vdop * 5.0
        msg.position_covariance[0] = h_std ** 2
        msg.position_covariance[4] = h_std ** 2
        msg.position_covariance[8] = v_std ** 2
        msg.position_covariance_type = 1
    else:
        msg.position_covariance_type = 0

    return msg


def to_battery_state(state: DroneStateVector, msg: Any) -> Any:
    """Fill a sensor_msgs/BatteryState."""
    _make_ros_stamp(msg, state.timestamp_wall)
    msg.header.frame_id = f"{state.vehicle_id}/battery"

    msg.voltage       = state.battery_voltage
    msg.current       = -state.battery_current   # ROS2: positive = charging
    msg.charge        = float("nan")             # Not measured directly
    msg.capacity      = float("nan")
    msg.design_capacity = float("nan")
    msg.percentage    = state.battery_soc
    msg.temperature   = state.battery_temperature

    # Power supply status: 2=DISCHARGING, 1=CHARGING, 3=FULL
    if state.battery_current > 0.1:
        msg.power_supply_status = 2   # DISCHARGING
    elif state.battery_soc > 0.99:
        msg.power_supply_status = 3   # FULL
    else:
        msg.power_supply_status = 1   # CHARGING

    # Health: 1=GOOD, 2=OVERHEAT, 5=COLD
    if state.battery_temperature > 50.0:
        msg.power_supply_health = 2
    elif state.battery_temperature < -10.0:
        msg.power_supply_health = 5
    else:
        msg.power_supply_health = 1

    return msg


def to_pose_stamped(state: DroneStateVector, msg: Any,
                    frame_id: str = "map") -> Any:
    """Fill a geometry_msgs/PoseStamped (position + orientation only)."""
    _make_ros_stamp(msg, state.timestamp_wall)
    msg.header.frame_id = frame_id

    ex, ey, ez = _ned_pos_to_enu(state.x, state.y, state.z)
    msg.pose.position.x = ex
    msg.pose.position.y = ey
    msg.pose.position.z = ez

    ew, ex_q, ey_q, ez_q = _ned_quat_to_enu(state.q0, state.q1, state.q2, state.q3)
    msg.pose.orientation.x = ex_q
    msg.pose.orientation.y = ey_q
    msg.pose.orientation.z = ez_q
    msg.pose.orientation.w = ew

    return msg


def to_twist_stamped(state: DroneStateVector, msg: Any,
                     frame_id: str = "base_link") -> Any:
    """Fill a geometry_msgs/TwistStamped (linear + angular velocity)."""
    _make_ros_stamp(msg, state.timestamp_wall)
    msg.header.frame_id = f"{state.vehicle_id}/{frame_id}"

    tvx, tvy, tvz = _ned_vel_to_enu(state.vx, state.vy, state.vz)
    msg.twist.linear.x = tvx
    msg.twist.linear.y = tvy
    msg.twist.linear.z = tvz

    tax, tay, taz = _ned_vel_to_enu(state.roll_rate, state.pitch_rate, state.yaw_rate)
    msg.twist.angular.x = tax
    msg.twist.angular.y = tay
    msg.twist.angular.z = taz

    return msg


# ── ROS2 → DroneStateUpdate ───────────────────────────────────────────────────

def from_pose_stamped(msg: Any, vehicle_id: str) -> DroneStateUpdate:
    """Convert a geometry_msgs/PoseStamped to a DroneStateUpdate.

    Handles ENU → NED conversion.
    """
    upd = DroneStateUpdate(
        vehicle_id     = vehicle_id,
        source         = DataSource.ROS2,
        timestamp_wall = time.time(),
        timestamp_mono = time.monotonic(),
    )

    # Position: ENU → NED
    ex = msg.pose.position.x
    ey = msg.pose.position.y
    ez = msg.pose.position.z
    nx, ny, nz = _enu_pos_to_ned(ex, ey, ez)
    upd.position = np.array([nx, ny, nz], dtype=np.float64)

    # Orientation: ROS2 (x,y,z,w) → NED Hamilton (w,x,y,z)
    # (inverse of _ned_quat_to_enu)
    qx = msg.pose.orientation.x
    qy = msg.pose.orientation.y
    qz = msg.pose.orientation.z
    qw = msg.pose.orientation.w

    # ENU → NED: apply conjugate of NED→ENU rotation
    s = 1.0 / math.sqrt(2.0)
    rw, rx, ry, rz = 0.0, -s, -s, 0.0   # conjugate of q_rot

    w1 = rw*qw - rx*qx - ry*qy - rz*qz
    x1 = rw*qx + rx*qw + ry*qz - rz*qy
    y1 = rw*qy - rx*qz + ry*qw + rz*qx
    z1 = rw*qz + rx*qy - ry*qx + rz*qw

    w2 = w1*(-rw) + x1*(-rx) + y1*(-ry) + z1*(-rz)
    x2 = -w1*(-rx) + x1*(-rw) - y1*(-rz) + z1*(-ry)
    y2 = -w1*(-ry) + x1*(-rz) + y1*(-rw) - z1*(-rx)
    z2 = -w1*(-rz) - x1*(-ry) + y1*(-rx) + z1*(-rw)

    norm = math.sqrt(w2**2 + x2**2 + y2**2 + z2**2)
    if norm > 1e-10:
        upd.quaternion = np.array([w2/norm, x2/norm, y2/norm, z2/norm])
    else:
        upd.quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    return upd


def from_twist_stamped(msg: Any, vehicle_id: str) -> DroneStateUpdate:
    """Convert a geometry_msgs/TwistStamped to a DroneStateUpdate."""
    upd = DroneStateUpdate(
        vehicle_id     = vehicle_id,
        source         = DataSource.ROS2,
        timestamp_wall = time.time(),
        timestamp_mono = time.monotonic(),
    )

    vx_enu = msg.twist.linear.x
    vy_enu = msg.twist.linear.y
    vz_enu = msg.twist.linear.z
    vx_ned, vy_ned, vz_ned = _enu_vel_to_ned(vx_enu, vy_enu, vz_enu)
    upd.velocity = np.array([vx_ned, vy_ned, vz_ned], dtype=np.float64)

    wx_enu = msg.twist.angular.x
    wy_enu = msg.twist.angular.y
    wz_enu = msg.twist.angular.z
    p_ned, q_ned, r_ned = _enu_vel_to_ned(wx_enu, wy_enu, wz_enu)
    upd.angular_velocity = np.array([p_ned, q_ned, r_ned], dtype=np.float64)

    return upd
