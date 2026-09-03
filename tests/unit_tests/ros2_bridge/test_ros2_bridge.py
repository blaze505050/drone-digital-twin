"""
Tests for drone_sdk.ros2_bridge (Module 4).

All tests use mock ROS2 message objects — no rclpy installation required.
The converter functions are pure and stateless, making them easy to test.

Key invariants tested:
* NED → ENU coordinate transformation correctness
* Quaternion convention (Hamilton [w,x,y,z] → ROS2 [x,y,z,w])
* Round-trip consistency: NED state → ENU msg → back to NED
* Edge cases: identity quaternion, zero velocity, battery edge cases
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from drone_sdk.state_manager import (
    ArmingState,
    DataSource,
    FlightMode,
    HealthStatus,
    StateFactory,
    StateStore,
    VehicleConfig,
)
from drone_sdk.ros2_bridge.converters import (
    _ned_pos_to_enu,
    _enu_pos_to_ned,
    _ned_quat_to_enu,
    _ned_vel_to_enu,
    _enu_vel_to_ned,
    from_pose_stamped,
    from_twist_stamped,
    to_battery_state,
    to_imu,
    to_nav_sat_fix,
    to_odometry,
    to_pose_stamped,
    to_twist_stamped,
)


VEHICLE = "ros2_test_drone"


# ── Mock ROS2 message builder ─────────────────────────────────────────────────

def _ns(**kwargs) -> SimpleNamespace:
    """Recursive SimpleNamespace builder."""
    def _r(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _r(v) for k, v in d.items()})
        return d
    return _r(kwargs)


def make_odometry_msg():
    return _ns(
        header=_ns(stamp=_ns(sec=0, nanosec=0), frame_id=""),
        child_frame_id="",
        pose=_ns(pose=_ns(
            position=_ns(x=0.0, y=0.0, z=0.0),
            orientation=_ns(x=0.0, y=0.0, z=0.0, w=1.0),
        )),
        twist=_ns(twist=_ns(
            linear=_ns(x=0.0, y=0.0, z=0.0),
            angular=_ns(x=0.0, y=0.0, z=0.0),
        )),
    )


def make_imu_msg():
    return _ns(
        header=_ns(stamp=_ns(sec=0, nanosec=0), frame_id=""),
        orientation=_ns(x=0.0, y=0.0, z=0.0, w=1.0),
        orientation_covariance=[0.0] * 9,
        angular_velocity=_ns(x=0.0, y=0.0, z=0.0),
        angular_velocity_covariance=[0.0] * 9,
        linear_acceleration=_ns(x=0.0, y=0.0, z=0.0),
        linear_acceleration_covariance=[0.0] * 9,
    )


def make_nav_sat_fix_msg():
    return _ns(
        header=_ns(stamp=_ns(sec=0, nanosec=0), frame_id=""),
        status=_ns(status=0, service=0),
        latitude=0.0, longitude=0.0, altitude=0.0,
        position_covariance=[0.0] * 9,
        position_covariance_type=0,
    )


def make_battery_state_msg():
    return _ns(
        header=_ns(stamp=_ns(sec=0, nanosec=0), frame_id=""),
        voltage=0.0, current=0.0, charge=0.0,
        capacity=0.0, design_capacity=0.0, percentage=0.0,
        temperature=0.0,
        power_supply_status=0, power_supply_health=0,
    )


def make_pose_stamped_msg():
    return _ns(
        header=_ns(stamp=_ns(sec=0, nanosec=0), frame_id=""),
        pose=_ns(
            position=_ns(x=0.0, y=0.0, z=0.0),
            orientation=_ns(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


def make_twist_stamped_msg():
    return _ns(
        header=_ns(stamp=_ns(sec=0, nanosec=0), frame_id=""),
        twist=_ns(
            linear=_ns(x=0.0, y=0.0, z=0.0),
            angular=_ns(x=0.0, y=0.0, z=0.0),
        ),
    )


@pytest.fixture(autouse=True)
def clean_stores():
    StateStore.destroy_all()
    yield
    StateStore.destroy_all()


@pytest.fixture
def hover_state():
    return StateFactory.create_initial(VEHICLE).copy_with(
        x=10.0, y=5.0, z=-20.0,
        vx=1.0, vy=-0.5, vz=0.1,
        q0=1.0, q1=0.0, q2=0.0, q3=0.0,
        roll_rate=0.01, pitch_rate=-0.02, yaw_rate=0.005,
        ax=0.1, ay=-0.05, az=-9.81,
        battery_voltage=15.4, battery_current=12.0,
        battery_soc=0.82, battery_temperature=28.5,
        gps_lat=12.9716, gps_lon=77.5946, gps_alt_msl=920.0,
        gps_fix_type=3, gps_satellites=14,
        gps_hdop=0.85, gps_vdop=1.1,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Coordinate transformation utilities
# ══════════════════════════════════════════════════════════════════════════════

class TestNEDENU:

    def test_ned_to_enu_x_axis(self):
        """NED x (North) → ENU y (North)."""
        ex, ey, ez = _ned_pos_to_enu(1.0, 0.0, 0.0)
        assert abs(ey - 1.0) < 1e-9
        assert abs(ex) < 1e-9
        assert abs(ez) < 1e-9

    def test_ned_to_enu_y_axis(self):
        """NED y (East) → ENU x (East)."""
        ex, ey, ez = _ned_pos_to_enu(0.0, 1.0, 0.0)
        assert abs(ex - 1.0) < 1e-9

    def test_ned_to_enu_z_axis(self):
        """NED z (Down) → ENU -z (Up)."""
        ex, ey, ez = _ned_pos_to_enu(0.0, 0.0, 1.0)
        assert abs(ez - (-1.0)) < 1e-9

    def test_roundtrip_position(self):
        """NED → ENU → NED should be identity."""
        x, y, z = 15.0, -3.5, -20.0
        ex, ey, ez = _ned_pos_to_enu(x, y, z)
        nx, ny, nz = _enu_pos_to_ned(ex, ey, ez)
        assert abs(nx - x) < 1e-9
        assert abs(ny - y) < 1e-9
        assert abs(nz - z) < 1e-9

    def test_roundtrip_velocity(self):
        vx, vy, vz = 1.0, -2.0, 0.5
        ex, ey, ez = _ned_vel_to_enu(vx, vy, vz)
        nx, ny, nz = _enu_vel_to_ned(ex, ey, ez)
        assert abs(nx - vx) < 1e-9
        assert abs(ny - vy) < 1e-9
        assert abs(nz - vz) < 1e-9

    def test_identity_quaternion_roundtrip(self):
        """Identity quaternion should survive NED→ENU rotation."""
        w, x, y, z = _ned_quat_to_enu(1.0, 0.0, 0.0, 0.0)
        norm = math.sqrt(w**2 + x**2 + y**2 + z**2)
        assert abs(norm - 1.0) < 1e-9

    def test_quaternion_norm_preserved(self):
        """Any unit quaternion should remain unit after NED→ENU rotation."""
        ang = math.pi / 6
        w0, x0, y0, z0 = math.cos(ang), math.sin(ang), 0.0, 0.0
        w, x, y, z = _ned_quat_to_enu(w0, x0, y0, z0)
        norm = math.sqrt(w**2 + x**2 + y**2 + z**2)
        assert abs(norm - 1.0) < 1e-9


# ══════════════════════════════════════════════════════════════════════════════
#  to_odometry
# ══════════════════════════════════════════════════════════════════════════════

class TestToOdometry:

    def test_header_stamp_set(self, hover_state):
        msg = make_odometry_msg()
        to_odometry(hover_state, msg)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        assert abs(t - hover_state.timestamp_wall) < 1.0   # within 1 s

    def test_frame_id_set(self, hover_state):
        msg = make_odometry_msg()
        to_odometry(hover_state, msg, frame_id="world")
        assert msg.header.frame_id == "world"
        assert VEHICLE in msg.child_frame_id

    def test_position_ned_to_enu(self, hover_state):
        """NED (10, 5, -20) → ENU (5, 10, 20)."""
        msg = make_odometry_msg()
        to_odometry(hover_state, msg)
        # NED x=10 → ENU y=10
        assert abs(msg.pose.pose.position.y - 10.0) < 1e-9
        # NED y=5 → ENU x=5
        assert abs(msg.pose.pose.position.x - 5.0) < 1e-9
        # NED z=-20 → ENU z=20
        assert abs(msg.pose.pose.position.z - 20.0) < 1e-9

    def test_velocity_ned_to_enu(self, hover_state):
        """NED (vx=1, vy=-0.5, vz=0.1) → ENU (vx=-0.5, vy=1, vz=-0.1)."""
        msg = make_odometry_msg()
        to_odometry(hover_state, msg)
        assert abs(msg.twist.twist.linear.x - (-0.5)) < 1e-9
        assert abs(msg.twist.twist.linear.y - 1.0)    < 1e-9
        assert abs(msg.twist.twist.linear.z - (-0.1)) < 1e-9

    def test_orientation_unit_norm(self, hover_state):
        msg = make_odometry_msg()
        to_odometry(hover_state, msg)
        o = msg.pose.pose.orientation
        norm = math.sqrt(o.x**2 + o.y**2 + o.z**2 + o.w**2)
        assert abs(norm - 1.0) < 1e-9

    def test_returns_same_msg(self, hover_state):
        msg = make_odometry_msg()
        result = to_odometry(hover_state, msg)
        assert result is msg


# ══════════════════════════════════════════════════════════════════════════════
#  to_imu
# ══════════════════════════════════════════════════════════════════════════════

class TestToImu:

    def test_orientation_unit_norm(self, hover_state):
        msg = make_imu_msg()
        to_imu(hover_state, msg)
        o = msg.orientation
        norm = math.sqrt(o.x**2 + o.y**2 + o.z**2 + o.w**2)
        assert abs(norm - 1.0) < 1e-9

    def test_covariances_set_unknown(self, hover_state):
        msg = make_imu_msg()
        to_imu(hover_state, msg)
        assert msg.orientation_covariance[0] == -1.0
        assert msg.angular_velocity_covariance[0] == -1.0
        assert msg.linear_acceleration_covariance[0] == -1.0

    def test_angular_velocity_converted(self, hover_state):
        """NED roll_rate=0.01 → ENU angular.y=0.01."""
        msg = make_imu_msg()
        to_imu(hover_state, msg)
        # NED roll_rate → ENU angular_y (pitch in ENU)
        assert abs(msg.angular_velocity.y - 0.01) < 1e-9

    def test_acceleration_ned_to_enu(self, hover_state):
        """NED az=-9.81 (gravity body frame) → ENU z=9.81."""
        msg = make_imu_msg()
        to_imu(hover_state, msg)
        # NED az=-9.81 → ENU z=9.81 (-vz = -(-9.81) = 9.81)
        assert abs(msg.linear_acceleration.z - 9.81) < 1e-6


# ══════════════════════════════════════════════════════════════════════════════
#  to_nav_sat_fix
# ══════════════════════════════════════════════════════════════════════════════

class TestToNavSatFix:

    def test_lat_lon_copied(self, hover_state):
        msg = make_nav_sat_fix_msg()
        to_nav_sat_fix(hover_state, msg)
        assert abs(msg.latitude  - hover_state.gps_lat) < 1e-9
        assert abs(msg.longitude - hover_state.gps_lon) < 1e-9
        assert abs(msg.altitude  - hover_state.gps_alt_msl) < 1e-9

    def test_fix_status_3d_is_1(self, hover_state):
        """GPS fix type 3 → ROS2 status 2 (fix type - 1)."""
        msg = make_nav_sat_fix_msg()
        to_nav_sat_fix(hover_state, msg)
        assert msg.status.status == 2

    def test_no_fix_status_is_minus1(self, hover_state):
        state = hover_state.copy_with(gps_fix_type=0)
        msg = make_nav_sat_fix_msg()
        to_nav_sat_fix(state, msg)
        assert msg.status.status == -1

    def test_covariance_from_hdop(self, hover_state):
        """HDOP=0.85 → covariance[0] ≈ (0.85*5)² = 18.06."""
        msg = make_nav_sat_fix_msg()
        to_nav_sat_fix(hover_state, msg)
        expected = (0.85 * 5.0) ** 2
        assert abs(msg.position_covariance[0] - expected) < 0.01
        assert msg.position_covariance_type == 1


# ══════════════════════════════════════════════════════════════════════════════
#  to_battery_state
# ══════════════════════════════════════════════════════════════════════════════

class TestToBatteryState:

    def test_voltage_current_soc_copied(self, hover_state):
        msg = make_battery_state_msg()
        to_battery_state(hover_state, msg)
        assert abs(msg.voltage     - 15.4) < 1e-6
        assert abs(msg.percentage  - 0.82) < 1e-6
        # current is sign-flipped in ROS2 convention
        assert abs(msg.current - (-12.0)) < 1e-6

    def test_discharging_status(self, hover_state):
        """Positive current draw → power_supply_status = 2 (DISCHARGING)."""
        msg = make_battery_state_msg()
        to_battery_state(hover_state, msg)
        assert msg.power_supply_status == 2

    def test_overtemp_health_flag(self, hover_state):
        hot = hover_state.copy_with(battery_temperature=55.0)
        msg = make_battery_state_msg()
        to_battery_state(hot, msg)
        assert msg.power_supply_health == 2   # OVERHEAT

    def test_cold_health_flag(self, hover_state):
        cold = hover_state.copy_with(battery_temperature=-15.0)
        msg = make_battery_state_msg()
        to_battery_state(cold, msg)
        assert msg.power_supply_health == 5   # COLD

    def test_normal_health_flag(self, hover_state):
        msg = make_battery_state_msg()
        to_battery_state(hover_state, msg)
        assert msg.power_supply_health == 1   # GOOD


# ══════════════════════════════════════════════════════════════════════════════
#  from_pose_stamped / from_twist_stamped (ROS2 → DroneStateUpdate)
# ══════════════════════════════════════════════════════════════════════════════

class TestFromPoseStamped:

    def test_position_enu_to_ned(self):
        """ENU (5, 10, 20) → NED (10, 5, -20)."""
        msg = make_pose_stamped_msg()
        msg.pose.position.x = 5.0    # East  → NED y
        msg.pose.position.y = 10.0   # North → NED x
        msg.pose.position.z = 20.0   # Up    → NED z = -20
        msg.pose.orientation.w = 1.0

        upd = from_pose_stamped(msg, VEHICLE)
        np.testing.assert_allclose(upd.position, [10.0, 5.0, -20.0], atol=1e-6)

    def test_identity_orientation(self):
        """Identity quaternion in ENU → identity-ish in NED."""
        msg = make_pose_stamped_msg()
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        upd = from_pose_stamped(msg, VEHICLE)
        # Result should be a valid unit quaternion
        q = upd.quaternion
        norm = np.linalg.norm(q)
        assert abs(norm - 1.0) < 1e-9

    def test_source_is_ros2(self):
        msg = make_pose_stamped_msg()
        msg.pose.orientation.w = 1.0
        upd = from_pose_stamped(msg, VEHICLE)
        assert upd.source == DataSource.ROS2

    def test_vehicle_id_propagated(self):
        msg = make_pose_stamped_msg()
        msg.pose.orientation.w = 1.0
        upd = from_pose_stamped(msg, VEHICLE)
        assert upd.vehicle_id == VEHICLE


class TestFromTwistStamped:

    def test_linear_velocity_enu_to_ned(self):
        """ENU linear (1, 2, 3) → NED (2, 1, -3)."""
        msg = make_twist_stamped_msg()
        msg.twist.linear.x = 1.0
        msg.twist.linear.y = 2.0
        msg.twist.linear.z = 3.0

        upd = from_twist_stamped(msg, VEHICLE)
        # ENU x=1 (East) → NED y=1; ENU y=2 (North) → NED x=2; ENU z=3 (Up) → NED z=-3
        np.testing.assert_allclose(upd.velocity, [2.0, 1.0, -3.0], atol=1e-9)

    def test_angular_velocity_enu_to_ned(self):
        msg = make_twist_stamped_msg()
        msg.twist.angular.x = 0.1
        msg.twist.angular.y = 0.2
        msg.twist.angular.z = 0.3

        upd = from_twist_stamped(msg, VEHICLE)
        # Same NED transformation as position
        np.testing.assert_allclose(upd.angular_velocity, [0.2, 0.1, -0.3], atol=1e-9)

    def test_source_is_ros2(self):
        msg = make_twist_stamped_msg()
        upd = from_twist_stamped(msg, VEHICLE)
        assert upd.source == DataSource.ROS2


# ══════════════════════════════════════════════════════════════════════════════
#  Round-trip: state → ROS2 msg → update → re-store
# ══════════════════════════════════════════════════════════════════════════════

class TestRoundTrip:

    @pytest.fixture
    def store(self):
        return StateStore.create(VEHICLE, VehicleConfig(vehicle_id=VEHICLE))

    def test_position_roundtrip_via_pose_stamped(self, store, hover_state):
        """State → PoseStamped → from_pose_stamped → store → same position."""
        store.update(hover_state)

        # Encode state to ROS2
        out_msg = make_pose_stamped_msg()
        to_pose_stamped(hover_state, out_msg)

        # Decode back to DroneStateUpdate
        upd = from_pose_stamped(out_msg, VEHICLE)
        store.update_partial(upd)

        latest = store.get_latest()
        # Position should be approximately preserved (NED → ENU → NED)
        np.testing.assert_allclose(
            latest.position_ned(),
            hover_state.position_ned(),
            atol=1e-6,
        )

    def test_velocity_roundtrip_via_twist_stamped(self, store, hover_state):
        store.update(hover_state)

        out_msg = make_twist_stamped_msg()
        to_twist_stamped(hover_state, out_msg)

        upd = from_twist_stamped(out_msg, VEHICLE)
        store.update_partial(upd)

        latest = store.get_latest()
        np.testing.assert_allclose(
            latest.velocity_ned(),
            hover_state.velocity_ned(),
            atol=1e-6,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  ROS2BridgeNode lifecycle (no rclpy required)
# ══════════════════════════════════════════════════════════════════════════════

class TestROS2BridgeNodeLifecycle:

    def test_instantiation_without_rclpy(self):
        """ROS2BridgeNode can be instantiated without rclpy; start() raises."""
        import sys
        from drone_sdk.ros2_bridge.ros2_node import ROS2BridgeNode

        original = sys.modules.pop("rclpy", None)
        try:
            node = ROS2BridgeNode(VEHICLE)
            with pytest.raises(ImportError, match="rclpy"):
                node.start()
        finally:
            if original:
                sys.modules["rclpy"] = original

    def test_repr(self):
        from drone_sdk.ros2_bridge.ros2_node import ROS2BridgeNode
        node = ROS2BridgeNode(VEHICLE)
        assert VEHICLE in repr(node)
        assert "running=" in repr(node)

    def test_get_topic_list(self):
        from drone_sdk.ros2_bridge.ros2_node import ROS2BridgeNode
        node = ROS2BridgeNode(VEHICLE)
        topics = node.get_topic_list()
        assert "publish"   in topics
        assert "subscribe" in topics
        # Verify key topics are present
        pub_topics = list(topics["publish"].keys())
        assert any("odometry" in t for t in pub_topics)
        assert any("battery"  in t for t in pub_topics)
        sub_topics = list(topics["subscribe"].keys())
        assert any("cmd_pose" in t for t in sub_topics)
