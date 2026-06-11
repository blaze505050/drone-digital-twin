"""Tests for state_manager.schema"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from drone_sdk.state_manager import (
    ArmingState,
    DataSource,
    DroneStateUpdate,
    DroneStateVector,
    FlightMode,
    HealthStatus,
    StateFactory,
    VehicleConfig,
)


class TestFlightMode:
    def test_integer_values_match_px4(self):
        assert FlightMode.UNKNOWN        == 0
        assert FlightMode.MANUAL         == 1
        assert FlightMode.POSITION_HOLD  == 4
        assert FlightMode.OFFBOARD       == 5

    def test_missing_value_returns_unknown(self):
        assert FlightMode(999) == FlightMode.UNKNOWN

    def test_is_int_enum(self):
        # FlightMode members ARE ints (IntEnum mixin)
        assert isinstance(FlightMode.MANUAL, int)
        assert isinstance(FlightMode.OFFBOARD, int)
        # Values are consistent with PX4 MAVLink custom mode integers
        assert int(FlightMode.UNKNOWN) == 0


class TestArmingState:
    def test_values(self):
        assert ArmingState.DISARMED == 0
        assert ArmingState.ARMED    == 2

    def test_missing_returns_disarmed(self):
        assert ArmingState(999) == ArmingState.DISARMED


class TestDataSource:
    def test_is_string(self):
        assert DataSource.MAVLINK == "mavlink"
        assert DataSource.ROS2    == "ros2"

    def test_serialises_as_string(self):
        import json
        encoded = json.dumps({"source": DataSource.MAVLINK.value})
        assert json.loads(encoded)["source"] == "mavlink"


class TestDroneStateVector:
    def test_default_quaternion_is_identity(self, initial_state):
        s = initial_state
        assert s.q0 == 1.0
        assert s.q1 == s.q2 == s.q3 == 0.0

    def test_quaternion_norm_of_identity(self, initial_state):
        assert abs(initial_state.quaternion_norm - 1.0) < 1e-10

    def test_position_ned_returns_numpy(self, initial_state):
        p = initial_state.position_ned()
        assert isinstance(p, np.ndarray)
        assert p.shape == (3,)
        assert p.dtype == np.float64

    def test_velocity_ned_returns_numpy(self, initial_state):
        v = initial_state.velocity_ned()
        assert v.shape == (3,)

    def test_quaternion_returns_numpy(self, initial_state):
        q = initial_state.quaternion()
        assert q.shape == (4,)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10

    def test_euler_angles_returns_numpy(self, initial_state):
        e = initial_state.euler_angles()
        assert e.shape == (3,)

    def test_angular_velocity_body_returns_numpy(self, initial_state):
        w = initial_state.angular_velocity_body()
        assert w.shape == (3,)

    def test_rotor_speeds_returns_numpy(self, initial_state):
        rs = initial_state.rotor_speeds()
        assert rs.shape == (4,)

    def test_is_armed_false_when_disarmed(self, initial_state):
        assert not initial_state.is_armed

    def test_is_armed_true_when_armed(self, initial_state):
        armed = initial_state.copy_with(arming_state=ArmingState.ARMED)
        assert armed.is_armed

    def test_is_airborne_requires_arm_and_altitude(self, initial_state):
        assert not initial_state.is_airborne
        armed = initial_state.copy_with(arming_state=ArmingState.ARMED, altitude_agl=5.0)
        assert armed.is_airborne
        low = initial_state.copy_with(arming_state=ArmingState.ARMED, altitude_agl=0.05)
        assert not low.is_airborne  # below 10 cm threshold

    def test_has_gps_fix_requires_fix_type_3(self, initial_state):
        assert not initial_state.has_gps_fix
        fixed = initial_state.copy_with(gps_fix_type=3)
        assert fixed.has_gps_fix

    def test_copy_with_does_not_mutate_original(self, initial_state):
        original_x = initial_state.x
        updated = initial_state.copy_with(x=100.0)
        assert initial_state.x == original_x
        assert updated.x == 100.0

    def test_copy_with_preserves_unmodified_fields(self, hover_state):
        updated = hover_state.copy_with(x=99.0)
        assert updated.battery_soc    == hover_state.battery_soc
        assert updated.vehicle_id     == hover_state.vehicle_id
        assert updated.flight_mode    == hover_state.flight_mode

    def test_to_dict_is_json_serialisable(self, hover_state):
        import json
        d = hover_state.to_dict()
        encoded = json.dumps(d)  # should not raise
        decoded = json.loads(encoded)
        assert decoded["vehicle_id"] == hover_state.vehicle_id

    def test_to_dict_enum_as_primitive(self, hover_state):
        d = hover_state.to_dict()
        # flight_mode should be int, not FlightMode
        assert isinstance(d["flight_mode"], int)
        assert isinstance(d["source"], str)

    def test_repr_contains_vehicle_id(self, initial_state):
        r = repr(initial_state)
        assert initial_state.vehicle_id in r


class TestDroneStateUpdate:
    def test_default_timestamps_are_set(self, vehicle_id):
        before = time.time()
        upd = DroneStateUpdate(vehicle_id=vehicle_id, source=DataSource.MAVLINK)
        after = time.time()
        assert before <= upd.timestamp_wall <= after

    def test_all_fields_default_to_none(self, vehicle_id):
        upd = DroneStateUpdate(vehicle_id=vehicle_id, source=DataSource.ROS2)
        assert upd.position          is None
        assert upd.velocity          is None
        assert upd.battery_voltage   is None
        assert upd.flight_mode       is None

    def test_can_set_position(self, vehicle_id):
        upd = DroneStateUpdate(vehicle_id=vehicle_id, source=DataSource.SITL)
        upd.position = np.array([1.0, 2.0, -3.0])
        assert upd.position[2] == -3.0


class TestVehicleConfig:
    def test_battery_voltage_max_derived(self):
        cfg = VehicleConfig(vehicle_id="v1", battery_cells=4, cell_voltage_max=4.2)
        assert abs(cfg.battery_voltage_max - 16.8) < 1e-9

    def test_battery_voltage_min_derived(self):
        cfg = VehicleConfig(vehicle_id="v1", battery_cells=4, cell_voltage_min=3.0)
        assert abs(cfg.battery_voltage_min - 12.0) < 1e-9

    def test_battery_voltage_nominal_derived(self):
        cfg = VehicleConfig(vehicle_id="v1", battery_cells=6, cell_voltage_nominal=3.7)
        assert abs(cfg.battery_voltage_nominal - 22.2) < 1e-9

    def test_default_history_size(self):
        cfg = VehicleConfig(vehicle_id="v2")
        assert cfg.history_size == 5_000
