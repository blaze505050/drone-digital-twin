"""Tests for validator, event_bus, health, and factory."""
from __future__ import annotations

import math
import time
import threading
from typing import List

import numpy as np
import pytest

from drone_sdk.state_manager import (
    ArmingState,
    DataSource,
    DroneStateUpdate,
    DroneStateVector,
    EventBus,
    EventType,
    FactoryError,
    FlightMode,
    PhysicsValidator,
    StateFactory,
    VehicleConfig,
    Event,
)
from drone_sdk.state_manager.event_bus import EventBusError
from drone_sdk.state_manager.health import HealthMonitor


# ══════════════════════════════════════════════════════════════════════════════
#  PhysicsValidator
# ══════════════════════════════════════════════════════════════════════════════

class TestPhysicsValidator:

    # ── Quaternion checks ─────────────────────────────────────────────────────

    def test_identity_quaternion_is_valid(self, validator, initial_state):
        result = validator.validate(initial_state)
        assert result.is_valid

    def test_zero_quaternion_is_invalid(self, validator, initial_state):
        bad = initial_state.copy_with(q0=0.0, q1=0.0, q2=0.0, q3=0.0)
        result = validator.validate(bad)
        assert not result.is_valid
        assert any("Quaternion" in i for i in result.issues)

    def test_unnormalised_quaternion_invalid(self, validator, initial_state):
        bad = initial_state.copy_with(q0=2.0, q1=0.0, q2=0.0, q3=0.0)
        result = validator.validate(bad)
        assert not result.is_valid

    def test_near_identity_within_tolerance_valid(self, validator, initial_state):
        tiny = 0.0001
        # Slightly un-normalised but within tolerance
        good = initial_state.copy_with(q0=1.0 + tiny * 0.5, q1=0.0, q2=0.0, q3=0.0)
        # norm ≈ 1.00005 — within default tol of 0.001
        result = validator.validate(good)
        assert result.is_valid

    def test_is_valid_quaternion_helper(self, validator):
        assert validator.is_valid_quaternion(1, 0, 0, 0) is True
        assert validator.is_valid_quaternion(0, 0, 0, 0) is False

    # ── Attitude checks ───────────────────────────────────────────────────────

    def test_excessive_roll_invalid(self, validator, initial_state):
        bad = initial_state.copy_with(roll=math.pi)   # 180° > 90° limit
        result = validator.validate(bad)
        assert not result.is_valid
        assert any("Roll" in i for i in result.issues)

    def test_excessive_pitch_invalid(self, validator, initial_state):
        bad = initial_state.copy_with(pitch=math.pi)
        result = validator.validate(bad)
        assert not result.is_valid

    def test_yaw_unrestricted(self, validator, initial_state):
        for yaw in [-math.pi, 0.0, math.pi - 0.001]:
            s = initial_state.copy_with(yaw=yaw)
            result = validator.validate(s)
            yaw_issues = [i for i in result.issues if "Yaw" in i]
            assert yaw_issues == []

    # ── Velocity checks ───────────────────────────────────────────────────────

    def test_normal_velocity_valid(self, validator, initial_state):
        s = initial_state.copy_with(vx=5.0, vy=3.0, vz=-2.0)
        result = validator.validate(s)
        # No velocity issues
        vel_issues = [i for i in result.issues if "speed" in i.lower()]
        assert vel_issues == []

    def test_excessive_groundspeed_invalid(self, validator, initial_state):
        # 50 m/s > max_speed_ms=30.0
        s = initial_state.copy_with(vx=50.0, vy=0.0, vz=0.0)
        result = validator.validate(s)
        assert not result.is_valid
        assert any("speed" in i.lower() for i in result.issues)

    def test_excessive_vertical_speed_invalid(self, validator, initial_state):
        s = initial_state.copy_with(vz=20.0)  # 20 m/s > 15 limit
        result = validator.validate(s)
        assert not result.is_valid

    def test_is_valid_speed_helper(self, validator):
        assert validator.is_valid_speed(5.0, 3.0, -2.0) is True
        assert validator.is_valid_speed(50.0, 0.0, 0.0) is False

    # ── Battery checks ────────────────────────────────────────────────────────

    def test_nominal_battery_valid(self, validator, hover_state):
        result = validator.validate(hover_state)
        batt_issues = [i for i in result.issues if "battery" in i.lower() or "Battery" in i]
        assert batt_issues == []

    def test_zero_voltage_valid(self, validator, initial_state):
        # 0.0 voltage means "no telemetry yet" — should not flag
        s = initial_state.copy_with(battery_voltage=0.0)
        result = validator.validate(s)
        assert not any("voltage" in i.lower() for i in result.issues)

    def test_over_voltage_invalid(self, validator, initial_state):
        s = initial_state.copy_with(battery_voltage=99.0)
        result = validator.validate(s)
        assert any("voltage" in i.lower() or "Voltage" in i for i in result.issues)

    def test_under_voltage_invalid(self, validator, initial_state):
        s = initial_state.copy_with(battery_voltage=5.0)  # 4S min is 12V
        result = validator.validate(s)
        assert any("voltage" in i.lower() or "Voltage" in i for i in result.issues)

    def test_soc_out_of_range_invalid(self, validator, initial_state):
        s = initial_state.copy_with(battery_soc=1.5)
        result = validator.validate(s)
        assert any("SoC" in i for i in result.issues)

    def test_is_valid_battery_voltage_helper(self, validator):
        assert validator.is_valid_battery_voltage(0.0)    is True   # No telemetry
        assert validator.is_valid_battery_voltage(15.0)   is True   # 4S nominal
        assert validator.is_valid_battery_voltage(99.0)   is False  # Over-voltage
        assert validator.is_valid_battery_voltage(5.0)    is False  # Under-voltage

    # ── Rotor checks ──────────────────────────────────────────────────────────

    def test_nominal_rotors_valid(self, validator, initial_state):
        s = initial_state.copy_with(omega1=500.0, omega2=500.0, omega3=500.0, omega4=500.0)
        result = validator.validate(s)
        rotor_issues = [i for i in result.issues if "Rotor" in i]
        assert rotor_issues == []

    def test_over_speed_rotor_invalid(self, validator, initial_state):
        s = initial_state.copy_with(omega1=9999.0)   # >1600 limit
        result = validator.validate(s)
        assert any("Rotor 1" in i for i in result.issues)

    def test_negative_rotor_invalid(self, validator, initial_state):
        s = initial_state.copy_with(omega2=-50.0)
        result = validator.validate(s)
        assert any("Rotor 2" in i for i in result.issues)

    # ── ValidationResult ─────────────────────────────────────────────────────

    def test_result_bool_is_is_valid(self, validator, initial_state):
        result = validator.validate(initial_state)
        assert bool(result) == result.is_valid

    def test_result_repr(self, validator, initial_state):
        result = validator.validate(initial_state)
        assert "ValidationResult" in repr(result)


# ══════════════════════════════════════════════════════════════════════════════
#  EventBus
# ══════════════════════════════════════════════════════════════════════════════

class TestEventBus:

    def test_subscribe_returns_id(self, event_bus):
        sid = event_bus.subscribe(EventType.STATE_UPDATED, lambda e: None)
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_custom_subscriber_id(self, event_bus):
        sid = event_bus.subscribe(EventType.STATE_UPDATED, lambda e: None,
                                   subscriber_id="my_unique_id")
        assert sid == "my_unique_id"

    def test_duplicate_subscriber_id_raises(self, event_bus):
        event_bus.subscribe(EventType.STATE_UPDATED, lambda e: None, "dup_id")
        with pytest.raises(EventBusError):
            event_bus.subscribe(EventType.STATE_UPDATED, lambda e: None, "dup_id")

    def test_non_callable_raises(self, event_bus):
        with pytest.raises(EventBusError):
            event_bus.subscribe(EventType.STATE_UPDATED, "not_a_callable")

    def test_publish_delivers_to_subscriber(self, event_bus, vehicle_id):
        received: List[Event] = []
        event_bus.subscribe(EventType.STATE_UPDATED, received.append)
        event = Event(EventType.STATE_UPDATED, vehicle_id, data={"seq": 1})
        count = event_bus.publish(event)
        assert count == 1
        assert len(received) == 1
        assert received[0].data == {"seq": 1}

    def test_emit_convenience_method(self, event_bus, vehicle_id):
        received: List[Event] = []
        event_bus.subscribe(EventType.VEHICLE_ARMED, received.append)
        event_bus.emit(EventType.VEHICLE_ARMED)
        assert len(received) == 1

    def test_wrong_event_type_not_delivered(self, event_bus, vehicle_id):
        received: List[Event] = []
        event_bus.subscribe(EventType.VEHICLE_ARMED, received.append)
        event_bus.emit(EventType.STATE_UPDATED)
        assert len(received) == 0

    def test_multi_type_subscription(self, event_bus):
        received: List[Event] = []
        event_bus.subscribe(
            [EventType.VEHICLE_ARMED, EventType.VEHICLE_DISARMED],
            received.append,
        )
        event_bus.emit(EventType.VEHICLE_ARMED)
        event_bus.emit(EventType.VEHICLE_DISARMED)
        event_bus.emit(EventType.STATE_UPDATED)
        assert len(received) == 2

    def test_subscribe_all_receives_every_type(self, event_bus):
        received: List[Event] = []
        event_bus.subscribe_all(received.append)
        event_bus.emit(EventType.STATE_UPDATED)
        event_bus.emit(EventType.VEHICLE_ARMED)
        event_bus.emit(EventType.HEALTH_CRITICAL)
        assert len(received) == 3

    def test_unsubscribe_stops_delivery(self, event_bus):
        received: List[Event] = []
        sid = event_bus.subscribe(EventType.STATE_UPDATED, received.append)
        event_bus.emit(EventType.STATE_UPDATED)
        event_bus.unsubscribe(sid)
        event_bus.emit(EventType.STATE_UPDATED)
        assert len(received) == 1

    def test_unsubscribe_nonexistent_returns_false(self, event_bus):
        assert event_bus.unsubscribe("ghost_id") is False

    def test_unsubscribe_all(self, event_bus):
        for i in range(5):
            event_bus.subscribe(EventType.STATE_UPDATED, lambda e: None, str(i))
        count = event_bus.unsubscribe_all()
        assert count == 5
        assert event_bus.subscriber_count == 0

    def test_subscriber_count(self, event_bus):
        assert event_bus.subscriber_count == 0
        event_bus.subscribe(EventType.STATE_UPDATED, lambda e: None)
        assert event_bus.subscriber_count == 1

    def test_callback_exception_does_not_stop_other_callbacks(self, event_bus):
        results: List[str] = []

        def bad(e):
            raise RuntimeError("fail")

        def good(e):
            results.append("ok")

        event_bus.subscribe(EventType.STATE_UPDATED, bad,  "bad_one")
        event_bus.subscribe(EventType.STATE_UPDATED, good, "good_one")
        event_bus.emit(EventType.STATE_UPDATED)
        assert "ok" in results

    def test_thread_safe_concurrent_subscribe_publish(self, event_bus):
        errors: List[Exception] = []
        received: List[Event] = []
        lock = threading.Lock()

        def publisher():
            for _ in range(100):
                event_bus.emit(EventType.STATE_UPDATED)
                time.sleep(0.001)

        def subscriber_manager():
            for i in range(50):
                sid = str(i + 1000)
                try:
                    event_bus.subscribe(EventType.STATE_UPDATED,
                                        lambda e: None, sid)
                    time.sleep(0.002)
                    event_bus.unsubscribe(sid)
                except Exception as exc:  # noqa: BLE001
                    with lock:
                        errors.append(exc)

        t1 = threading.Thread(target=publisher)
        t2 = threading.Thread(target=subscriber_manager)
        t1.start(); t2.start()
        t1.join(timeout=10); t2.join(timeout=10)

        assert errors == []

    def test_get_stats_returns_dict(self, event_bus):
        event_bus.subscribe(EventType.STATE_UPDATED, lambda e: None)
        stats = event_bus.get_stats()
        assert "subscriber_count" in stats
        assert stats["subscriber_count"] == 1


# ══════════════════════════════════════════════════════════════════════════════
#  HealthMonitor
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthMonitor:

    def test_initial_status_is_no_data(self, config):
        monitor = HealthMonitor(config)
        assert monitor.get_status().name == "NO_DATA"

    def test_is_stale_before_any_update(self, config):
        monitor = HealthMonitor(config)
        assert monitor.is_stale()

    def test_score_zero_before_updates(self, config):
        monitor = HealthMonitor(config)
        assert monitor.get_score() == 0.0

    def test_score_increases_with_valid_updates(self, config):
        monitor = HealthMonitor(config, ema_alpha=0.3)
        t0 = time.monotonic()
        for i in range(60):
            monitor.record_update(t0 + i * 0.02, is_valid=True)   # Simulate 50 Hz
        score = monitor.get_score()
        assert score > 0.5

    def test_score_decreases_with_invalid_updates(self, config):
        # Use real monotonic time so data_age is computed correctly.
        monitor = HealthMonitor(config, ema_alpha=0.3, valid_window=20)
        for _ in range(20):
            monitor.record_update(time.monotonic(), is_valid=True)
        score_good = monitor.get_score()
        for _ in range(20):
            monitor.record_update(time.monotonic(), is_valid=False)
        score_bad = monitor.get_score()
        # valid_score drops from 1.0 → 0.0; overall must fall
        assert score_bad < score_good

    def test_stale_detection(self, config):
        monitor = HealthMonitor(config)
        monitor.record_update(time.monotonic(), is_valid=True)
        assert not monitor.is_stale(max_age_ms=5000.0)
        time.sleep(0.05)
        assert monitor.is_stale(max_age_ms=10.0)

    def test_metrics_fields_populated(self, config):
        monitor = HealthMonitor(config)
        t0 = time.monotonic()
        for i in range(10):
            monitor.record_update(t0 + i * 0.02, is_valid=True)
        m = monitor.get_metrics()
        assert m.update_rate_hz >= 0
        assert 0.0 <= m.overall_score <= 1.0
        assert m.total_updates == 10
        assert m.total_invalid == 0

    def test_reset_clears_statistics(self, config):
        monitor = HealthMonitor(config)
        t0 = time.monotonic()
        for i in range(20):
            monitor.record_update(t0 + i * 0.02, is_valid=True)
        monitor.reset()
        assert monitor.get_metrics().total_updates == 0
        assert monitor.is_stale()


# ══════════════════════════════════════════════════════════════════════════════
#  StateFactory
# ══════════════════════════════════════════════════════════════════════════════

class TestStateFactory:

    def test_create_initial_passes_validator(self, vehicle_id, validator):
        state = StateFactory.create_initial(vehicle_id)
        result = validator.validate(state)
        assert result.is_valid, result.issues

    def test_create_initial_identity_quaternion(self, vehicle_id):
        state = StateFactory.create_initial(vehicle_id)
        assert abs(state.quaternion_norm - 1.0) < 1e-10

    def test_create_initial_disarmed(self, vehicle_id):
        state = StateFactory.create_initial(vehicle_id)
        assert state.arming_state == ArmingState.DISARMED

    def test_from_6dof_correct_position(self, vehicle_id):
        pos = np.array([10.0, 20.0, -30.0])
        vel = np.zeros(3)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.zeros(3)
        state = StateFactory.from_6dof(vehicle_id, pos, vel, quat, omega)
        assert abs(state.x - 10.0) < 1e-9
        assert abs(state.y - 20.0) < 1e-9
        assert abs(state.z - (-30.0)) < 1e-9

    def test_from_6dof_correct_euler_from_identity_quat(self, vehicle_id):
        pos = vel = np.zeros(3)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.zeros(3)
        state = StateFactory.from_6dof(vehicle_id, pos, vel, quat, omega)
        assert abs(state.roll) < 1e-9
        assert abs(state.pitch) < 1e-9
        assert abs(state.yaw) < 1e-9

    def test_from_6dof_90deg_roll(self, vehicle_id):
        """Quaternion for 90° roll: q = [cos(45°), sin(45°), 0, 0]."""
        ang = math.pi / 4
        quat = np.array([math.cos(ang), math.sin(ang), 0.0, 0.0])
        state = StateFactory.from_6dof(vehicle_id, np.zeros(3), np.zeros(3), quat, np.zeros(3))
        assert abs(state.roll - math.pi / 2) < 1e-6

    def test_from_6dof_groundspeed_derived(self, vehicle_id):
        vel = np.array([3.0, 4.0, 0.0])   # groundspeed = 5
        state = StateFactory.from_6dof(vehicle_id, np.zeros(3), vel,
                                        np.array([1,0,0,0]), np.zeros(3))
        assert abs(state.groundspeed - 5.0) < 1e-9

    def test_from_6dof_heading_from_yaw(self, vehicle_id):
        """Yaw of -90° (west) → heading 270°."""
        ang = math.radians(-45)
        quat = np.array([math.cos(ang), 0.0, 0.0, math.sin(ang)])
        state = StateFactory.from_6dof(vehicle_id, np.zeros(3), np.zeros(3), quat, np.zeros(3))
        assert 0.0 <= state.heading < 360.0

    def test_from_6dof_wrong_shape_raises(self, vehicle_id):
        with pytest.raises(FactoryError, match="shape"):
            StateFactory.from_6dof(vehicle_id, np.zeros(2), np.zeros(3),
                                    np.array([1,0,0,0]), np.zeros(3))

    def test_from_dict_roundtrip(self, hover_state):
        d = hover_state.to_dict()
        restored = StateFactory.from_dict(d)
        assert restored.vehicle_id   == hover_state.vehicle_id
        assert restored.flight_mode  == hover_state.flight_mode
        assert abs(restored.x - hover_state.x) < 1e-9

    def test_from_dict_missing_field_raises(self):
        with pytest.raises(FactoryError):
            StateFactory.from_dict({"vehicle_id": "x"})   # missing required fields

    def test_merge_update_battery_only(self, initial_state, partial_update):
        merged = StateFactory.merge_update(initial_state, partial_update)
        assert abs(merged.battery_voltage - partial_update.battery_voltage) < 1e-9
        assert abs(merged.battery_soc - partial_update.battery_soc) < 1e-9
        # Position unchanged
        assert merged.x == initial_state.x

    def test_merge_update_position(self, initial_state, vehicle_id):
        upd = DroneStateUpdate(vehicle_id=vehicle_id, source=DataSource.MAVLINK)
        upd.position = np.array([1.0, 2.0, -3.0])
        merged = StateFactory.merge_update(initial_state, upd)
        assert abs(merged.x - 1.0) < 1e-9
        assert abs(merged.y - 2.0) < 1e-9
        assert abs(merged.z - (-3.0)) < 1e-9

    def test_merge_update_quaternion_derives_euler(self, initial_state, vehicle_id):
        upd = DroneStateUpdate(vehicle_id=vehicle_id, source=DataSource.MAVLINK)
        ang = math.pi / 4
        upd.quaternion = np.array([math.cos(ang), math.sin(ang), 0.0, 0.0])
        merged = StateFactory.merge_update(initial_state, upd)
        assert abs(merged.roll - math.pi / 2) < 1e-6

    def test_merge_vehicle_id_mismatch_raises(self, initial_state, vehicle_id):
        upd = DroneStateUpdate(vehicle_id="wrong_drone", source=DataSource.MAVLINK)
        with pytest.raises(FactoryError, match="mismatch"):
            StateFactory.merge_update(initial_state, upd)

    def test_merge_increments_sequence(self, initial_state, partial_update):
        merged = StateFactory.merge_update(initial_state, partial_update)
        assert merged.sequence == initial_state.sequence + 1
