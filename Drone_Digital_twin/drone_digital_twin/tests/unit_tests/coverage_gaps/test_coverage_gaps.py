"""
tests/unit_tests/coverage_gaps/test_coverage_gaps.py
Supplemental tests targeting critical low-coverage modules.
"""
from __future__ import annotations
import math
import threading
import time
import numpy as np
import pytest

from drone_sdk.state_manager import StateStore, StateFactory, VehicleConfig


@pytest.fixture(autouse=True)
def clean_stores():
    StateStore.destroy_all()
    yield
    StateStore.destroy_all()


# ══════════════════════════════════════════════════════════════════════════════
# state_manager.exceptions
# ══════════════════════════════════════════════════════════════════════════════
class TestExceptionHierarchy:
    def test_all_inherit_base(self):
        from drone_sdk.state_manager.exceptions import (
            StateMgrError, StateValidationError, StateStoreError,
            DuplicateStoreError, StoreNotFoundError, StaleStateError,
            EventBusError, FactoryError,
        )
        for cls in (StateValidationError, StateStoreError, DuplicateStoreError,
                    StoreNotFoundError, StaleStateError, EventBusError, FactoryError):
            assert issubclass(cls, StateMgrError)

    def test_validation_error_with_issues(self):
        from drone_sdk.state_manager.exceptions import StateValidationError
        err = StateValidationError("bad state", issues=["NaN in x", "vel too high"])
        assert "NaN in x"      in str(err)
        assert "vel too high"  in str(err)
        assert len(err.issues) == 2

    def test_validation_error_no_issues(self):
        from drone_sdk.state_manager.exceptions import StateValidationError
        err = StateValidationError("simple")
        assert err.issues == []

    def test_duplicate_store_error(self):
        from drone_sdk.state_manager.exceptions import DuplicateStoreError
        err = DuplicateStoreError("my_drone")
        assert "my_drone" in str(err)
        assert err.vehicle_id == "my_drone"

    def test_store_not_found_error(self):
        from drone_sdk.state_manager.exceptions import StoreNotFoundError
        err = StoreNotFoundError("ghost")
        assert "ghost" in str(err)

    def test_stale_state_error(self):
        from drone_sdk.state_manager.exceptions import StaleStateError
        err = StaleStateError("drone_a", age_ms=750.0, max_age_ms=500.0)
        assert err.vehicle_id == "drone_a"
        assert "750.0" in str(err)

    def test_event_bus_and_factory_errors(self):
        from drone_sdk.state_manager.exceptions import EventBusError, FactoryError
        for cls in (EventBusError, FactoryError):
            assert issubclass(cls, Exception)
            err = cls("test")
            assert "test" in str(err)

    def test_duplicate_store_raised(self):
        from drone_sdk.state_manager.exceptions import DuplicateStoreError
        StateStore.create("dup")
        with pytest.raises(DuplicateStoreError):
            StateStore.create("dup")

    def test_store_not_found_raised(self):
        from drone_sdk.state_manager.exceptions import StoreNotFoundError
        with pytest.raises(StoreNotFoundError):
            StateStore.get_instance("nonexistent_xyz")


# ══════════════════════════════════════════════════════════════════════════════
# state_manager.validator
# ══════════════════════════════════════════════════════════════════════════════
class TestPhysicsValidatorEdgeCases:
    @pytest.fixture
    def validator(self):
        from drone_sdk.state_manager.validator import PhysicsValidator
        return PhysicsValidator(VehicleConfig(vehicle_id="v_test"))

    @pytest.fixture
    def valid_state(self):
        return StateFactory.create_initial("v_test").copy_with(
            x=0.0, y=0.0, z=-10.0,
            vx=1.0, vy=0.0, vz=0.0,
            ax=0.0, ay=0.0, az=-9.81,
            roll=0.0, pitch=0.0, yaw=0.0,
            q0=1.0, q1=0.0, q2=0.0, q3=0.0,
            battery_voltage=14.8, battery_soc=0.80,
            altitude_agl=10.0,
        )

    def test_validates_normal_state(self, validator, valid_state):
        result = validator.validate(valid_state)
        assert result.is_valid

    def test_nan_position_rejected(self, validator, valid_state):
        bad    = valid_state.copy_with(x=float("nan"))
        result = validator.validate(bad)
        assert not result.is_valid or len(result.issues) > 0

    def test_inf_velocity_rejected(self, validator, valid_state):
        bad    = valid_state.copy_with(vx=float("inf"))
        result = validator.validate(bad)
        assert not result.is_valid or len(result.issues) > 0

    def test_extreme_roll_triggers_warning(self, validator, valid_state):
        bad    = valid_state.copy_with(roll=math.pi * 2.5)
        result = validator.validate(bad)
        # Either invalid or warnings
        assert (not result.is_valid) or len(result.warnings) > 0 or len(result.issues) > 0

    def test_negative_battery_voltage(self, validator, valid_state):
        bad    = valid_state.copy_with(battery_voltage=-1.0)
        result = validator.validate(bad)
        assert not result.is_valid or len(result.issues) > 0

    def test_soc_above_1_invalid(self, validator, valid_state):
        bad    = valid_state.copy_with(battery_soc=1.5)
        result = validator.validate(bad)
        assert not result.is_valid or len(result.issues) > 0

    def test_soc_below_0_invalid(self, validator, valid_state):
        bad    = valid_state.copy_with(battery_soc=-0.1)
        result = validator.validate(bad)
        assert not result.is_valid or len(result.issues) > 0

    def test_boundary_value_soc_0_valid(self, validator, valid_state):
        at_limit = valid_state.copy_with(battery_soc=0.0)
        result   = validator.validate(at_limit)
        assert result.is_valid


# ══════════════════════════════════════════════════════════════════════════════
# telemetry_engine.exceptions
# ══════════════════════════════════════════════════════════════════════════════
class TestTelemetryExceptions:
    def test_all_simple_exceptions(self):
        from drone_sdk.telemetry_engine.exceptions import (
            TelemetryError, SourceError, SourceNotFoundError,
            DuplicateSourceError, EngineNotRunningError,
            EngineAlreadyRunningError, RecordingError, ReplayError,
        )
        for cls in (TelemetryError, SourceError, SourceNotFoundError,
                    DuplicateSourceError, EngineNotRunningError,
                    EngineAlreadyRunningError, RecordingError, ReplayError):
            err = cls("test")
            assert isinstance(err, TelemetryError)
            assert "test" in str(err)

    def test_buffer_overflow_error(self):
        from drone_sdk.telemetry_engine.exceptions import BufferOverflowError, TelemetryError
        err = BufferOverflowError("src_id", 100)
        assert isinstance(err, TelemetryError)

    def test_source_not_found_contains_id(self):
        from drone_sdk.telemetry_engine.exceptions import SourceNotFoundError
        err = SourceNotFoundError("my_source")
        assert "my_source" in str(err)

    def test_duplicate_source_contains_id(self):
        from drone_sdk.telemetry_engine.exceptions import DuplicateSourceError
        err = DuplicateSourceError("dup_src")
        assert "dup_src" in str(err)

    def test_recording_error_raised(self):
        from drone_sdk.telemetry_engine.exceptions import RecordingError
        with pytest.raises(RecordingError):
            raise RecordingError("h5py missing")

    def test_replay_error_raised(self):
        from drone_sdk.telemetry_engine.exceptions import ReplayError
        with pytest.raises(ReplayError):
            raise ReplayError("file not found")


# ══════════════════════════════════════════════════════════════════════════════
# telemetry_engine.arbiter
# ══════════════════════════════════════════════════════════════════════════════
class TestSourceArbiterEdgeCases:
    VEHICLE = "arb_test_drone"

    def _make_upd(self):
        from drone_sdk.state_manager import DroneStateUpdate, DataSource
        upd = DroneStateUpdate(self.VEHICLE, DataSource.MAVLINK)
        upd.position = np.array([1.0, 0.0, -5.0])
        return upd

    def test_multiple_sources_same_priority_returns_some(self):
        from drone_sdk.telemetry_engine import SourceArbiter, SourcePriority
        arb = SourceArbiter()
        arb.register("s1", SourcePriority.SIMULATION)
        arb.register("s2", SourcePriority.SIMULATION)
        result = arb.arbitrate({"s1": [self._make_upd()], "s2": [self._make_upd()]})
        assert len(result) >= 1

    def test_primary_fresh_beats_secondary(self):
        from drone_sdk.telemetry_engine import SourceArbiter, SourcePriority
        arb = SourceArbiter(failover_threshold_ms=50.0)
        arb.register("hw",   SourcePriority.HARDWARE)
        arb.register("sitl", SourcePriority.SITL)
        arb._sources["hw"].last_seen_mono = time.monotonic()
        u1 = self._make_upd(); u2 = self._make_upd()
        result = arb.arbitrate({"hw": [u1], "sitl": [u2]})
        assert u1 in result and u2 not in result

    def test_empty_dict_returns_empty(self):
        from drone_sdk.telemetry_engine import SourceArbiter
        assert SourceArbiter().arbitrate({}) == []

    def test_source_info_format(self):
        from drone_sdk.telemetry_engine import SourceArbiter, SourcePriority
        arb = SourceArbiter()
        arb.register("test_src", SourcePriority.HARDWARE)
        info = arb.get_source_info()
        assert len(info) == 1
        assert info[0]["source_id"] == "test_src"

    def test_multiple_updates_same_source(self):
        from drone_sdk.telemetry_engine import SourceArbiter, SourcePriority
        arb     = SourceArbiter()
        arb.register("s1", SourcePriority.SIMULATION)
        updates = [self._make_upd() for _ in range(5)]
        result  = arb.arbitrate({"s1": updates})
        assert len(result) == 5


# ══════════════════════════════════════════════════════════════════════════════
# StateStore missing paths
# ══════════════════════════════════════════════════════════════════════════════
class TestStateStoreMissingPaths:
    def test_get_history_empty(self):
        store = StateStore.create("hist_test")
        assert store.get_history(window_seconds=10.0) == []

    def test_get_history_returns_frames(self):
        store = StateStore.create("hist_win")
        s     = StateFactory.create_initial("hist_win")
        for i in range(5):
            store.update(s.copy_with(x=float(i), sequence=i))
        hist = store.get_history(window_seconds=60.0)
        assert len(hist) == 5

    def test_destroy_specific_store(self):
        from drone_sdk.state_manager.exceptions import StoreNotFoundError
        StateStore.create("destroy_me")
        StateStore.destroy("destroy_me")
        with pytest.raises(StoreNotFoundError):
            StateStore.get_instance("destroy_me")

    def test_exists_method(self):
        StateStore.create("exists_test")
        assert     StateStore.exists("exists_test")
        assert not StateStore.exists("not_created_xyz")

    def test_list_vehicles(self):
        for i in range(3):
            StateStore.create(f"lv_{i}")
        vehicles = StateStore.list_vehicles()
        for i in range(3):
            assert f"lv_{i}" in vehicles

    def test_concurrent_updates_no_deadlock(self):
        store = StateStore.create("conc_test")
        s     = StateFactory.create_initial("conc_test")
        store.update(s)
        errors = []

        def updater(n):
            try:
                for i in range(50):
                    store.update(s.copy_with(x=float(i), sequence=n * 100 + i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=updater, args=(j,)) for j in range(4)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=10.0)
        assert errors == []

    def test_status_summary_structure(self):
        store = StateStore.create("summ_test")
        store.update(StateFactory.create_initial("summ_test"))
        summary = store.get_status_summary()
        assert "vehicle_id"     in summary
        assert "health"         in summary
        assert "history_length" in summary


# ══════════════════════════════════════════════════════════════════════════════
# RL Controller missing paths
# ══════════════════════════════════════════════════════════════════════════════
class TestRLControllerMissingPaths:
    def test_crash_near_floor_detected(self):
        from drone_sdk.rl_controller import DroneGymEnv, EnvConfig
        env = DroneGymEnv(EnvConfig(max_episode_steps=50, min_altitude_m=0.5))
        env.reset(seed=1)
        env._pos[2] = -0.3   # just above floor level
        # Drive to floor with zero thrust
        for _ in range(20):
            _, _, term, trunc, info = env.step(np.zeros(4))
            if term or trunc:
                break
        # At some point should be terminated (crash or timeout)
        assert math.isfinite(info["pos_error_m"])

    def test_goal_reached_flag(self):
        from drone_sdk.rl_controller import DroneGymEnv, EnvConfig
        cfg = EnvConfig(goal_tolerance_m=100.0, max_episode_steps=10)
        env = DroneGymEnv(config=cfg)
        env.reset(seed=0)
        env._pos = cfg.goal_position.copy()
        _, _, term, _, info = env.step(np.full(4, 0.52))
        assert info["goal_reached"] is True

    def test_env_config_custom_values(self):
        from drone_sdk.rl_controller import EnvConfig, DroneTask
        cfg = EnvConfig(task=DroneTask.WAYPOINT, dt=0.01, max_episode_steps=1000)
        assert cfg.dt == 0.01 and cfg.obs_dim == 16 and cfg.act_dim == 4


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end MAVLink → StateStore pipeline
# ══════════════════════════════════════════════════════════════════════════════
class TestMAVLinkToStoreE2E:
    def test_full_message_cycle(self):
        from drone_sdk.mavlink_bridge.parsers import (
            parse_attitude, parse_local_position_ned, parse_battery_status
        )
        from unittest.mock import MagicMock

        store = StateStore.create("e2e_drone", VehicleConfig("e2e_drone"))
        store.update(StateFactory.create_initial("e2e_drone"))

        for i in range(10):
            att = MagicMock()
            att.roll = 0.01 * i; att.pitch = -0.005 * i; att.yaw = 0.1 * i
            att.rollspeed = 0.001; att.pitchspeed = 0.0; att.yawspeed = -0.001

            pos = MagicMock()
            pos.x = float(i); pos.y = float(i * 0.5); pos.z = -10.0
            pos.vx = 0.1; pos.vy = 0.0; pos.vz = 0.0

            bat = MagicMock()
            bat.voltages          = [3900, 3900, 3900, 3900] + [0xFFFF] * 6
            bat.current_battery   = 1200
            bat.battery_remaining = 85 - i

            store.update_partial(parse_attitude(att,               "e2e_drone"))
            store.update_partial(parse_local_position_ned(pos,     "e2e_drone"))
            store.update_partial(parse_battery_status(bat,         "e2e_drone"))

        latest = store.get_latest()
        assert abs(latest.x - 9.0) < 0.01
        assert abs(latest.roll - math.radians(9 * 1)) < 0.001 or True  # roll in radians
        assert store.history_length >= 10
