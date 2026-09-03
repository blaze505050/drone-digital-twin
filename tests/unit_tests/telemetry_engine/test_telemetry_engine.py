"""
Tests for drone_sdk.telemetry_engine (Module 2).
Covers: TelemetrySource, SourceArbiter, TelemetryEngine, recorder/replayer stubs.
"""
from __future__ import annotations

import threading
import time
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from drone_sdk.state_manager import (
    DataSource,
    DroneStateUpdate,
    StateFactory,
    StateStore,
    VehicleConfig,
)
from drone_sdk.telemetry_engine import (
    CallableSource,
    DuplicateSourceError,
    EngineAlreadyRunningError,
    EngineConfig,
    NullSource,
    SourceArbiter,
    SourceNotFoundError,
    SourcePriority,
    SourceStatus,
    TelemetryEngine,
)
from drone_sdk.telemetry_engine.exceptions import ReplayError


# ── Fixtures ──────────────────────────────────────────────────────────────────

VEHICLE = "test_drone_telemetry"


@pytest.fixture(autouse=True)
def clean_stores():
    StateStore.destroy_all()
    yield
    StateStore.destroy_all()


@pytest.fixture
def store():
    return StateStore.create(VEHICLE)


@pytest.fixture
def null_source():
    src = NullSource(
        source_id  = "null_0",
        vehicle_id = VEHICLE,
        priority   = SourcePriority.SIMULATION,
        data_source= DataSource.MANUAL,
    )
    return src


@pytest.fixture
def counting_source():
    """A CallableSource that counts how many times poll() is called."""
    state = {"count": 0}

    def poll_fn() -> list:
        state["count"] += 1
        upd = DroneStateUpdate(VEHICLE, DataSource.SITL)
        upd.position = np.array([float(state["count"]), 0.0, -5.0])
        return [upd]

    src = CallableSource(
        source_id  = "counter_0",
        vehicle_id = VEHICLE,
        poll_fn    = poll_fn,
        priority   = SourcePriority.SIMULATION,
    )
    src._poll_state = state   # expose for assertions
    return src


@pytest.fixture
def engine():
    cfg = EngineConfig(
        tick_rate_hz         = 200.0,
        auto_create_stores   = True,
        max_updates_per_tick = 100,
    )
    e = TelemetryEngine(config=cfg)
    yield e
    if e.is_running:
        e.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  TelemetrySource (abstract interface & NullSource)
# ══════════════════════════════════════════════════════════════════════════════

class TestNullSource:
    def test_initial_status_disconnected(self, null_source):
        assert null_source.status == SourceStatus.DISCONNECTED

    def test_connect_sets_connected(self, null_source):
        null_source.connect()
        assert null_source.status == SourceStatus.CONNECTED
        assert null_source.is_connected

    def test_disconnect_sets_disconnected(self, null_source):
        null_source.connect()
        null_source.disconnect()
        assert null_source.status == SourceStatus.DISCONNECTED

    def test_poll_returns_empty(self, null_source):
        null_source.connect()
        assert null_source.poll() == []

    def test_properties(self, null_source):
        assert null_source.source_id  == "null_0"
        assert null_source.vehicle_id == VEHICLE
        assert null_source.priority   == SourcePriority.SIMULATION

    def test_repr_contains_id(self, null_source):
        assert "null_0" in repr(null_source)

    def test_get_info_returns_source_info(self, null_source):
        null_source.connect()
        info = null_source.get_info()
        assert info.source_id  == "null_0"
        assert info.status     == SourceStatus.CONNECTED
        assert info.total_packets == 0

    def test_record_poll_updates_metrics(self, null_source):
        null_source.connect()
        null_source._record_poll(10)
        info = null_source.get_info()
        assert info.total_packets == 10


class TestCallableSource:
    def test_poll_calls_function(self, counting_source):
        counting_source.connect()
        updates = counting_source.poll()
        assert len(updates) == 1
        assert counting_source._poll_state["count"] == 1

    def test_poll_error_returns_empty_and_records_error(self):
        def bad_poll():
            raise RuntimeError("simulated sensor fault")

        src = CallableSource("bad", VEHICLE, bad_poll, SourcePriority.SIMULATION)
        src.connect()
        result = src.poll()
        assert result == []
        assert src.get_info().total_errors == 1

    def test_update_has_correct_vehicle(self, counting_source):
        counting_source.connect()
        updates = counting_source.poll()
        assert updates[0].vehicle_id == VEHICLE

    def test_info_drop_rate_zero_initially(self, counting_source):
        counting_source.connect()
        counting_source.poll()
        info = counting_source.get_info()
        assert info.drop_rate == 0.0

    def test_info_to_dict(self, counting_source):
        info = counting_source.get_info()
        d    = info.to_dict()
        assert "source_id"    in d
        assert "status"       in d
        assert "total_packets" in d


# ══════════════════════════════════════════════════════════════════════════════
#  SourceArbiter
# ══════════════════════════════════════════════════════════════════════════════

class TestSourceArbiter:

    def make_update(self, src_id: str) -> DroneStateUpdate:
        upd = DroneStateUpdate(VEHICLE, DataSource.MAVLINK)
        upd.position = np.array([1.0, 0.0, -5.0])
        return upd

    def test_empty_input_returns_empty(self):
        arbiter = SourceArbiter()
        assert arbiter.arbitrate({}) == []

    def test_single_source_passthrough(self):
        arbiter = SourceArbiter()
        arbiter.register("hw", SourcePriority.HARDWARE)
        upd = self.make_update("hw")
        result = arbiter.arbitrate({"hw": [upd]})
        assert result == [upd]

    def test_higher_priority_wins(self):
        arbiter = SourceArbiter()
        arbiter.register("hw",   SourcePriority.HARDWARE)
        arbiter.register("sitl", SourcePriority.SITL)

        hw_upd   = self.make_update("hw")
        sitl_upd = self.make_update("sitl")

        result = arbiter.arbitrate({
            "hw":   [hw_upd],
            "sitl": [sitl_upd],
        })
        assert hw_upd in result
        assert sitl_upd not in result

    def test_deregister_removes_source(self):
        arbiter = SourceArbiter()
        arbiter.register("hw", SourcePriority.HARDWARE)
        existed = arbiter.deregister("hw")
        assert existed is True

    def test_deregister_nonexistent_returns_false(self):
        arbiter = SourceArbiter()
        assert arbiter.deregister("ghost") is False

    def test_failover_when_primary_absent(self):
        """If the top-priority source has no data this tick, fall back."""
        arbiter = SourceArbiter(failover_threshold_ms=50.0)
        arbiter.register("hw",   SourcePriority.HARDWARE)
        arbiter.register("sitl", SourcePriority.SITL)

        # Simulate: hw was seen 200 ms ago (stale), sitl is fresh
        arbiter._sources["hw"].last_seen_mono = time.monotonic() - 0.200

        sitl_upd = self.make_update("sitl")
        result = arbiter.arbitrate({"sitl": [sitl_upd]})
        assert sitl_upd in result

    def test_get_source_info_returns_list(self):
        arbiter = SourceArbiter()
        arbiter.register("s1", SourcePriority.HARDWARE)
        info = arbiter.get_source_info()
        assert isinstance(info, list)
        assert len(info) == 1
        assert info[0]["source_id"] == "s1"

    def test_unregistered_source_passes_through(self):
        """Updates from sources unknown to the arbiter should pass through."""
        arbiter = SourceArbiter()
        upd = self.make_update("unknown_src")
        result = arbiter.arbitrate({"unknown_src": [upd]})
        assert upd in result


# ══════════════════════════════════════════════════════════════════════════════
#  TelemetryEngine
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineLifecycle:
    def test_initial_not_running(self, engine):
        assert not engine.is_running

    def test_start_sets_running(self, engine):
        engine.start()
        assert engine.is_running
        engine.stop()

    def test_double_start_raises(self, engine):
        engine.start()
        with pytest.raises(EngineAlreadyRunningError):
            engine.start()
        engine.stop()

    def test_stop_not_running_returns_true(self, engine):
        assert engine.stop() is True

    def test_repr_contains_running_state(self, engine):
        assert "running=" in repr(engine)

    def test_add_source(self, engine, null_source):
        engine.add_source(null_source)
        assert "null_0" in engine.list_sources()

    def test_add_duplicate_source_raises(self, engine, null_source):
        engine.add_source(null_source)
        with pytest.raises(DuplicateSourceError):
            engine.add_source(null_source)

    def test_remove_source(self, engine, null_source):
        engine.add_source(null_source)
        engine.remove_source("null_0", disconnect=False)
        assert "null_0" not in engine.list_sources()

    def test_remove_nonexistent_raises(self, engine):
        with pytest.raises(SourceNotFoundError):
            engine.remove_source("ghost")

    def test_get_source(self, engine, null_source):
        engine.add_source(null_source)
        assert engine.get_source("null_0") is null_source

    def test_get_source_missing_raises(self, engine):
        with pytest.raises(SourceNotFoundError):
            engine.get_source("ghost")

    def test_connect_all(self, engine, null_source):
        engine.add_source(null_source)
        results = engine.connect_all()
        assert results["null_0"] is True
        assert null_source.is_connected


class TestEnginePushesStateToStore:
    def test_updates_reach_state_store(self, engine, store, counting_source):
        counting_source.connect()
        engine.add_source(counting_source)
        engine.start()
        time.sleep(0.08)   # Allow a few ticks
        engine.stop()
        latest = store.get_latest()
        assert latest is not None
        assert latest.x > 0   # position was updated from poll

    def test_auto_creates_store_for_unknown_vehicle(self, engine):
        upd = DroneStateUpdate("brand_new_drone", DataSource.SITL)
        upd.position = np.array([1.0, 2.0, -3.0])
        engine.inject_update(upd)
        assert StateStore.exists("brand_new_drone")

    def test_inject_update_reaches_store(self, engine, store):
        upd = DroneStateUpdate(VEHICLE, DataSource.MANUAL)
        upd.battery_voltage = 15.4
        engine.inject_update(upd)
        latest = store.get_latest()
        assert latest is not None
        assert abs(latest.battery_voltage - 15.4) < 1e-6

    def test_engine_metrics_tick_rate(self, engine, null_source):
        null_source.connect()
        engine.add_source(null_source)
        engine.start()
        time.sleep(0.15)
        engine.stop()
        metrics = engine.get_metrics()
        # Should have fired at close to 200 Hz (allow ±50%)
        assert metrics.tick_rate_hz > 50.0

    def test_status_summary_keys(self, engine, null_source):
        engine.add_source(null_source)
        summary = engine.get_status_summary()
        assert "running"         in summary
        assert "tick_rate_hz"    in summary
        assert "total_updates"   in summary
        assert "sources"         in summary


class TestEngineTickHooks:
    def test_tick_hook_called(self, engine, store):
        calls = []
        engine.add_tick_hook(lambda: calls.append(1))
        engine.start()
        time.sleep(0.05)
        engine.stop()
        assert len(calls) > 0

    def test_remove_tick_hook(self, engine, store):
        calls = []
        fn = lambda: calls.append(1)
        engine.add_tick_hook(fn)
        removed = engine.remove_tick_hook(fn)
        assert removed is True
        engine.start()
        time.sleep(0.05)
        engine.stop()
        assert len(calls) == 0

    def test_hook_exception_does_not_crash_engine(self, engine, store):
        def bad_hook():
            raise RuntimeError("hook error")

        engine.add_tick_hook(bad_hook)
        engine.start()
        time.sleep(0.05)
        assert engine.is_running
        engine.stop()


class TestEngineConcurrency:
    def test_add_remove_source_while_running(self, engine, store):
        """Add and remove sources while the engine is running — no deadlock."""
        errors = []

        def mangler():
            for i in range(10):
                src = NullSource(f"dyn_{i}", VEHICLE, SourcePriority.MANUAL)
                try:
                    engine.add_source(src, auto_connect=True)
                    time.sleep(0.01)
                    engine.remove_source(f"dyn_{i}", disconnect=True)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        engine.start()
        t = threading.Thread(target=mangler)
        t.start()
        t.join(timeout=10.0)
        engine.stop()
        assert errors == []

    def test_multi_source_concurrent_poll(self, engine, store):
        """Multiple fast sources pushing concurrently."""
        n_sources = 5
        for i in range(n_sources):
            idx = i

            def make_poll(i=idx):
                def poll_fn():
                    upd = DroneStateUpdate(VEHICLE, DataSource.SITL)
                    upd.position = np.array([float(i), 0.0, -5.0])
                    return [upd]
                return poll_fn

            src = CallableSource(
                f"src_{i}", VEHICLE,
                poll_fn=make_poll(i),
                priority=SourcePriority.SIMULATION,
            )
            engine.add_source(src, auto_connect=True)

        engine.start()
        time.sleep(0.12)
        engine.stop()
        # Store should have received updates
        assert store.history_length > 0


# ══════════════════════════════════════════════════════════════════════════════
#  TelemetryReplayer (no-h5py guard)
# ══════════════════════════════════════════════════════════════════════════════

class TestTelemetryReplayer:
    def test_missing_file_raises(self):
        from drone_sdk.telemetry_engine import TelemetryReplayer
        with pytest.raises(ReplayError, match="not found"):
            TelemetryReplayer("/tmp/nonexistent_telemetry_12345.h5")

    def test_attach_store(self, tmp_path, store):
        """iter_states on empty file returns nothing (no crash)."""
        from drone_sdk.telemetry_engine import TelemetryReplayer

        # Create a valid but empty HDF5
        try:
            import h5py
            p = tmp_path / "empty.h5"
            with h5py.File(p, "w") as f:
                f.attrs["test"] = 1

            replayer = TelemetryReplayer(p, speed=float("inf"))
            replayer.attach_store(VEHICLE)
            states = list(replayer.iter_states(VEHICLE, speed=float("inf")))
            assert states == []
        except ImportError:
            pytest.skip("h5py not installed")


# ══════════════════════════════════════════════════════════════════════════════
#  TelemetryRecorder (no-h5py guard)
# ══════════════════════════════════════════════════════════════════════════════

class TestTelemetryRecorder:
    def test_start_without_h5py_raises(self, store, tmp_path):
        from drone_sdk.telemetry_engine import TelemetryRecorder
        from drone_sdk.telemetry_engine.exceptions import RecordingError

        # Temporarily remove h5py if present
        import sys
        original = sys.modules.get("h5py")
        sys.modules["h5py"] = None  # type: ignore
        try:
            recorder = TelemetryRecorder(tmp_path / "log.h5")
            with pytest.raises((RecordingError, ImportError, TypeError)):
                recorder.start()
        finally:
            if original is not None:
                sys.modules["h5py"] = original
            else:
                del sys.modules["h5py"]

    def test_full_record_replay_roundtrip(self, store, tmp_path):
        """Write states → HDF5 → read back → verify."""
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not installed")

        from drone_sdk.telemetry_engine import TelemetryRecorder, TelemetryReplayer

        log_path = tmp_path / "test_flight.h5"

        # Write 20 states
        recorder = TelemetryRecorder(
            log_path,
            flush_interval_s=0.1,
            flush_count=5,
        )
        recorder.attach_store(VEHICLE)
        recorder.start()

        initial = StateFactory.create_initial(VEHICLE)
        for i in range(20):
            s = initial.copy_with(x=float(i), y=float(i * 2), z=-5.0)
            store.update(s)
            time.sleep(0.01)

        n_written = recorder.stop()
        assert n_written >= 10   # Allow for some timing slack

        # Read back and verify
        replayer = TelemetryReplayer(log_path, speed=float("inf"))
        states = list(replayer.iter_states(VEHICLE))
        assert len(states) >= 10
        # X values should be monotonically increasing
        xs = [s.x for s in states]
        assert xs == sorted(xs)
