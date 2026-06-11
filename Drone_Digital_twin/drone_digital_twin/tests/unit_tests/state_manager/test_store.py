"""Tests for state_manager.store — including concurrency."""
from __future__ import annotations

import threading
import time
from typing import List

import numpy as np
import pytest

from drone_sdk.state_manager import (
    ArmingState,
    DataSource,
    DroneStateUpdate,
    DroneStateVector,
    DuplicateStoreError,
    EventType,
    FlightMode,
    StateFactory,
    StateStore,
    StoreNotFoundError,
    VehicleConfig,
)


# ── Singleton lifecycle ───────────────────────────────────────────────────────

class TestSingletonLifecycle:
    def test_create_returns_store(self, vehicle_id, config):
        store = StateStore.create(vehicle_id, config)
        assert isinstance(store, StateStore)
        assert store.vehicle_id == vehicle_id

    def test_duplicate_create_raises(self, vehicle_id, config):
        StateStore.create(vehicle_id, config)
        with pytest.raises(DuplicateStoreError) as exc_info:
            StateStore.create(vehicle_id, config)
        assert vehicle_id in str(exc_info.value)

    def test_get_instance_returns_same_object(self, store, vehicle_id):
        retrieved = StateStore.get_instance(vehicle_id)
        assert retrieved is store

    def test_get_instance_missing_raises(self):
        with pytest.raises(StoreNotFoundError):
            StateStore.get_instance("nonexistent_drone")

    def test_get_or_create_returns_existing(self, store, vehicle_id):
        retrieved = StateStore.get_or_create(vehicle_id)
        assert retrieved is store

    def test_get_or_create_creates_if_missing(self):
        new_id = "fresh_drone"
        store = StateStore.get_or_create(new_id)
        assert isinstance(store, StateStore)
        assert StateStore.exists(new_id)

    def test_destroy_removes_from_registry(self, vehicle_id, config):
        StateStore.create(vehicle_id, config)
        assert StateStore.exists(vehicle_id)
        destroyed = StateStore.destroy(vehicle_id)
        assert destroyed is True
        assert not StateStore.exists(vehicle_id)

    def test_destroy_nonexistent_returns_false(self):
        assert StateStore.destroy("ghost_drone") is False

    def test_destroy_all_clears_registry(self, config):
        StateStore.create("d1", config)
        StateStore.create("d2", VehicleConfig("d2"))
        count = StateStore.destroy_all()
        assert count == 2
        assert StateStore.list_vehicles() == []

    def test_list_vehicles(self, config):
        StateStore.create("alpha", config)
        StateStore.create("beta", VehicleConfig("beta"))
        ids = StateStore.list_vehicles()
        assert "alpha" in ids
        assert "beta" in ids

    def test_exists(self, vehicle_id, config):
        assert not StateStore.exists(vehicle_id)
        StateStore.create(vehicle_id, config)
        assert StateStore.exists(vehicle_id)

    def test_update_after_shutdown_raises(self, store, initial_state):
        store.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            store.update(initial_state)


# ── State write and read ──────────────────────────────────────────────────────

class TestStateWriteRead:
    def test_get_latest_none_before_first_update(self, store):
        assert store.get_latest() is None

    def test_update_stores_state(self, store, initial_state):
        store.update(initial_state)
        latest = store.get_latest()
        assert latest is not None
        assert latest.vehicle_id == initial_state.vehicle_id

    def test_update_stamps_sequence(self, store, initial_state):
        store.update(initial_state)
        store.update(initial_state)
        latest = store.get_latest()
        assert latest.sequence == 1   # second update → sequence 1

    def test_update_returns_true_for_valid_state(self, store, hover_state):
        result = store.update(hover_state)
        assert result is True

    def test_update_returns_false_for_invalid_state(self, store, initial_state):
        # Create a state with invalid quaternion
        bad = initial_state.copy_with(q0=0.0, q1=0.0, q2=0.0, q3=0.0)
        result = store.update(bad)
        assert result is False

    def test_invalid_state_still_stored(self, store, initial_state):
        bad = initial_state.copy_with(q0=0.0, q1=0.0, q2=0.0, q3=0.0)
        store.update(bad)
        latest = store.get_latest()
        assert latest is not None
        assert latest.is_valid is False

    def test_update_stamps_health_score(self, store, initial_state):
        for _ in range(60):   # Enough updates to build up EMA
            store.update(initial_state)
        latest = store.get_latest()
        assert 0.0 <= latest.health_score <= 1.0

    def test_update_partial_with_no_existing_state(self, store, partial_update):
        store.update_partial(partial_update)
        latest = store.get_latest()
        assert latest is not None
        assert abs(latest.battery_voltage - partial_update.battery_voltage) < 1e-9

    def test_update_partial_merges_correctly(self, store, hover_state, partial_update):
        store.update(hover_state)
        store.update_partial(partial_update)
        latest = store.get_latest()
        # Battery updated
        assert abs(latest.battery_voltage - partial_update.battery_voltage) < 1e-9
        # Position preserved from hover_state
        assert latest.x == hover_state.x
        assert latest.altitude_agl == hover_state.altitude_agl

    def test_sequence_increments_monotonically(self, store, initial_state):
        seqs = []
        for _ in range(10):
            store.update(initial_state)
            seqs.append(store.get_latest().sequence)
        assert seqs == sorted(seqs)
        assert seqs == list(range(len(seqs)))


# ── History ───────────────────────────────────────────────────────────────────

class TestHistory:
    def test_history_empty_before_updates(self, store):
        assert store.get_history() == []

    def test_history_grows_with_updates(self, store, initial_state):
        for _ in range(5):
            store.update(initial_state)
        assert len(store.get_history()) == 5

    def test_history_respects_ring_buffer(self, vehicle_id):
        cfg = VehicleConfig(vehicle_id=vehicle_id, history_size=10)
        s = StateStore.create(vehicle_id, cfg)
        state = StateFactory.create_initial(vehicle_id)
        for _ in range(25):
            s.update(state)
        assert len(s.get_history()) == 10
        assert s.history_length == 10

    def test_get_history_n_returns_last_n(self, store, initial_state):
        for _ in range(20):
            store.update(initial_state)
        last5 = store.get_history(n=5)
        assert len(last5) == 5

    def test_get_history_window_seconds(self, store, initial_state):
        for _ in range(10):
            store.update(initial_state)
            time.sleep(0.01)   # 10 ms apart
        window = store.get_history(window_seconds=0.08)
        # Should have roughly 8 states, allow ±2 for timing jitter
        assert 5 <= len(window) <= 10

    def test_get_history_chronological_order(self, store, initial_state):
        for i in range(5):
            s = initial_state.copy_with(x=float(i))
            store.update(s)
        history = store.get_history()
        xs = [s.x for s in history]
        assert xs == sorted(xs)

    def test_get_history_numpy_returns_arrays(self, store, hover_state):
        for _ in range(5):
            store.update(hover_state)
        arrays = store.get_history_numpy(["x", "y", "z"])
        assert "x" in arrays
        assert isinstance(arrays["x"], np.ndarray)
        assert len(arrays["x"]) == 5

    def test_get_history_numpy_empty_store(self, store):
        arrays = store.get_history_numpy(["x", "y"])
        assert len(arrays["x"]) == 0

    def test_get_history_numpy_default_fields(self, store, hover_state):
        for _ in range(3):
            store.update(hover_state)
        arrays = store.get_history_numpy()
        assert "timestamp_wall" in arrays
        assert "x" in arrays
        assert "battery_soc" in arrays

    def test_reset_clears_history(self, store, initial_state):
        for _ in range(10):
            store.update(initial_state)
        assert store.history_length == 10
        store.reset()
        assert store.history_length == 0
        assert store.get_latest() is None
        assert store.sequence == 0


# ── Events ────────────────────────────────────────────────────────────────────

class TestStoreEvents:
    def test_state_updated_event_fires(self, store, initial_state):
        received = []
        store.subscribe(EventType.STATE_UPDATED, lambda e: received.append(e))
        store.update(initial_state)
        assert len(received) == 1
        assert received[0].event_type == EventType.STATE_UPDATED

    def test_state_invalid_event_fires_on_bad_state(self, store, initial_state):
        received = []
        store.subscribe(EventType.STATE_INVALID, lambda e: received.append(e))
        bad = initial_state.copy_with(q0=0.0, q1=0.0, q2=0.0, q3=0.0)
        store.update(bad)
        assert len(received) == 1

    def test_vehicle_armed_event_fires(self, store, initial_state):
        received = []
        store.subscribe(EventType.VEHICLE_ARMED, lambda e: received.append(e))
        store.update(initial_state)
        armed = initial_state.copy_with(arming_state=ArmingState.ARMED)
        store.update(armed)
        assert len(received) == 1

    def test_vehicle_disarmed_event_fires(self, store, initial_state):
        received = []
        store.subscribe(EventType.VEHICLE_DISARMED, lambda e: received.append(e))
        armed = initial_state.copy_with(arming_state=ArmingState.ARMED)
        store.update(armed)
        store.update(initial_state)   # back to DISARMED
        assert len(received) == 1

    def test_flight_mode_changed_event_fires(self, store, initial_state):
        received = []
        store.subscribe(EventType.FLIGHT_MODE_CHANGED, lambda e: received.append(e))
        store.update(initial_state)
        new_mode = initial_state.copy_with(flight_mode=FlightMode.POSITION_HOLD)
        store.update(new_mode)
        assert len(received) == 1
        old_m, new_m = received[0].data
        assert new_m == FlightMode.POSITION_HOLD

    def test_store_reset_event_fires(self, store, initial_state):
        received = []
        store.subscribe(EventType.STORE_RESET, lambda e: received.append(e))
        store.update(initial_state)
        store.reset()
        assert len(received) == 1

    def test_unsubscribe_stops_delivery(self, store, initial_state):
        received = []
        sub_id = store.subscribe(EventType.STATE_UPDATED, lambda e: received.append(e))
        store.update(initial_state)
        store.unsubscribe(sub_id)
        store.update(initial_state)
        assert len(received) == 1   # only the first one

    def test_subscriber_exception_does_not_crash_store(self, store, initial_state):
        def bad_callback(e):
            raise ValueError("intentional test error")

        store.subscribe(EventType.STATE_UPDATED, bad_callback)
        # Should not raise
        store.update(initial_state)
        assert store.get_latest() is not None


# ── Thread safety ─────────────────────────────────────────────────────────────

class TestThreadSafety:
    """Stress-test concurrent reads and writes."""

    def test_concurrent_writes(self, store, initial_state):
        errors: List[Exception] = []
        n_threads = 10
        n_writes_each = 50

        def writer():
            try:
                for i in range(n_writes_each):
                    s = initial_state.copy_with(x=float(i))
                    store.update(s)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert errors == [], f"Thread errors: {errors}"
        assert store.history_length > 0

    def test_concurrent_reads_and_writes(self, store, initial_state):
        errors: List[Exception] = []
        stop_event = threading.Event()

        def writer():
            try:
                for i in range(100):
                    store.update(initial_state.copy_with(x=float(i)))
                    time.sleep(0.001)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def reader():
            try:
                while not stop_event.is_set():
                    state = store.get_latest()
                    if state is not None:
                        _ = state.x   # Access a field
                    time.sleep(0.0005)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        writer_thread = threading.Thread(target=writer)
        reader_threads = [threading.Thread(target=reader) for _ in range(4)]

        for t in reader_threads:
            t.start()
        writer_thread.start()
        writer_thread.join(timeout=10.0)
        stop_event.set()
        for t in reader_threads:
            t.join(timeout=5.0)

        assert errors == [], f"Concurrent read/write errors: {errors}"

    def test_concurrent_subscribe_unsubscribe(self, store, initial_state):
        """Subscribing and unsubscribing while events are firing must not deadlock."""
        errors: List[Exception] = []
        stop_event = threading.Event()
        sub_ids: List[str] = []
        lock = threading.Lock()

        def subscriber_manager():
            try:
                while not stop_event.is_set():
                    sid = store.subscribe(EventType.STATE_UPDATED, lambda e: None)
                    with lock:
                        sub_ids.append(sid)
                    time.sleep(0.002)
                    with lock:
                        if sub_ids:
                            old_sid = sub_ids.pop(0)
                    store.unsubscribe(old_sid)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def state_writer():
            for _ in range(50):
                store.update(initial_state)
                time.sleep(0.002)

        writer = threading.Thread(target=state_writer)
        manager = threading.Thread(target=subscriber_manager)

        manager.start()
        writer.start()
        writer.join(timeout=10.0)
        stop_event.set()
        manager.join(timeout=5.0)

        assert errors == [], f"Subscribe/unsubscribe concurrency errors: {errors}"


# ── Health and status ─────────────────────────────────────────────────────────

class TestHealthAndStatus:
    def test_is_healthy_after_sufficient_updates(self, store, initial_state):
        for _ in range(60):
            store.update(initial_state)
        # After 60 valid updates, health should be NOMINAL
        assert store.is_healthy()

    def test_is_stale_before_any_update(self, store):
        assert store.is_stale()

    def test_get_status_summary_keys(self, store, initial_state):
        store.update(initial_state)
        summary = store.get_status_summary()
        assert "vehicle_id"     in summary
        assert "health"         in summary
        assert "current_state"  in summary
        assert summary["vehicle_id"] == store.vehicle_id

    def test_repr_contains_vehicle_id(self, store):
        assert store.vehicle_id in repr(store)
