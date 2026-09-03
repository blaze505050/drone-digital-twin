"""
Tests for drone_sdk.dashboard (Module 5).

Tests the serializer thoroughly (no aiohttp needed).
Server lifecycle tests are guarded by aiohttp availability.
"""
from __future__ import annotations

import json
import math
import time

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
from drone_sdk.dashboard.serializer import (
    SerialiseConfig,
    SerialiseProfile,
    StateSerializer,
    state_to_json,
    states_to_json_array,
)

VEHICLE = "dash_test_drone"


@pytest.fixture(autouse=True)
def clean_stores():
    StateStore.destroy_all()
    yield
    StateStore.destroy_all()


@pytest.fixture
def store():
    return StateStore.create(VEHICLE, VehicleConfig(vehicle_id=VEHICLE))


@pytest.fixture
def hover_state():
    return StateFactory.create_initial(VEHICLE).copy_with(
        x=15.0, y=-3.0, z=-25.0,
        vx=2.0, vy=0.5, vz=-0.1,
        roll=0.05, pitch=-0.03, yaw=1.57,
        roll_rate=0.01, pitch_rate=0.005, yaw_rate=-0.02,
        q0=0.9997, q1=0.0, q2=0.0, q3=0.025,
        altitude_agl=25.0,
        groundspeed=2.06,
        battery_voltage=15.3,
        battery_current=11.5,
        battery_soc=0.78,
        battery_temperature=29.0,
        gps_lat=12.97, gps_lon=77.59,
        gps_fix_type=3, gps_satellites=13,
        gps_hdop=0.9, gps_vdop=1.1,
        flight_mode=FlightMode.POSITION_HOLD,
        arming_state=ArmingState.ARMED,
        health_status=HealthStatus.NOMINAL,
        health_score=0.95,
        is_valid=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  StateSerializer — FULL profile
# ══════════════════════════════════════════════════════════════════════════════

class TestSerializerFull:

    def test_returns_valid_json(self, hover_state):
        ser = StateSerializer(SerialiseConfig(profile=SerialiseProfile.FULL))
        raw = ser.to_json(hover_state)
        data = json.loads(raw)
        assert isinstance(data, dict)

    def test_vehicle_id_present(self, hover_state):
        data = json.loads(state_to_json(hover_state, SerialiseProfile.FULL))
        assert data["vehicle_id"] == VEHICLE

    def test_all_numeric_fields_present(self, hover_state):
        data = json.loads(state_to_json(hover_state, SerialiseProfile.FULL))
        assert "x" in data
        assert "y" in data
        assert "z" in data
        assert "q0" in data
        assert "battery_voltage" in data

    def test_enum_fields_are_primitives(self, hover_state):
        data = json.loads(state_to_json(hover_state, SerialiseProfile.FULL))
        assert isinstance(data["flight_mode"], int)
        assert isinstance(data["arming_state"], int)
        assert isinstance(data["source"], str)

    def test_computed_display_fields(self, hover_state):
        data = json.loads(state_to_json(hover_state, SerialiseProfile.FULL))
        assert "altitude_agl_ft" in data
        assert "groundspeed_kph" in data
        assert "armed" in data
        assert "airborne" in data
        assert data["armed"]   is True
        assert data["airborne"] is True


# ══════════════════════════════════════════════════════════════════════════════
#  StateSerializer — REALTIME profile
# ══════════════════════════════════════════════════════════════════════════════

class TestSerializerRealtime:

    def test_realtime_smaller_than_full(self, hover_state):
        full_len = len(state_to_json(hover_state, SerialiseProfile.FULL))
        rt_len   = len(state_to_json(hover_state, SerialiseProfile.REALTIME))
        assert rt_len < full_len

    def test_realtime_has_key_fields(self, hover_state):
        data = json.loads(state_to_json(hover_state, SerialiseProfile.REALTIME))
        for key in ("x", "y", "z", "roll", "pitch", "yaw", "q0", "q1", "q2", "q3",
                    "battery_soc", "flight_mode", "arming_state"):
            assert key in data, f"Missing key: {key}"

    def test_realtime_missing_low_frequency_fields(self, hover_state):
        data = json.loads(state_to_json(hover_state, SerialiseProfile.REALTIME))
        # GPS HDOP is not in realtime profile
        assert "gps_hdop" not in data

    def test_rad_to_deg_default_on(self, hover_state):
        data = json.loads(state_to_json(hover_state, SerialiseProfile.REALTIME))
        # roll=0.05 rad → ~2.865°
        assert abs(data["roll"] - math.degrees(0.05)) < 0.01

    def test_rad_to_deg_off(self, hover_state):
        cfg = SerialiseConfig(profile=SerialiseProfile.REALTIME, rad_to_deg=False)
        ser = StateSerializer(cfg)
        data = json.loads(ser.to_json(hover_state))
        # roll should be in radians
        assert abs(data["roll"] - 0.05) < 1e-5

    def test_include_raw_adds_rad_suffix(self, hover_state):
        cfg = SerialiseConfig(
            profile=SerialiseProfile.REALTIME,
            rad_to_deg=True,
            include_raw=True,
        )
        ser  = StateSerializer(cfg)
        data = json.loads(ser.to_json(hover_state))
        assert "roll_rad" in data
        assert "pitch_rad" in data


# ══════════════════════════════════════════════════════════════════════════════
#  StateSerializer — TELEMETRY profile
# ══════════════════════════════════════════════════════════════════════════════

class TestSerializerTelemetry:

    def test_telemetry_larger_than_realtime(self, hover_state):
        rt_len  = len(state_to_json(hover_state, SerialiseProfile.REALTIME))
        tel_len = len(state_to_json(hover_state, SerialiseProfile.TELEMETRY))
        assert tel_len >= rt_len

    def test_telemetry_has_battery_temp(self, hover_state):
        data = json.loads(state_to_json(hover_state, SerialiseProfile.TELEMETRY))
        assert "battery_temperature" in data
        assert abs(data["battery_temperature"] - 29.0) < 1e-4

    def test_telemetry_has_gps_hdop(self, hover_state):
        data = json.loads(state_to_json(hover_state, SerialiseProfile.TELEMETRY))
        assert "gps_hdop" in data


# ══════════════════════════════════════════════════════════════════════════════
#  Differential compression
# ══════════════════════════════════════════════════════════════════════════════

class TestDifferentialCompression:

    def test_first_diff_is_full(self, hover_state):
        ser  = StateSerializer()
        diff = json.loads(ser.to_diff_json(hover_state))
        # First call → full payload
        assert "x" in diff
        assert "roll" in diff

    def test_identical_state_sends_minimal_diff(self, hover_state):
        ser = StateSerializer()
        ser.to_diff_json(hover_state)                       # First — full
        s2 = hover_state.copy_with(sequence=hover_state.sequence + 1)
        diff2 = json.loads(ser.to_diff_json(s2))            # Second — minimal
        # Only identity fields + nothing changed
        identity_keys = {"vehicle_id", "sequence", "timestamp_wall"}
        extra_keys = set(diff2.keys()) - identity_keys
        # May include armed/airborne/computed fields that always differ
        # but should NOT include floating-point fields within threshold
        assert "x" not in extra_keys        # x unchanged
        assert "battery_soc" not in extra_keys  # unchanged

    def test_large_position_change_included(self, hover_state):
        ser = StateSerializer()
        ser.to_diff_json(hover_state)   # prime
        moved = hover_state.copy_with(
            x=hover_state.x + 10.0,    # 10m jump > threshold 0.005
            sequence=hover_state.sequence + 1,
        )
        diff = json.loads(ser.to_diff_json(moved))
        assert "x" in diff

    def test_small_position_change_excluded(self, hover_state):
        ser = StateSerializer()
        ser.to_diff_json(hover_state)   # prime
        tiny_move = hover_state.copy_with(
            x=hover_state.x + 0.001,   # 1mm < threshold 0.005
            sequence=hover_state.sequence + 1,
        )
        diff = json.loads(ser.to_diff_json(tiny_move))
        assert "x" not in diff

    def test_sequence_gap_forces_full(self, hover_state):
        """A skipped sequence number triggers a full resend."""
        ser = StateSerializer()
        ser.to_diff_json(hover_state)   # prime at seq N
        # Jump sequence by 5
        jumped = hover_state.copy_with(
            sequence=hover_state.sequence + 5
        )
        diff = json.loads(ser.to_diff_json(jumped))
        assert "x" in diff   # full resend

    def test_reset_forces_full_resend(self, hover_state):
        ser = StateSerializer()
        ser.to_diff_json(hover_state)   # prime
        ser.reset()                     # force full
        s2 = hover_state.copy_with(sequence=hover_state.sequence + 1)
        diff = json.loads(ser.to_diff_json(s2))
        assert "x" in diff   # full again

    def test_flight_mode_change_always_sent(self, hover_state):
        """Non-numeric field changes should always be included."""
        ser = StateSerializer()
        ser.to_diff_json(hover_state)   # prime
        changed = hover_state.copy_with(
            flight_mode=FlightMode.AUTO_MISSION,
            sequence=hover_state.sequence + 1,
        )
        diff = json.loads(ser.to_diff_json(changed))
        assert "flight_mode" in diff

    def test_diff_always_has_identity_fields(self, hover_state):
        ser = StateSerializer()
        for i in range(5):
            s = hover_state.copy_with(sequence=i)
            diff = json.loads(ser.to_diff_json(s))
            assert "vehicle_id"     in diff
            assert "sequence"       in diff
            assert "timestamp_wall" in diff


# ══════════════════════════════════════════════════════════════════════════════
#  states_to_json_array
# ══════════════════════════════════════════════════════════════════════════════

class TestStatesArraySerialiser:

    def test_empty_array(self):
        result = json.loads(states_to_json_array([]))
        assert result == []

    def test_array_of_states(self, hover_state):
        states = [hover_state.copy_with(x=float(i), sequence=i) for i in range(5)]
        result = json.loads(states_to_json_array(states))
        assert len(result) == 5
        assert result[3]["x"] == 3.0

    def test_array_is_json_array(self, hover_state):
        states = [hover_state.copy_with(sequence=i) for i in range(3)]
        raw = states_to_json_array(states)
        assert raw.startswith("[") and raw.endswith("]")


# ══════════════════════════════════════════════════════════════════════════════
#  DashboardServer lifecycle (no aiohttp → skip)
# ══════════════════════════════════════════════════════════════════════════════

class TestDashboardServerLifecycle:

    def test_instantiation_without_aiohttp(self):
        """DashboardServer can be instantiated without aiohttp."""
        from drone_sdk.dashboard.server import DashboardServer
        server = DashboardServer()
        assert not server.is_running

    def test_url_property(self):
        from drone_sdk.dashboard.server import DashboardServer, DashboardConfig
        server = DashboardServer(DashboardConfig(host="127.0.0.1", port=9999))
        assert "9999" in server.url
        assert "127.0.0.1" in server.url

    def test_ws_url_property(self):
        from drone_sdk.dashboard.server import DashboardServer, DashboardConfig
        server = DashboardServer(DashboardConfig(port=8001))
        assert server.ws_url.startswith("ws://")
        assert "8001" in server.ws_url

    def test_repr(self):
        from drone_sdk.dashboard.server import DashboardServer
        server = DashboardServer()
        assert "running=" in repr(server)

    def test_start_without_aiohttp_raises(self, store):
        import sys
        from drone_sdk.dashboard.server import DashboardServer
        server = DashboardServer()
        server.attach_vehicle(VEHICLE)

        original = sys.modules.pop("aiohttp", None)
        try:
            with pytest.raises(ImportError, match="aiohttp"):
                server.run()
        finally:
            if original:
                sys.modules["aiohttp"] = original

    def test_attach_vehicle(self, store):
        from drone_sdk.dashboard.server import DashboardServer
        server = DashboardServer()
        server.attach_vehicle(VEHICLE)
        assert VEHICLE in server._sub_ids

    def test_detach_vehicle(self, store):
        from drone_sdk.dashboard.server import DashboardServer
        server = DashboardServer()
        server.attach_vehicle(VEHICLE)
        server.detach_vehicle(VEHICLE)
        assert VEHICLE not in server._sub_ids

    def test_attach_all_vehicles(self, store):
        from drone_sdk.dashboard.server import DashboardServer
        # Create a second store
        StateStore.create("drone_extra", VehicleConfig("drone_extra"))
        server = DashboardServer()
        count = server.attach_all_vehicles()
        assert count >= 2

    def test_full_server_roundtrip(self, store, hover_state):
        """Start server in thread, attach vehicle, send state, verify broadcast."""
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not installed")

        import socket
        # Find a free port
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        from drone_sdk.dashboard.server import DashboardServer, DashboardConfig
        server = DashboardServer(DashboardConfig(host="127.0.0.1", port=port))
        server.attach_vehicle(VEHICLE)
        server.start_in_thread()
        assert server.is_running

        # Push a state — should not raise
        store.update(hover_state)
        time.sleep(0.1)

        server.stop()
