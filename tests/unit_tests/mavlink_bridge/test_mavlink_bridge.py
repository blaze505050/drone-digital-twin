"""
Tests for drone_sdk.mavlink_bridge (Module 3).

All tests use mock MAVLink message objects — no real pymavlink installation
required.  The parser functions are pure (stateless), so they are easy to
unit-test in complete isolation.
"""
from __future__ import annotations

import math
import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from drone_sdk.state_manager import (
    ArmingState,
    DataSource,
    DroneStateUpdate,
    FlightMode,
    StateFactory,
    StateStore,
    VehicleConfig,
)
from drone_sdk.mavlink_bridge.parsers import (
    MESSAGE_PARSERS,
    dispatch,
    parse_actuator_output_status,
    parse_altitude,
    parse_attitude,
    parse_attitude_quaternion,
    parse_battery_status,
    parse_global_position_int,
    parse_gps_raw_int,
    parse_heartbeat,
    parse_highres_imu,
    parse_local_position_ned,
    parse_vfr_hud,
)


# ── Mock MAVLink message factory ──────────────────────────────────────────────

VEHICLE = "test_drone_mav"


def make_msg(msg_type: str, sysid: int = 1, compid: int = 1, **fields: Any):
    """Create a mock MAVLink message with the given type and fields."""
    msg = MagicMock()
    msg.get_type.return_value     = msg_type
    msg.get_srcSystem.return_value = sysid
    msg.get_srcComponent.return_value = compid
    for k, v in fields.items():
        setattr(msg, k, v)
    return msg


@pytest.fixture(autouse=True)
def clean_stores():
    StateStore.destroy_all()
    yield
    StateStore.destroy_all()


# ══════════════════════════════════════════════════════════════════════════════
#  parse_heartbeat
# ══════════════════════════════════════════════════════════════════════════════

class TestParseHeartbeat:

    def test_armed_vehicle(self):
        """Base mode bit 7 set → ARMED."""
        msg = make_msg("HEARTBEAT",
                       base_mode=0x81,    # 0x80 (armed) | 0x01 (custom mode)
                       custom_mode=(5 << 16))   # main mode 5 = POSITION_HOLD
        upd = parse_heartbeat(msg, VEHICLE)
        assert upd is not None
        assert upd.arming_state == ArmingState.ARMED
        assert upd.flight_mode == FlightMode.POSITION_HOLD

    def test_disarmed_vehicle(self):
        msg = make_msg("HEARTBEAT",
                       base_mode=0x01,    # custom mode but not armed
                       custom_mode=(4 << 16))   # ALTITUDE_HOLD
        upd = parse_heartbeat(msg, VEHICLE)
        assert upd.arming_state == ArmingState.DISARMED
        assert upd.flight_mode == FlightMode.ALTITUDE_HOLD

    def test_unknown_flight_mode(self):
        msg = make_msg("HEARTBEAT",
                       base_mode=0x00,    # no custom mode
                       custom_mode=0)
        upd = parse_heartbeat(msg, VEHICLE)
        assert upd.flight_mode == FlightMode.UNKNOWN

    def test_gcs_heartbeat_returns_none(self):
        """Heartbeats from GCS (type=6) should be ignored."""
        msg = make_msg("HEARTBEAT", base_mode=0, custom_mode=0, type=6)
        upd = parse_heartbeat(msg, VEHICLE)
        assert upd is None

    def test_all_main_modes_mapped(self):
        """Every PX4 main mode in our table should produce a known FlightMode."""
        from drone_sdk.mavlink_bridge.parsers import _PX4_MAIN_MODE
        for px4_mode, expected_fm in _PX4_MAIN_MODE.items():
            msg = make_msg("HEARTBEAT",
                           base_mode=0x01,
                           custom_mode=(px4_mode << 16))
            upd = parse_heartbeat(msg, VEHICLE)
            assert upd.flight_mode == expected_fm

    def test_output_is_drone_state_update(self):
        msg = make_msg("HEARTBEAT", base_mode=0, custom_mode=0)
        upd = parse_heartbeat(msg, VEHICLE)
        assert isinstance(upd, DroneStateUpdate)
        assert upd.vehicle_id == VEHICLE
        assert upd.source == DataSource.MAVLINK


# ══════════════════════════════════════════════════════════════════════════════
#  parse_attitude
# ══════════════════════════════════════════════════════════════════════════════

class TestParseAttitude:

    def test_euler_angles_parsed(self):
        msg = make_msg("ATTITUDE",
                       roll=0.1, pitch=-0.05, yaw=1.57,
                       rollspeed=0.01, pitchspeed=0.0, yawspeed=-0.02)
        upd = parse_attitude(msg, VEHICLE)
        assert upd is not None
        np.testing.assert_allclose(upd.euler, [0.1, -0.05, 1.57], atol=1e-9)

    def test_angular_rates_parsed(self):
        msg = make_msg("ATTITUDE",
                       roll=0.0, pitch=0.0, yaw=0.0,
                       rollspeed=0.5, pitchspeed=-0.3, yawspeed=0.1)
        upd = parse_attitude(msg, VEHICLE)
        np.testing.assert_allclose(upd.angular_velocity, [0.5, -0.3, 0.1], atol=1e-9)

    def test_zero_attitude(self):
        msg = make_msg("ATTITUDE",
                       roll=0.0, pitch=0.0, yaw=0.0,
                       rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0)
        upd = parse_attitude(msg, VEHICLE)
        np.testing.assert_allclose(upd.euler, [0.0, 0.0, 0.0], atol=1e-15)

    def test_timestamp_set(self):
        before = time.time()
        msg = make_msg("ATTITUDE",
                       roll=0.0, pitch=0.0, yaw=0.0,
                       rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0)
        upd = parse_attitude(msg, VEHICLE)
        after = time.time()
        assert before <= upd.timestamp_wall <= after


# ══════════════════════════════════════════════════════════════════════════════
#  parse_attitude_quaternion
# ══════════════════════════════════════════════════════════════════════════════

class TestParseAttitudeQuaternion:

    def test_identity_quaternion(self):
        """MAVLink identity: q1=1, q2=0, q3=0, q4=0 → Hamilton [1,0,0,0]."""
        msg = make_msg("ATTITUDE_QUATERNION",
                       q1=1.0, q2=0.0, q3=0.0, q4=0.0,
                       rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0)
        upd = parse_attitude_quaternion(msg, VEHICLE)
        np.testing.assert_allclose(upd.quaternion, [1.0, 0.0, 0.0, 0.0], atol=1e-9)

    def test_90deg_roll_quaternion(self):
        """90° roll: q = [cos(45°), sin(45°), 0, 0]."""
        w = math.cos(math.pi / 4)
        x = math.sin(math.pi / 4)
        msg = make_msg("ATTITUDE_QUATERNION",
                       q1=w, q2=x, q3=0.0, q4=0.0,
                       rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0)
        upd = parse_attitude_quaternion(msg, VEHICLE)
        # Hamilton: [w, x, y, z]
        np.testing.assert_allclose(upd.quaternion, [w, x, 0.0, 0.0], atol=1e-9)

    def test_quaternion_field_order(self):
        """Verify MAVLink q1,q2,q3,q4 → Hamilton w,x,y,z mapping."""
        msg = make_msg("ATTITUDE_QUATERNION",
                       q1=0.1, q2=0.2, q3=0.3, q4=0.4,
                       rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0)
        upd = parse_attitude_quaternion(msg, VEHICLE)
        # q1=w=0.1, q2=x=0.2, q3=y=0.3, q4=z=0.4
        assert abs(upd.quaternion[0] - 0.1) < 1e-9   # w
        assert abs(upd.quaternion[1] - 0.2) < 1e-9   # x
        assert abs(upd.quaternion[2] - 0.3) < 1e-9   # y
        assert abs(upd.quaternion[3] - 0.4) < 1e-9   # z


# ══════════════════════════════════════════════════════════════════════════════
#  parse_local_position_ned
# ══════════════════════════════════════════════════════════════════════════════

class TestParseLocalPositionNED:

    def test_position_parsed(self):
        msg = make_msg("LOCAL_POSITION_NED",
                       x=10.5, y=-3.2, z=-8.0,
                       vx=1.0, vy=0.5, vz=-0.1)
        upd = parse_local_position_ned(msg, VEHICLE)
        np.testing.assert_allclose(upd.position, [10.5, -3.2, -8.0], atol=1e-9)

    def test_velocity_parsed(self):
        msg = make_msg("LOCAL_POSITION_NED",
                       x=0.0, y=0.0, z=0.0,
                       vx=2.5, vy=-1.0, vz=0.3)
        upd = parse_local_position_ned(msg, VEHICLE)
        np.testing.assert_allclose(upd.velocity, [2.5, -1.0, 0.3], atol=1e-9)

    def test_ned_convention_preserved(self):
        """z should be negative for altitude (NED: positive = down)."""
        msg = make_msg("LOCAL_POSITION_NED",
                       x=0.0, y=0.0, z=-15.0,
                       vx=0.0, vy=0.0, vz=0.0)
        upd = parse_local_position_ned(msg, VEHICLE)
        assert upd.position[2] == -15.0   # 15 m altitude → z = -15


# ══════════════════════════════════════════════════════════════════════════════
#  parse_global_position_int
# ══════════════════════════════════════════════════════════════════════════════

class TestParseGlobalPositionInt:

    def test_lat_lon_converted_from_degE7(self):
        msg = make_msg("GLOBAL_POSITION_INT",
                       lat=128_567_891,   # 12.8567891°
                       lon=770_123_456,   # 77.0123456°
                       alt=100_000,       # 100 m MSL
                       relative_alt=50_000,  # 50 m AGL
                       hdg=0xFFFF)        # invalid heading
        upd = parse_global_position_int(msg, VEHICLE)
        assert abs(upd.gps_lat - 12.8567891) < 1e-5
        assert abs(upd.gps_lon - 77.0123456) < 1e-5

    def test_alt_converted_from_mm(self):
        msg = make_msg("GLOBAL_POSITION_INT",
                       lat=0, lon=0,
                       alt=120_000,          # 120 m MSL
                       relative_alt=10_000,  # 10 m AGL
                       hdg=0)
        upd = parse_global_position_int(msg, VEHICLE)
        assert abs(upd.gps_alt_msl - 120.0) < 1e-6
        assert abs(upd.altitude_agl - 10.0) < 1e-6

    def test_bengaluru_coordinates(self):
        """Spot-check: Bengaluru HAL Airport coords."""
        lat_degE7 = 129_600_000   # 12.96°N
        lon_degE7 = 774_170_000   # 77.417°E
        msg = make_msg("GLOBAL_POSITION_INT",
                       lat=lat_degE7, lon=lon_degE7,
                       alt=902_000, relative_alt=0, hdg=0)
        upd = parse_global_position_int(msg, VEHICLE)
        assert abs(upd.gps_lat - 12.96) < 0.01
        assert abs(upd.gps_lon - 77.417) < 0.01


# ══════════════════════════════════════════════════════════════════════════════
#  parse_gps_raw_int
# ══════════════════════════════════════════════════════════════════════════════

class TestParseGpsRawInt:

    def test_3d_fix(self):
        msg = make_msg("GPS_RAW_INT",
                       fix_type=3, satellites_visible=12,
                       eph=90, epv=150)
        upd = parse_gps_raw_int(msg, VEHICLE)
        assert upd.gps_fix_type   == 3
        assert upd.gps_satellites == 12
        assert abs(upd.gps_hdop - 0.90) < 1e-9
        assert abs(upd.gps_vdop - 1.50) < 1e-9

    def test_invalid_eph_returns_99(self):
        """eph=0xFFFF means unknown → 99.9."""
        msg = make_msg("GPS_RAW_INT",
                       fix_type=0, satellites_visible=0,
                       eph=0xFFFF, epv=0xFFFF)
        upd = parse_gps_raw_int(msg, VEHICLE)
        assert upd.gps_hdop == 99.9
        assert upd.gps_vdop == 99.9

    def test_no_fix(self):
        msg = make_msg("GPS_RAW_INT",
                       fix_type=0, satellites_visible=0,
                       eph=9999, epv=9999)
        upd = parse_gps_raw_int(msg, VEHICLE)
        assert upd.gps_fix_type == 0


# ══════════════════════════════════════════════════════════════════════════════
#  parse_highres_imu
# ══════════════════════════════════════════════════════════════════════════════

class TestParseHighresIMU:

    def test_acceleration_parsed(self):
        msg = make_msg("HIGHRES_IMU",
                       xacc=0.1, yacc=-0.05, zacc=-9.81)
        upd = parse_highres_imu(msg, VEHICLE)
        np.testing.assert_allclose(
            upd.acceleration, [0.1, -0.05, -9.81], atol=1e-6
        )

    def test_gravity_vector(self):
        """Hovering drone: only gravity in -Z body (assuming level)."""
        msg = make_msg("HIGHRES_IMU",
                       xacc=0.0, yacc=0.0, zacc=-9.81)
        upd = parse_highres_imu(msg, VEHICLE)
        assert abs(upd.acceleration[2] - (-9.81)) < 1e-6


# ══════════════════════════════════════════════════════════════════════════════
#  parse_battery_status
# ══════════════════════════════════════════════════════════════════════════════

class TestParseBatteryStatus:

    def test_4s_battery_parsed(self):
        """4S battery: 4 cells × ~3875 mV = 15500 mV total → 15.5 V."""
        msg = make_msg("BATTERY_STATUS",
                       voltages=[3875, 3875, 3875, 3875,
                                  0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                                  0xFFFF, 0xFFFF],
                       current_battery=1200,   # 12.00 A in cA
                       battery_remaining=82)   # 82%
        upd = parse_battery_status(msg, VEHICLE)
        assert abs(upd.battery_voltage - 15.5) < 0.01
        assert abs(upd.battery_current - 12.0) < 0.01
        assert abs(upd.battery_soc - 0.82) < 0.001

    def test_all_invalid_cells(self):
        """All cells = 0xFFFF → voltage = 0.0."""
        msg = make_msg("BATTERY_STATUS",
                       voltages=[0xFFFF] * 10,
                       current_battery=-1,
                       battery_remaining=-1)
        upd = parse_battery_status(msg, VEHICLE)
        assert upd.battery_voltage == 0.0

    def test_soc_out_of_range_clamped(self):
        """battery_remaining = -1 → SoC = 1.0 (unknown, assume full)."""
        msg = make_msg("BATTERY_STATUS",
                       voltages=[0xFFFF] * 10,
                       current_battery=0,
                       battery_remaining=-1)
        upd = parse_battery_status(msg, VEHICLE)
        assert upd.battery_soc == 1.0

    def test_fully_charged(self):
        msg = make_msg("BATTERY_STATUS",
                       voltages=[4200, 4200, 4200, 4200,
                                  0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                                  0xFFFF, 0xFFFF],
                       current_battery=0,
                       battery_remaining=100)
        upd = parse_battery_status(msg, VEHICLE)
        assert abs(upd.battery_soc - 1.0) < 0.001
        assert abs(upd.battery_voltage - 16.8) < 0.01


# ══════════════════════════════════════════════════════════════════════════════
#  parse_actuator_output_status
# ══════════════════════════════════════════════════════════════════════════════

class TestParseActuatorOutputStatus:

    def test_hover_motors(self):
        """All motors at ~50% throttle → 50% of max omega."""
        from drone_sdk.mavlink_bridge.parsers import _MAX_MOTOR_OMEGA_RADS
        msg = make_msg("ACTUATOR_OUTPUT_STATUS",
                       actuator=[0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
        upd = parse_actuator_output_status(msg, VEHICLE)
        expected = 0.5 * _MAX_MOTOR_OMEGA_RADS
        np.testing.assert_allclose(
            upd.rotor_omega, [expected] * 4, atol=1e-6
        )

    def test_motor_ordering_px4_x(self):
        """PX4 X config: output[0]=FR, output[1]=BL, output[2]=FL, output[3]=BR."""
        from drone_sdk.mavlink_bridge.parsers import _MAX_MOTOR_OMEGA_RADS
        # Different throttle per motor to verify ordering
        msg = make_msg("ACTUATOR_OUTPUT_STATUS",
                       actuator=[0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
        upd = parse_actuator_output_status(msg, VEHICLE)
        # omega1=front-right=output[0]=0.4, omega2=front-left=output[2]=0.2
        # omega3=back-right=output[3]=0.1, omega4=back-left=output[1]=0.3
        assert abs(upd.rotor_omega[0] - 0.4 * _MAX_MOTOR_OMEGA_RADS) < 1e-6
        assert abs(upd.rotor_omega[1] - 0.2 * _MAX_MOTOR_OMEGA_RADS) < 1e-6
        assert abs(upd.rotor_omega[2] - 0.1 * _MAX_MOTOR_OMEGA_RADS) < 1e-6
        assert abs(upd.rotor_omega[3] - 0.3 * _MAX_MOTOR_OMEGA_RADS) < 1e-6

    def test_zero_throttle(self):
        msg = make_msg("ACTUATOR_OUTPUT_STATUS",
                       actuator=[0.0, 0.0, 0.0, 0.0])
        upd = parse_actuator_output_status(msg, VEHICLE)
        np.testing.assert_allclose(upd.rotor_omega, [0.0, 0.0, 0.0, 0.0])

    def test_clamped_above_one(self):
        """Actuator output > 1.0 should be clamped to 1.0."""
        from drone_sdk.mavlink_bridge.parsers import _MAX_MOTOR_OMEGA_RADS
        msg = make_msg("ACTUATOR_OUTPUT_STATUS",
                       actuator=[1.5, 1.5, 1.5, 1.5])
        upd = parse_actuator_output_status(msg, VEHICLE)
        expected = _MAX_MOTOR_OMEGA_RADS
        np.testing.assert_allclose(upd.rotor_omega, [expected] * 4, atol=1e-6)

    def test_too_few_outputs_returns_none(self):
        msg = make_msg("ACTUATOR_OUTPUT_STATUS",
                       actuator=[0.5, 0.5])   # only 2 outputs
        upd = parse_actuator_output_status(msg, VEHICLE)
        assert upd is None

    def test_missing_actuator_attr_returns_none(self):
        """A message whose .actuator attribute raises AttributeError → None."""
        msg = MagicMock()
        del msg.actuator   # Force AttributeError on access
        upd = parse_actuator_output_status(msg, VEHICLE)
        assert upd is None


# ══════════════════════════════════════════════════════════════════════════════
#  dispatch
# ══════════════════════════════════════════════════════════════════════════════

class TestDispatch:

    def test_known_message_dispatched(self):
        msg = make_msg("ATTITUDE",
                       roll=0.1, pitch=0.0, yaw=0.0,
                       rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0)
        upd = dispatch(msg, VEHICLE)
        assert upd is not None

    def test_unknown_message_returns_none(self):
        msg = make_msg("PING")   # not in our parser table
        upd = dispatch(msg, VEHICLE)
        assert upd is None

    def test_bad_data_not_in_table(self):
        msg = make_msg("BAD_DATA")
        upd = dispatch(msg, VEHICLE)
        assert upd is None

    def test_all_registered_message_types(self):
        """Every message type in MESSAGE_PARSERS should have a callable."""
        for msg_type, parser in MESSAGE_PARSERS.items():
            assert callable(parser), f"Parser for {msg_type} is not callable"

    def test_parser_exception_returns_none(self):
        """If a parser raises, dispatch returns None and does not propagate."""
        broken_msg = MagicMock()
        broken_msg.get_type.return_value = "ATTITUDE"
        broken_msg.get_srcSystem.return_value = 1
        broken_msg.get_srcComponent.return_value = 1
        # Accessing .roll will raise AttributeError since it's a MagicMock
        # with no spec — actually MagicMock will return a MagicMock, so
        # let's force a real exception by making roll raise.
        type(broken_msg).roll = property(lambda self: (_ for _ in ()).throw(ValueError("bad")))
        # dispatch should catch this and return None
        result = dispatch(broken_msg, VEHICLE)
        assert result is None

    def test_dispatch_returns_drone_state_update(self):
        msg = make_msg("LOCAL_POSITION_NED",
                       x=1.0, y=2.0, z=-5.0,
                       vx=0.0, vy=0.0, vz=0.0)
        upd = dispatch(msg, VEHICLE)
        assert isinstance(upd, DroneStateUpdate)
        assert upd.vehicle_id == VEHICLE
        assert upd.source     == DataSource.MAVLINK


# ══════════════════════════════════════════════════════════════════════════════
#  MAVLinkSource lifecycle (no real connection — mock pymavlink)
# ══════════════════════════════════════════════════════════════════════════════

class TestMAVLinkSourceLifecycle:

    @pytest.fixture
    def mock_mavutil(self):
        """Mock the entire pymavlink.mavutil module."""
        with patch.dict("sys.modules", {"pymavlink": MagicMock(), "pymavlink.mavutil": MagicMock()}):
            yield

    def test_connect_without_pymavlink_raises_importerror(self):
        """If pymavlink is not installed, connect() raises ImportError."""
        import sys
        # Temporarily remove pymavlink from sys.modules
        original = sys.modules.pop("pymavlink", None)
        try:
            from drone_sdk.mavlink_bridge.mavlink_source import MAVLinkSource
            from drone_sdk.telemetry_engine.source import SourcePriority
            src = MAVLinkSource("s", VEHICLE, "udp:127.0.0.1:14550",
                                priority=SourcePriority.SITL)
            with pytest.raises(ImportError, match="pymavlink"):
                src.connect()
        finally:
            if original:
                sys.modules["pymavlink"] = original

    def test_source_initial_status(self):
        from drone_sdk.mavlink_bridge.mavlink_source import MAVLinkSource
        from drone_sdk.telemetry_engine.source import SourcePriority, SourceStatus
        src = MAVLinkSource("s", VEHICLE, "udp:127.0.0.1:14550",
                            priority=SourcePriority.SITL)
        assert src.status == SourceStatus.DISCONNECTED

    def test_poll_when_disconnected_returns_empty(self):
        from drone_sdk.mavlink_bridge.mavlink_source import MAVLinkSource
        from drone_sdk.telemetry_engine.source import SourcePriority
        src = MAVLinkSource("s", VEHICLE, "udp:127.0.0.1:14550",
                            priority=SourcePriority.SITL)
        updates = src.poll()
        assert updates == []

    def test_get_message_counts_initially_empty(self):
        from drone_sdk.mavlink_bridge.mavlink_source import MAVLinkSource
        from drone_sdk.telemetry_engine.source import SourcePriority
        src = MAVLinkSource("s", VEHICLE, "udp:127.0.0.1:14550",
                            priority=SourcePriority.SITL)
        assert src.get_message_counts() == {}

    def test_connection_presets_are_strings(self):
        from drone_sdk.mavlink_bridge import ConnectionPreset
        for attr in vars(ConnectionPreset):
            if not attr.startswith("_"):
                assert isinstance(getattr(ConnectionPreset, attr), str)


# ══════════════════════════════════════════════════════════════════════════════
#  Integration: parser → StateStore via merge_update
# ══════════════════════════════════════════════════════════════════════════════

class TestParserToStateStoreIntegration:
    """Verify the full pipeline: MAVLink msg → parse → merge → StateStore."""

    @pytest.fixture
    def store(self):
        return StateStore.create(VEHICLE, VehicleConfig(vehicle_id=VEHICLE))

    def test_attitude_update_reflected_in_store(self, store):
        state = StateFactory.create_initial(VEHICLE)
        store.update(state)

        msg = make_msg("ATTITUDE",
                       roll=0.3, pitch=-0.1, yaw=1.2,
                       rollspeed=0.05, pitchspeed=0.0, yawspeed=-0.02)
        upd = parse_attitude(msg, VEHICLE)
        store.update_partial(upd)

        latest = store.get_latest()
        assert abs(latest.roll  - 0.3) < 1e-6
        assert abs(latest.pitch - (-0.1)) < 1e-6
        assert abs(latest.yaw   - 1.2) < 1e-6

    def test_position_update_reflected_in_store(self, store):
        state = StateFactory.create_initial(VEHICLE)
        store.update(state)

        msg = make_msg("LOCAL_POSITION_NED",
                       x=10.0, y=5.0, z=-20.0,
                       vx=1.0, vy=0.5, vz=0.0)
        upd = parse_local_position_ned(msg, VEHICLE)
        store.update_partial(upd)

        latest = store.get_latest()
        assert abs(latest.x - 10.0) < 1e-6
        assert abs(latest.y - 5.0)  < 1e-6
        assert abs(latest.z - (-20.0)) < 1e-6

    def test_battery_update_does_not_overwrite_position(self, store):
        """Partial update should merge, not replace."""
        initial = StateFactory.create_initial(VEHICLE).copy_with(
            x=15.0, y=8.0, z=-10.0,
            battery_voltage=16.8
        )
        store.update(initial)

        # Battery-only update
        msg = make_msg("BATTERY_STATUS",
                       voltages=[3700, 3700, 3700, 3700,
                                  0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                                  0xFFFF, 0xFFFF],
                       current_battery=1000,
                       battery_remaining=75)
        upd = parse_battery_status(msg, VEHICLE)
        store.update_partial(upd)

        latest = store.get_latest()
        # Position preserved
        assert abs(latest.x - 15.0) < 1e-6
        assert abs(latest.y - 8.0)  < 1e-6
        # Battery updated
        assert abs(latest.battery_soc - 0.75) < 0.001

    def test_multi_message_fusion(self, store):
        """Simulate a realistic sequence of MAVLink messages."""
        state = StateFactory.create_initial(VEHICLE)
        store.update(state)

        messages = [
            make_msg("ATTITUDE", roll=0.05, pitch=-0.02, yaw=0.785,
                     rollspeed=0.0, pitchspeed=0.0, yawspeed=0.01),
            make_msg("LOCAL_POSITION_NED",
                     x=5.0, y=-2.0, z=-15.0,
                     vx=0.5, vy=-0.1, vz=0.05),
            make_msg("BATTERY_STATUS",
                     voltages=[3820, 3820, 3820, 3820,
                                0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                                0xFFFF, 0xFFFF],
                     current_battery=1150, battery_remaining=78),
            make_msg("HEARTBEAT",
                     base_mode=0x81,
                     custom_mode=(5 << 16)),   # POSITION_HOLD + ARMED
        ]

        for msg in messages:
            upd = dispatch(msg, VEHICLE)
            if upd:
                store.update_partial(upd)

        latest = store.get_latest()
        assert abs(latest.roll  - 0.05) < 1e-6
        assert abs(latest.x     - 5.0)  < 1e-6
        assert abs(latest.battery_soc - 0.78) < 0.001
        assert latest.arming_state == ArmingState.ARMED
        assert latest.flight_mode  == FlightMode.POSITION_HOLD
