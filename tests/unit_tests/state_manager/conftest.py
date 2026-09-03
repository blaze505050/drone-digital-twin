"""
tests/unit_tests/state_manager/conftest.py
==========================================
Shared pytest fixtures for the state_manager test suite.

Every fixture that creates a StateStore uses autouse teardown to guarantee
the singleton registry is clean between tests — critical for test isolation.
"""
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
    EventBus,
    EventType,
    FlightMode,
    HealthStatus,
    PhysicsValidator,
    StateFactory,
    StateStore,
    VehicleConfig,
)


# ── Default test configuration ────────────────────────────────────────────────

DEFAULT_VEHICLE = "test_drone_0"


@pytest.fixture
def vehicle_id() -> str:
    return DEFAULT_VEHICLE


@pytest.fixture
def config(vehicle_id: str) -> VehicleConfig:
    """Standard VehicleConfig for unit tests."""
    return VehicleConfig(
        vehicle_id              = vehicle_id,
        vehicle_type            = "quadrotor_x",
        max_altitude_m          = 500.0,
        max_speed_ms            = 30.0,
        max_rotor_omega_rads    = 1600.0,
        battery_cells           = 4,
        expected_update_rate_hz = 50.0,
        max_state_age_ms        = 200.0,   # Short for tests
        history_size            = 100,     # Small for tests
    )


@pytest.fixture(autouse=True)
def cleanup_stores():
    """Guarantee StateStore registry is clean before and after every test."""
    StateStore.destroy_all()
    yield
    StateStore.destroy_all()


@pytest.fixture
def store(vehicle_id: str, config: VehicleConfig) -> StateStore:
    """A fresh StateStore for the default vehicle."""
    return StateStore.create(vehicle_id, config)


@pytest.fixture
def validator(config: VehicleConfig) -> PhysicsValidator:
    return PhysicsValidator(config)


@pytest.fixture
def event_bus(vehicle_id: str) -> EventBus:
    return EventBus(vehicle_id)


# ── State construction helpers ────────────────────────────────────────────────

@pytest.fixture
def initial_state(vehicle_id: str) -> DroneStateVector:
    """Ground-rest initial state that passes all validators."""
    return StateFactory.create_initial(vehicle_id, source=DataSource.MANUAL)


@pytest.fixture
def hover_state(vehicle_id: str) -> DroneStateVector:
    """Hovering state at 10 m AGL."""
    return StateFactory.create_initial(vehicle_id, source=DataSource.SITL).copy_with(
        x             = 0.0,
        y             = 0.0,
        z             = -10.0,        # NED: -z = altitude
        altitude_agl  = 10.0,
        altitude_amsl = 10.0,
        arming_state  = ArmingState.ARMED,
        flight_mode   = FlightMode.POSITION_HOLD,
        battery_voltage  = 15.5,     # 4S, ~3.875 V/cell
        battery_soc      = 0.85,
        battery_current  = 12.0,
        battery_remaining_wh = 18.7,
        omega1 = 500.0,
        omega2 = 500.0,
        omega3 = 500.0,
        omega4 = 500.0,
        gps_fix_type   = 3,
        gps_satellites = 12,
        gps_hdop       = 0.9,
        gps_vdop       = 1.2,
        is_valid       = True,
    )


@pytest.fixture
def from_6dof_state(vehicle_id: str) -> DroneStateVector:
    """State built from a rigid_body.py-style 6-DOF output."""
    position         = np.array([5.0, 3.0, -8.0])
    velocity         = np.array([1.0, 0.5, -0.1])
    quaternion       = np.array([1.0, 0.0, 0.0, 0.0])   # level
    angular_velocity = np.array([0.01, -0.02, 0.005])
    rotor_omega      = np.array([480.0, 482.0, 479.0, 481.0])

    return StateFactory.from_6dof(
        vehicle_id,
        position,
        velocity,
        quaternion,
        angular_velocity,
        rotor_omega      = rotor_omega,
        timestamp_sim    = 1.0,
        sequence         = 100,
        source           = DataSource.SITL,
    )


@pytest.fixture
def partial_update(vehicle_id: str) -> DroneStateUpdate:
    """A typical MAVLink-style partial update (battery only)."""
    upd = DroneStateUpdate(
        vehicle_id       = vehicle_id,
        source           = DataSource.MAVLINK,
        timestamp_wall   = time.time(),
        timestamp_mono   = time.monotonic(),
    )
    upd.battery_voltage = 15.1
    upd.battery_current = 14.2
    upd.battery_soc     = 0.72
    return upd
