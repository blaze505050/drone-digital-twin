"""Tests for digital_twin_core: MEKF, parallel twin model, residual monitor, and recalibration."""
import math
import numpy as np
import pytest

from drone_sdk.digital_twin_core import (
    ClosedLoopDigitalTwin,
    DynamicTwinModel,
    MultiplicativeEKF,
    OnlineRecalibrator,
    ResidualReport,
    TwinPhysicsParameters,
    TwinResidualMonitor,
)
from drone_sdk.predictive_maintenance import HealthIndex
from drone_sdk.state_manager.schema import DataSource, DroneStateVector, HealthStatus


def test_mekf_propagation_and_updates():
    ekf = MultiplicativeEKF(vehicle_id="test_uav", init_pos_ned=np.array([0.0, 0.0, -10.0]))
    assert ekf.p[2] == -10.0
    assert ekf.q[0] == 1.0

    # Predict step (hover specific force: az = -9.81 m/s^2)
    accel = np.array([0.0, 0.0, -9.81])
    gyro = np.zeros(3)
    ekf.predict(accel, gyro, dt=0.02)

    # State update
    state = ekf.get_state_vector()
    assert isinstance(state, DroneStateVector)
    assert abs(state.z - (-10.0)) < 0.1
    assert state.is_valid

    # GPS update
    gps_pos = np.array([1.0, 2.0, -10.0])
    gps_vel = np.array([0.1, 0.0, 0.0])
    for _ in range(5):
        ekf.update_gps(gps_pos, gps_vel)
    assert abs(ekf.p[0] - 1.0) < 0.4

    # Baro update
    ekf.update_barometer(alt_baro_m=10.5)
    assert abs(ekf.p[2] - (-10.5)) < 0.5


def test_dynamic_twin_model():
    twin = DynamicTwinModel(vehicle_id="test_twin")
    # Commanded hover action [0.55, 0.55, 0.55, 0.55]
    action = np.full(4, 0.55)

    pred1 = twin.step(action, dt=0.02)
    assert isinstance(pred1, DroneStateVector)
    assert pred1.source == DataSource.TWIN
    assert pred1.is_airborne or pred1.altitude_agl >= 0.0
    assert pred1.vehicle_id == "test_twin"


def test_residual_monitor():
    monitor = TwinResidualMonitor()
    twin = DynamicTwinModel()
    action = np.full(4, 0.55)

    pred_state = twin.step(action, dt=0.02)
    # Identical actual state
    report_nom = monitor.update(predicted=pred_state, actual=pred_state)

    assert isinstance(report_nom, ResidualReport)
    assert report_nom.status == HealthStatus.NOMINAL
    assert report_nom.pos_error_m == 0.0
    assert report_nom.health_score > 0.95

    # Divergent state test (e.g. 1.2m offset)
    divergent_state = pred_state.copy_with(x=pred_state.x + 1.2)
    report_deg = monitor.update(predicted=pred_state, actual=divergent_state)

    assert report_deg.status == HealthStatus.DEGRADED
    assert report_deg.pos_error_m >= 1.2

    # Verify integration with HealthIndex
    hi = HealthIndex()
    monitor.feed_into_health_index(hi, weight=0.35)
    assert "twin_residual" in hi.to_dict()["subsystems"]


def test_online_recalibrator():
    twin = DynamicTwinModel()
    recal = OnlineRecalibrator(twin_model=twin, window_size=30, recal_interval_frames=10)

    action = np.full(4, 0.55)
    for _ in range(15):
        state = twin.step(action, dt=0.02)
        recal.record_frame(action, state)

    scales = recal.recalibrate()
    assert "mass_scale" in scales
    assert "thrust_scale" in scales
    assert "drag_scale" in scales


def test_closed_loop_digital_twin_orchestrator():
    dt_system = ClosedLoopDigitalTwin(vehicle_id="closed_loop_uav")

    # Step sensor telemetry
    accel = np.array([0.0, 0.0, -9.81])
    gyro = np.zeros(3)
    est_state = dt_system.update_sensors(accel, gyro, dt=0.02, gps_pos=np.array([0.0, 0.0, -10.0]), gps_vel=np.zeros(3))
    assert est_state.source == DataSource.SITL

    # Step twin model command
    action = np.full(4, 0.55)
    pred_state, report = dt_system.step(action, dt=0.02)

    assert pred_state.source == DataSource.TWIN
    assert isinstance(report, ResidualReport)
    assert dt_system.latest_predicted_state is not None
    assert dt_system.latest_estimated_state is not None
