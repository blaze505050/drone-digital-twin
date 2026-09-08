"""
Unit and Negative Tests for Data Provenance Firewall & Stream Contracts (V00).

Verifies that:
1. Synthetic fallback data is NEVER marked as DataStatus.REAL or is_empirical=True.
2. Missing datasets gracefully tag DataStatus.SYNTHETIC_FALLBACK.
3. Coordinates module accurately transforms NED <-> FRD and computes specific force.
4. Typed stream contracts reject invalid / corrupted data.
"""
import math
import numpy as np
import pytest

from drone_sdk.battery_twin.nasa_adapter import NASABatteryDatasetAdapter
from drone_sdk.contracts import (
    DataStatus,
    SensorMeasurement,
    StateEstimate,
    TwinPrediction,
    ReferenceTruth,
    CommandIntent,
    AppliedActuation,
    GRAVITY_NED,
    STANDARD_GRAVITY_MPS2,
    quat_to_rot_matrix,
    rot_matrix_to_quat,
    specific_force_from_acceleration,
    acceleration_from_specific_force,
)
from drone_sdk.validation_framework.benchmarks import EuRoCDatasetLoader, ZurichUAVDatasetLoader
from drone_sdk.validation_framework.flight_log_validator import RealFlightLogValidator


def test_provenance_firewall_euroc_fallback():
    loader = EuRoCDatasetLoader(sequence="V1_01_easy", data_path=None)
    assert loader.is_synthetic_fallback is True
    assert loader.data_status == DataStatus.SYNTHETIC_FALLBACK
    assert loader.data_status.is_empirical is False

    traj = loader.trajectory
    assert traj.status == DataStatus.SYNTHETIC_FALLBACK

    est_pos = traj.pos_ned + 0.05
    results = loader.evaluate_state_estimator(est_pos)
    assert results["data_status"] == DataStatus.SYNTHETIC_FALLBACK.value
    assert results["is_empirical"] is False


def test_provenance_firewall_zurich_fallback():
    loader = ZurichUAVDatasetLoader(sequence="urban_street_01", data_path=None)
    assert loader.is_synthetic_fallback is True
    assert loader.data_status == DataStatus.SYNTHETIC_FALLBACK
    assert loader.data_status.is_empirical is False

    traj = loader.trajectory
    assert traj.status == DataStatus.SYNTHETIC_FALLBACK

    est_pos = traj.pos_ned + 0.05
    results = loader.evaluate_canyon_tracking(est_pos)
    assert results["data_status"] == DataStatus.SYNTHETIC_FALLBACK.value
    assert results["is_empirical"] is False


def test_provenance_firewall_nasa_adapter():
    adapter = NASABatteryDatasetAdapter(cell_id="B0005", data_path=None)
    assert adapter.is_synthetic_fallback is True
    assert adapter.data_status == DataStatus.SYNTHETIC_FALLBACK
    assert adapter.data_status.is_empirical is False

    cycle = adapter.get_cycle(1)
    assert cycle.status == DataStatus.SYNTHETIC_FALLBACK

    # Empty/mock ecm model
    class MockECM:
        def step(self, current, dt, temperature_c):
            class State:
                v_terminal = 3.8
            return State()

    metrics = adapter.validate_ecm_against_cycle(MockECM(), cycle_index=1)
    assert metrics["data_status"] == DataStatus.SYNTHETIC_FALLBACK.value
    assert metrics["is_empirical"] is False


def test_provenance_firewall_flight_log_validator():
    validator = RealFlightLogValidator(vehicle_id="test_provenance")
    report = validator.evaluate_flight(csv_path=None)
    assert report.is_real_vehicle_log is False
    assert report.status == DataStatus.SYNTHETIC_FALLBACK
    assert report.is_empirical is False
    assert report.to_dict()["data_status"] == DataStatus.SYNTHETIC_FALLBACK.value


def test_coordinate_transforms_and_specific_force():
    # Test level hover: q = [1, 0, 0, 0] (no rotation)
    q_level = np.array([1.0, 0.0, 0.0, 0.0])
    acc_hover_ned = np.array([0.0, 0.0, 0.0])  # Stationary in NED
    f_body = specific_force_from_acceleration(acc_hover_ned, q_level)
    # Accelerometer measures upward reaction force: [0, 0, -9.80665]
    assert np.allclose(f_body, np.array([0.0, 0.0, -STANDARD_GRAVITY_MPS2]))

    acc_recovered = acceleration_from_specific_force(f_body, q_level)
    assert np.allclose(acc_recovered, acc_hover_ned)

    # Test 90 degree yaw
    q_yaw90 = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    R = quat_to_rot_matrix(q_yaw90)
    q_recovered = rot_matrix_to_quat(R)
    # Signs can be + or - for equivalent quaternion
    assert np.allclose(q_yaw90, q_recovered) or np.allclose(q_yaw90, -q_recovered)


def test_stream_contracts_validation():
    # Valid sensor measurement
    meas = SensorMeasurement(
        vehicle_id="uav_01",
        sensor_type="imu",
        status=DataStatus.REAL,
        values=np.array([0.0, 0.0, -9.81]),
    )
    assert meas.is_valid()

    # Corrupted measurement with NaN
    bad_meas = SensorMeasurement(
        vehicle_id="uav_01",
        sensor_type="imu",
        status=DataStatus.INVALID,
        values=np.array([np.nan, 0.0, -9.81]),
    )
    assert not bad_meas.is_valid()
