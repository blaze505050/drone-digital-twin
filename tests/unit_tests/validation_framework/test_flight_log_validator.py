"""Tests for RealFlightLogValidator and real flight trajectory evaluation."""
import os
import tempfile
import numpy as np
import pytest

from drone_sdk.validation_framework import RealFlightLogValidator, RealFlightValidationReport


def test_real_flight_log_evaluation():
    validator = RealFlightLogValidator(vehicle_id="test_firefly")
    report = validator.evaluate_flight()

    assert isinstance(report, RealFlightValidationReport)
    assert report.num_samples > 100
    assert report.duration_s > 5.0
    assert report.total_flight_dist_m > 5.0

    # ATE RMSE should be tightly bounded (< 35 cm) with MEKF fusing sensor stream
    assert report.pos_ate_rmse_m < 0.35
    assert report.vel_rmse_m_s < 0.40
    assert report.mean_twin_health > 0.70

    report_dict = report.to_dict()
    assert "ate_rmse_cm" in report_dict
    assert "twin_health_pct" in report_dict


def test_real_flight_csv_parsing():
    validator = RealFlightLogValidator(vehicle_id="test_csv_drone")

    # Create temporary flight log CSV
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tmp:
        tmp.write("timestamp,ax,ay,az,gx,gy,gz,x,y,z\n")
        for i in range(50):
            t = i * 0.02
            tmp.write(f"{t:.3f},0.0,0.0,-9.81,0.0,0.0,0.0,{t:.2f},0.0,-5.0\n")
        tmp_path = tmp.name

    try:
        report = validator.evaluate_flight(csv_path=tmp_path, flight_name="custom_flight")
        assert report.is_real_vehicle_log is True
        assert report.num_samples == 50
        assert report.pos_ate_rmse_m < 0.20
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
