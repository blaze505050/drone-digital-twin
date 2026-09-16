"""
Unit tests for Digital Twin comparison, residual tracking, and calibration.
"""
import numpy as np
import pytest

import dronepy


def test_twin_comparison_metrics():
    d1 = dronepy.Drone.quadcopter(mass=1.5)
    d2 = dronepy.Drone.quadcopter(mass=1.55)

    f1 = d1.simulate(duration=2.0)
    f2 = d2.simulate(duration=2.0)

    comp = dronepy.TwinComparison.compare(f1, f2)
    assert comp.rmse_position >= 0.0
    assert comp.rmse_velocity >= 0.0
    assert len(comp.position_residuals) == len(comp.time)


def test_digital_twin_calibrator():
    drone = dronepy.Drone.quadcopter(mass=1.5)
    f = drone.simulate(duration=2.0)

    cal = dronepy.DigitalTwinCalibrator()
    cal_drone, result = cal.calibrate(drone, f)

    assert isinstance(result, dronepy.CalibrationResult)
    assert result.nominal_params["mass_kg"] == pytest.approx(1.5)
    summary_str = result.summary()
    assert "DronePy Calibration Result" in summary_str
