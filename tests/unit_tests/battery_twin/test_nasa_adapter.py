"""Tests for NASA Battery Dataset Adapter and degradation model calibration."""
import numpy as np
import pytest

from drone_sdk.battery_twin import (
    CellParameters,
    DegradationModel,
    TheveninECM,
    NASABatteryDatasetAdapter,
    NASACycleRecord,
)


def test_nasa_adapter_loading():
    adapter = NASABatteryDatasetAdapter(cell_id="B0005")
    assert adapter.total_cycles == 168

    rec1 = adapter.get_cycle(1)
    assert isinstance(rec1, NASACycleRecord)
    assert rec1.cycle_index == 1
    assert rec1.capacity_ah > 1.80
    assert rec1.soh >= 0.95
    assert len(rec1.voltage_v) > 10
    assert len(rec1.current_a) > 10

    # End of life cycle (~168) has degraded capacity
    rec_eol = adapter.get_cycle(168)
    assert rec_eol.capacity_ah < rec1.capacity_ah
    assert rec_eol.soh < 0.78


def test_nasa_adapter_invalid_cycle():
    adapter = NASABatteryDatasetAdapter(cell_id="B0005")
    with pytest.raises(IndexError):
        adapter.get_cycle(0)
    with pytest.raises(IndexError):
        adapter.get_cycle(999)


def test_nasa_adapter_capacity_fade_series():
    adapter = NASABatteryDatasetAdapter(cell_id="B0005")
    cycles, soh = adapter.extract_capacity_fade_series()

    assert len(cycles) == 168
    assert len(soh) == 168
    assert cycles[0] == 1
    assert cycles[-1] == 168
    assert soh[0] > soh[-1]


def test_nasa_degradation_calibration():
    adapter = NASABatteryDatasetAdapter(cell_id="B0005")
    model = DegradationModel()

    initial_alpha = getattr(model, "_a", 0.02)
    calibrated_a = adapter.calibrate_degradation_model(model, temperature_c=24.0)

    assert calibrated_a > 0.0
    # Model capacity_fade should work with the calibrated constant
    soh_50 = model.capacity_fade(50, temperature_c=24.0)
    soh_150 = model.capacity_fade(150, temperature_c=24.0)
    assert 0.75 < soh_50 <= 1.0
    assert 0.60 < soh_150 < soh_50


def test_nasa_ecm_validation():
    adapter = NASABatteryDatasetAdapter(cell_id="B0005")
    cell_params = CellParameters(capacity_ah=2.0)
    ecm = TheveninECM(cell_params)

    metrics = adapter.validate_ecm(ecm, cycle_index=1)
    assert "mae_v" in metrics
    assert "rmse_v" in metrics
    assert metrics["rmse_v"] < 0.50  # Reasonable voltage error
