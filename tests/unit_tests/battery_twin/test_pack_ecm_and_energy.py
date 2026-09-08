"""
Unit tests for Battery Pack Thevenin ECM & Energy Prediction Baseline (B12).
Verifies:
- OCV-SOC curve and multi-cell pack voltage scaling.
- Thevenin RC dynamic voltage sag under current loads and relaxation.
- Coulomb counting and cumulative electrical energy (Wh) integration.
- Analytical hover endurance estimation.
"""
import numpy as np
import pytest

from drone_sdk.battery_twin.pack_model import (
    EnergyPredictionBaseline,
    PackConfig,
    PackECMModel,
)
from drone_sdk.configuration.schema import BatteryPackConfig


def test_pack_ecm_voltage_sag_and_recovery():
    cfg = BatteryPackConfig(
        cells_in_series=4,
        cells_in_parallel=1,
        cell_capacity_ah=5.0,
        nominal_cell_voltage=3.7,
        cell_r0=0.015,
        cell_r1=0.010,
        cell_c1=1000.0,
    )
    pack = PackECMModel(cfg)

    # Initial state at 100% SOC
    st0 = pack.state
    assert np.isclose(st0.v_terminal, 4.2 * 4, atol=0.2)
    assert st0.soc == 1.0

    # Apply heavy 20A discharge step for 10 seconds
    for _ in range(10):
        st = pack.step(current_load_a=20.0, dt_sec=1.0)

    # Terminal voltage must drop under IR and RC polarization
    assert st.v_terminal < st0.v_terminal - 0.5
    assert st.soc < 1.0
    assert st.energy_consumed_wh > 0.5

    # Remove load (0A) for 10 seconds and check voltage recovery / relaxation
    v_loaded = st.v_terminal
    for _ in range(10):
        st = pack.step(current_load_a=0.0, dt_sec=1.0)

    assert st.v_terminal > v_loaded  # Voltage relaxed back upward


def test_energy_prediction_baseline_hover_endurance():
    cfg = BatteryPackConfig(
        cells_in_series=4,
        cell_capacity_ah=5.0,
        nominal_cell_voltage=3.7,
    )
    baseline = EnergyPredictionBaseline(pack_config=cfg, vehicle_mass_kg=1.50)

    # Check hover power calculation
    p_hover_w, i_hover_a = baseline.estimate_hover_power()
    assert 120.0 <= p_hover_w <= 280.0  # Typical 1.5kg quad hover power ~180-220W
    assert 8.0 <= i_hover_a <= 20.0

    # Check hover endurance ~ 15-25 minutes
    endurance_sec = baseline.predict_hover_endurance_sec(usable_soc_fraction=0.80)
    endurance_min = endurance_sec / 60.0
    assert 12.0 <= endurance_min <= 30.0

    # Simulate multi-phase mission profile
    # 60s hover at 180W + 30s climb at 300W
    res = baseline.simulate_mission_profile([
        (60.0, 180.0),
        (30.0, 300.0),
    ])
    assert res["total_duration_sec"] == 90.0
    assert res["total_energy_consumed_wh"] > 4.0
    assert res["final_soc"] < 1.0
    assert res["is_cutoff_triggered"] is False
