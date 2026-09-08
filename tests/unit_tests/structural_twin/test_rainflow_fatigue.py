"""
Unit tests for ASTM E1049-85 Rainflow Cycle Counting and Fatigue Damage (B30).
"""

import pytest
import numpy as np
from drone_sdk.structural_twin.fatigue import (
    RainflowCounter,
    RainflowCycle,
    CompositeFatigueModel,
    FatigueDamageReport,
)


def test_rainflow_simple_cycle():
    """Test standard ASTM E1049 cycle counting on known peak-valley sequence."""
    # Classic waveform: 0 -> 100 -> -50 -> 50 -> 0 MPa
    signal = np.array([0.0, 100.0, -50.0, 50.0, 0.0]) * 1e6
    cycles = RainflowCounter.count_cycles(signal)

    assert len(cycles) > 0
    for c in cycles:
        assert isinstance(c, RainflowCycle)
        assert c.stress_range_pa > 0
        assert c.count in [0.5, 1.0]


def test_basquin_sn_cycles_to_failure():
    """Test S-N curve cycle life estimation."""
    model = CompositeFatigueModel(
        sigma_ult_pa=700.0e6,
        fatigue_strength_coeff_pa=600.0e6,
        basquin_exponent=-0.095,
    )

    # Low stress -> high cycles
    n_low = model.cycles_to_failure(stress_range_pa=100.0e6, stress_mean_pa=0.0)
    # High stress -> lower cycles
    n_high = model.cycles_to_failure(stress_range_pa=500.0e6, stress_mean_pa=0.0)

    assert n_low > n_high
    assert n_low > 1e6
    assert n_high < 1e6


def test_composite_fatigue_cumulative_damage():
    """Test Palmgren-Miner cumulative damage and remaining life."""
    model = CompositeFatigueModel()

    # Cyclic stress history (e.g. 100 Hz flight vibration for 10 seconds)
    t = np.linspace(0, 10, 1000)
    stress_history_pa = (150.0 + 80.0 * np.sin(2 * np.pi * 5 * t) + 30.0 * np.sin(2 * np.pi * 12 * t)) * 1e6

    report = model.evaluate_fatigue_damage(stress_history_pa)

    assert isinstance(report, FatigueDamageReport)
    assert report.cumulative_damage_index >= 0.0
    assert report.total_cycles_counted > 0
    assert report.estimated_equivalent_flights_to_fail > 0.0
    assert report.max_stress_range_mpa > 0.0
    assert report.fatigue_life_margin <= 1.0


def test_zero_stress_history():
    """Test edge case with constant/zero stress."""
    model = CompositeFatigueModel()
    flat_stress = np.zeros(100)
    report = model.evaluate_fatigue_damage(flat_stress)
    assert report.cumulative_damage_index == 0.0
    assert report.total_cycles_counted == 0
    assert report.is_safe is True
