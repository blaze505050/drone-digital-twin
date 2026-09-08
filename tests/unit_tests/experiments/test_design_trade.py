"""
Unit tests for Measured Design Trade Study & Optimization (B17).
Verifies:
- Propeller candidate sweeps (9" vs 10" vs 11").
- Payload weight increment analysis (0g to 1000g).
- Calculation of T/W ratio, hover throttle %, endurance, and safety criteria.
"""
import numpy as np
import pytest

from drone_sdk.experiments.design_trade import DesignTradeStudy


def test_propeller_and_payload_trade_study():
    study = DesignTradeStudy()
    results = study.evaluate_propeller_and_payload_trades(
        payload_increments_kg=[0.0, 0.500],
    )

    # 3 props * 2 payloads = 6 points
    assert len(results) == 6

    for r in results:
        # Physical consistency checks
        assert r.total_mass_kg >= 1.40
        assert r.thrust_to_weight_ratio > 1.2
        assert 20.0 <= r.hover_throttle_pct <= 90.0
        assert r.hover_power_w > 100.0
        assert r.hover_endurance_min > 5.0
        assert r.hover_efficiency_g_per_w > 3.0

    # 11" prop should have higher hover efficiency (g/W) than 9" prop for the same 0g payload
    prop_9_res = [r for r in results if "9x4.5" in r.propeller_name and r.payload_mass_kg == 0.0][0]
    prop_11_res = [r for r in results if "11x4.7" in r.propeller_name and r.payload_mass_kg == 0.0][0]

    assert prop_11_res.hover_efficiency_g_per_w > prop_9_res.hover_efficiency_g_per_w
    assert prop_11_res.hover_power_w < prop_9_res.hover_power_w
    assert prop_11_res.hover_endurance_min > prop_9_res.hover_endurance_min

    # Verify formatting output
    table_str = study.format_trade_table(results)
    assert "| Propeller | Payload (kg) |" in table_str
    assert "APC 10x4.5 MR" in table_str
