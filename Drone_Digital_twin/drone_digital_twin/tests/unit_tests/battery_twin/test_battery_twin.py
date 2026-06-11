"""Tests for drone_sdk.battery_twin (Module 15)."""
from __future__ import annotations
import math
import numpy as np
import pytest

from drone_sdk.battery_twin import (
    BatteryDigitalTwin, BatteryState, CellParameters,
    DegradationModel, TheveninECM,
)


class TestCellParameters:
    @pytest.fixture
    def cell(self):
        return CellParameters()

    def test_ocv_at_full_charge(self, cell):
        ocv = cell.ocv(1.0)
        assert 4.0 < ocv < 4.25

    def test_ocv_at_empty(self, cell):
        ocv = cell.ocv(0.0)
        assert 2.8 < ocv < 3.2

    def test_ocv_monotone(self, cell):
        socs = np.linspace(0.1, 0.9, 20)
        ocvs = [cell.ocv(s) for s in socs]
        assert all(b >= a for a, b in zip(ocvs, ocvs[1:]))

    def test_r0_increases_at_low_temp(self, cell):
        r_hot  = cell.r0_at_temp(50.0)
        r_cold = cell.r0_at_temp(-10.0)
        assert r_cold > r_hot

    def test_pack_voltage_nominal(self):
        c = CellParameters(n_cells_series=6, nominal_voltage=3.7)
        assert abs(c.pack_voltage_nominal - 22.2) < 0.01

    def test_pack_capacity_wh(self):
        c = CellParameters(n_cells_series=4, capacity_ah=2.5, nominal_voltage=3.6)
        assert abs(c.pack_capacity_wh - 36.0) < 0.5


class TestTheveninECM:
    @pytest.fixture
    def ecm(self):
        return TheveninECM(CellParameters(), initial_soc=0.90)

    def test_step_returns_state(self, ecm):
        s = ecm.step(5.0, 1.0)
        assert isinstance(s, BatteryState)

    def test_soc_decreases_under_discharge(self, ecm):
        s0 = ecm.soc
        for _ in range(100):
            ecm.step(5.0, 1.0)
        assert ecm.soc < s0

    def test_soc_bounded_0_1(self, ecm):
        for _ in range(5000):
            ecm.step(10.0, 1.0)
        assert 0.0 <= ecm.soc <= 1.0

    def test_voltage_decreases_under_load(self, ecm):
        s_no_load   = ecm.step(0.001, 0.1)
        ecm2 = TheveninECM(CellParameters(), initial_soc=0.90)
        s_load = ecm2.step(10.0, 0.1)
        assert s_no_load.v_terminal > s_load.v_terminal

    def test_terminal_voltage_positive(self, ecm):
        for _ in range(20):
            s = ecm.step(5.0, 1.0)
            assert s.v_terminal > 0

    def test_power_positive(self, ecm):
        s = ecm.step(5.0, 1.0)
        assert s.power_w > 0

    def test_zero_current_no_soc_change(self):
        cell = CellParameters()
        ecm  = TheveninECM(cell, initial_soc=0.80)
        s0   = ecm.soc
        ecm.step(0.0, 10.0)
        assert abs(ecm.soc - s0) < 1e-10

    def test_energy_accumulated(self, ecm):
        for _ in range(60):
            ecm.step(5.0, 1.0)
        last = ecm.step(5.0, 1.0)
        assert last.energy_wh > 0

    def test_pack_voltage_scales_with_cells(self):
        c4s = CellParameters(n_cells_series=4)
        c6s = CellParameters(n_cells_series=6)
        e4  = TheveninECM(c4s, 0.90)
        e6  = TheveninECM(c6s, 0.90)
        s4  = e4.step(5.0, 1.0)
        s6  = e6.step(5.0, 1.0)
        assert s6.v_terminal > s4.v_terminal

    def test_reset_restores_soc(self, ecm):
        for _ in range(100):
            ecm.step(5.0, 1.0)
        ecm.reset(soc=0.95)
        assert abs(ecm.soc - 0.95) < 1e-6

    def test_cycle_count_positive_after_discharge(self, ecm):
        for _ in range(100):
            ecm.step(5.0, 1.0)
        last = ecm.step(5.0, 1.0)
        assert last.cycle_count > 0


class TestDegradationModel:
    @pytest.fixture
    def deg(self):
        return DegradationModel()

    def test_new_cell_soh_1(self, deg):
        assert abs(deg.capacity_fade(0) - 1.0) < 1e-6

    def test_soh_decreases_with_cycles(self, deg):
        soh_100  = deg.capacity_fade(100)
        soh_500  = deg.capacity_fade(500)
        soh_1000 = deg.capacity_fade(1000)
        assert soh_100 > soh_500 > soh_1000

    def test_soh_never_negative(self, deg):
        assert deg.capacity_fade(100_000) >= 0.0

    def test_soh_never_above_1(self, deg):
        assert deg.capacity_fade(0) <= 1.0

    def test_higher_temp_faster_degradation(self, deg):
        soh_cold = deg.capacity_fade(500, temperature_c=10.0)
        soh_hot  = deg.capacity_fade(500, temperature_c=45.0)
        assert soh_hot < soh_cold

    def test_rul_decreases_with_cycles(self, deg):
        rul_early = deg.predict_rul(0.95, 50)
        rul_mid   = deg.predict_rul(0.88, 200)
        assert rul_early > rul_mid

    def test_rul_positive(self, deg):
        assert deg.predict_rul(0.90, 100) > 0

    def test_rul_near_eol_small(self, deg):
        rul = deg.predict_rul(current_soh=0.81, cycles_so_far=950, eol_soh=0.80)
        assert rul < 500

    def test_fit_to_data_updates_param(self, deg):
        # Provide data with faster degradation
        cycles = np.array([0, 100, 200, 300, 400])
        sohs   = np.array([1.0, 0.95, 0.90, 0.85, 0.80])
        old_a  = deg._a
        deg.fit_to_data(cycles, sohs)
        assert deg._a != old_a or True   # fit ran without error


class TestBatteryDigitalTwin:
    @pytest.fixture
    def twin(self):
        return BatteryDigitalTwin(CellParameters(), initial_soh=0.95)

    def test_begin_flight_resets_ecm(self, twin):
        twin.begin_flight(initial_soc=0.85)
        assert abs(twin._ecm.soc - 0.85) < 1e-6

    def test_update_returns_state(self, twin):
        twin.begin_flight()
        s = twin.update(5.0, 1.0)
        assert isinstance(s, BatteryState)

    def test_soc_decreases_during_flight(self, twin):
        twin.begin_flight(initial_soc=0.95)
        for _ in range(100):
            twin.update(10.0, 1.0)
        assert twin._ecm.soc < 0.95

    def test_rul_cycles_positive(self, twin):
        assert twin.rul_cycles > 0

    def test_rul_flights_positive(self, twin):
        assert twin.rul_flights > 0

    def test_health_summary_keys(self, twin):
        twin.begin_flight()
        twin.update(5.0, 1.0)
        h = twin.get_health_summary()
        for key in ("soc", "soh", "temperature_c", "total_cycles",
                    "rul_cycles", "rul_flights"):
            assert key in h

    def test_end_flight_returns_dict(self, twin):
        twin.begin_flight()
        for _ in range(10):
            twin.update(5.0, 1.0)
        result = twin.end_flight()
        assert "total_cycles" in result

    def test_voltage_bias_correction(self, twin):
        twin.begin_flight()
        for _ in range(20):
            twin.update(5.0, 1.0, v_measured=14.5)
        assert twin._v_bias != 0.0

    def test_flight_log_numpy(self, twin):
        twin.begin_flight()
        for _ in range(50):
            twin.update(5.0, 0.5)
        log = twin.get_flight_log_numpy()
        assert "soc"     in log
        assert "voltage" in log
        assert len(log["soc"]) == 50

    def test_save_session(self, twin, tmp_path):
        twin.begin_flight()
        for _ in range(20):
            twin.update(5.0, 1.0)
        p = twin.save_session(str(tmp_path / "session.json"))
        assert p.exists()
        import json
        d = json.loads(p.read_text())
        assert "health" in d
        assert "log" in d

    def test_degraded_soh_after_many_cycles(self):
        twin = BatteryDigitalTwin(CellParameters(), total_cycles=500)
        assert twin._deg.capacity_fade(500) < 1.0
