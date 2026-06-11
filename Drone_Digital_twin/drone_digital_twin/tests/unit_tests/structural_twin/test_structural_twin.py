"""Tests for drone_sdk.structural_twin (Module 16)."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pytest

from drone_sdk.structural_twin import (
    BeamElement, CircularTube, DroneFrameConfig,
    DroneFrameFEM, FatigueMonitor, Material,
    ModalResult, StructuralHealthMonitor,
)


class TestMaterial:
    def test_cf_tube_properties(self):
        m = Material.carbon_fibre_tube()
        assert m.E > 50e9
        assert m.rho > 1000

    def test_aluminium_properties(self):
        m = Material.aluminium_6061()
        assert abs(m.E - 69e9) < 1e9
        assert m.sigma_y > 200e6

    def test_abs_plastic(self):
        m = Material.abs_plastic()
        assert m.E < 5e9
        assert m.rho < 1500


class TestCircularTube:
    @pytest.fixture
    def tube(self):
        return CircularTube(outer_radius=0.006, inner_radius=0.005)

    def test_area_positive(self, tube):
        assert tube.area > 0

    def test_area_formula(self, tube):
        expected = math.pi * (0.006**2 - 0.005**2)
        assert abs(tube.area - expected) < 1e-12

    def test_Iz_positive(self, tube):
        assert tube.Iz > 0

    def test_J_twice_Iz(self, tube):
        assert abs(tube.J - 2 * tube.Iz) < 1e-20

    def test_wall_thickness(self, tube):
        assert abs(tube.wall_thickness - 0.001) < 1e-10

    def test_solid_cylinder_larger_Iz(self):
        solid  = CircularTube(0.006, 0.0)
        hollow = CircularTube(0.006, 0.005)
        assert solid.Iz > hollow.Iz


class TestBeamElement:
    @pytest.fixture
    def beam(self):
        mat = Material.carbon_fibre_tube()
        sec = CircularTube(0.006, 0.005)
        return BeamElement(length=0.25, material=mat, section=sec)

    def test_stiffness_matrix_shape(self, beam):
        K = beam.stiffness_matrix()
        assert K.shape == (12, 12)

    def test_stiffness_matrix_symmetric(self, beam):
        K = beam.stiffness_matrix()
        assert np.allclose(K, K.T, atol=1e-8)

    def test_stiffness_matrix_positive_diagonal(self, beam):
        K = beam.stiffness_matrix()
        assert np.all(np.diag(K) >= 0)

    def test_mass_matrix_shape(self, beam):
        M = beam.mass_matrix_consistent()
        assert M.shape == (12, 12)

    def test_mass_matrix_symmetric(self, beam):
        M = beam.mass_matrix_consistent()
        assert np.allclose(M, M.T, atol=1e-8)

    def test_mass_matrix_positive_diagonal(self, beam):
        M = beam.mass_matrix_consistent()
        assert np.all(np.diag(M) >= 0)

    def test_stiffness_scales_with_E(self):
        mat_stiff  = Material.carbon_fibre_tube()
        mat_soft   = Material.abs_plastic()
        sec        = CircularTube(0.006, 0.005)
        K_stiff    = BeamElement(0.25, mat_stiff, sec).stiffness_matrix()
        K_soft     = BeamElement(0.25, mat_soft,  sec).stiffness_matrix()
        assert K_stiff[0, 0] > K_soft[0, 0]


class TestDroneFrameFEM:
    @pytest.fixture
    def fem(self):
        return DroneFrameFEM(DroneFrameConfig())

    def test_tip_deflection_positive(self, fem):
        d = fem.tip_deflection(tip_load_n=1.0)
        assert d > 0

    def test_tip_deflection_scales_with_load(self, fem):
        d1 = fem.tip_deflection(1.0)
        d2 = fem.tip_deflection(2.0)
        assert abs(d2 / d1 - 2.0) < 0.01

    def test_tip_deflection_small_for_cf(self, fem):
        # CF arm should deflect < 1 mm under 5N tip load
        d = fem.tip_deflection(5.0)
        assert d < 0.001

    def test_tip_stress_positive(self, fem):
        sigma = fem.tip_stress(thrust_n=15.0)
        assert sigma > 0

    def test_tip_stress_scales_with_thrust(self, fem):
        s1 = fem.tip_stress(10.0)
        s2 = fem.tip_stress(20.0)
        assert abs(s2 / s1 - 2.0) < 0.01

    def test_natural_frequency_positive(self, fem):
        f = fem.natural_frequency_first_bending()
        assert f > 0

    def test_natural_frequency_realistic_range(self, fem):
        # CF quadrotor arms typically 100-400 Hz first bending
        f = fem.natural_frequency_first_bending()
        assert 10 < f < 1000

    def test_modal_analysis_returns_3_modes(self, fem):
        modes = fem.modal_analysis()
        assert len(modes.frequencies) == 3

    def test_modal_frequencies_ascending(self, fem):
        modes = fem.modal_analysis()
        freqs = modes.frequencies
        assert freqs[0] < freqs[1] < freqs[2]

    def test_safety_factor_high_for_cf(self, fem):
        sf = fem.safety_factor(thrust_n=15.0)
        assert sf > 3.0   # CF should have high safety factor at operating load

    def test_modal_result_to_dict(self, fem):
        modes = fem.modal_analysis()
        d = modes.to_dict()
        assert "frequencies" in d
        assert len(d["frequencies"]) == 3


class TestStructuralHealthMonitor:
    @pytest.fixture
    def shm(self):
        return StructuralHealthMonitor(DroneFrameFEM(DroneFrameConfig()))

    def test_set_baseline(self, shm):
        f0 = shm.set_baseline()
        assert f0 > 0
        assert shm._baseline_hz == f0

    def test_healthy_at_baseline(self, shm):
        shm.set_baseline()
        result = shm.update_frequency(shm._baseline_hz)
        assert result["status"] == "healthy"
        assert result["damage_index"] == 0.0

    def test_damage_detected_on_freq_drop(self, shm):
        shm.set_baseline()
        f0 = shm._baseline_hz
        # 15% drop → should be warning or critical
        result = shm.update_frequency(f0 * 0.85)
        assert result["damage_index"] > 0.05

    def test_critical_on_large_drop(self, shm):
        shm.set_baseline()
        f0 = shm._baseline_hz
        result = shm.update_frequency(f0 * 0.70)   # 30% drop
        assert result["status"] == "critical"

    def test_upward_shift_not_damage(self, shm):
        shm.set_baseline()
        f0 = shm._baseline_hz
        result = shm.update_frequency(f0 * 1.02)   # Slight stiffening
        assert result["damage_index"] == 0.0

    def test_health_report_keys(self, shm):
        shm.set_baseline()
        shm.update_frequency(shm._baseline_hz)
        report = shm.get_health_report()
        assert "airworthy"        in report
        assert "latest_damage_idx" in report
        assert "n_measurements"   in report

    def test_airworthy_at_healthy(self, shm):
        shm.set_baseline()
        shm.update_frequency(shm._baseline_hz)
        assert shm.get_health_report()["airworthy"]

    def test_not_airworthy_on_severe_damage(self, shm):
        shm.set_baseline()
        shm.update_frequency(shm._baseline_hz * 0.75)
        assert not shm.get_health_report()["airworthy"]

    def test_history_accumulates(self, shm):
        shm.set_baseline()
        for _ in range(5):
            shm.update_frequency(shm._baseline_hz * 0.99)
        assert shm.get_health_report()["n_measurements"] == 5

    def test_save_creates_file(self, shm, tmp_path):
        shm.set_baseline()
        shm.update_frequency(shm._baseline_hz)
        p = shm.save(str(tmp_path / "shm.json"))
        assert p.exists()
        d = json.loads(p.read_text())
        assert "baseline_hz" in d
        assert "history"     in d


class TestFatigueMonitor:
    @pytest.fixture
    def monitor(self):
        return FatigueMonitor(Material.carbon_fibre_tube())

    def test_zero_damage_initially(self, monitor):
        assert monitor.damage_fraction == 0.0

    def test_remaining_life_1_initially(self, monitor):
        assert monitor.remaining_life_fraction == 1.0

    def test_add_stress_history_extracts_cycles(self, monitor):
        stress = np.sin(np.linspace(0, 10*math.pi, 200)) * 1e6
        n = monitor.add_stress_history(stress)
        assert n >= 0

    def test_damage_increases_after_loading(self, monitor):
        # High amplitude stress to accumulate damage
        stress = np.array([0, 500e6, 0, 500e6, 0, 500e6] * 10, dtype=float)
        monitor.add_stress_history(stress)
        assert monitor.damage_fraction > 0

    def test_damage_bounded_0_1(self, monitor):
        for _ in range(10):
            stress = np.array([0, 1000e6, 0, -1000e6] * 50, dtype=float)
            monitor.add_stress_history(stress)
        assert 0.0 <= monitor.damage_fraction <= 1.0

    def test_higher_stress_more_damage(self, monitor):
        m_low  = FatigueMonitor(Material.carbon_fibre_tube())
        m_high = FatigueMonitor(Material.carbon_fibre_tube())
        low_stress  = np.array([0, 100e6, 0, -100e6] * 5, dtype=float)
        high_stress = np.array([0, 400e6, 0, -400e6] * 5, dtype=float)
        m_low.add_stress_history(low_stress)
        m_high.add_stress_history(high_stress)
        assert m_high.damage_fraction >= m_low.damage_fraction

    def test_cycles_to_failure_decreases_with_stress(self, monitor):
        n1 = monitor.cycles_to_failure(100e6)
        n2 = monitor.cycles_to_failure(300e6)
        assert n1 > n2

    def test_get_summary_keys(self, monitor):
        stress = np.sin(np.linspace(0, 4*math.pi, 100)) * 50e6
        monitor.add_stress_history(stress)
        s = monitor.get_summary()
        assert "n_cycles"        in s
        assert "damage_fraction" in s
        assert "remaining_life"  in s
