"""Tests for drone_sdk.predictive_maintenance (Module 17)."""
from __future__ import annotations
import math
import numpy as np
import pytest

from drone_sdk.predictive_maintenance import (
    AnomalyDetector, FeatureExtractor, FlightLogAnalyser,
    HealthIndex, MaintenanceInterval, MaintenanceScheduler,
    SubsystemHealth,
)


SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
class TestFeatureExtractor:
    @pytest.fixture
    def sine_signal(self):
        t = np.linspace(0, 1, 200)
        return np.sin(2 * math.pi * 10 * t)   # 10 Hz sine

    def test_extract_1d_returns_10_features(self, sine_signal):
        f = FeatureExtractor.extract(sine_signal, dt=0.005)
        assert f.shape == (FeatureExtractor.N_FEATURES_PER_CH,)

    def test_extract_2d_returns_20_features(self, sine_signal):
        sig2 = np.column_stack([sine_signal, sine_signal * 2])
        f    = FeatureExtractor.extract(sig2, dt=0.005)
        assert f.shape == (2 * FeatureExtractor.N_FEATURES_PER_CH,)

    def test_rms_positive_for_nonzero_signal(self, sine_signal):
        f = FeatureExtractor.extract(sine_signal, dt=0.005)
        rms = f[2]   # Index 2 = rms
        assert rms > 0

    def test_mean_near_zero_for_zero_mean(self, sine_signal):
        f    = FeatureExtractor.extract(sine_signal, dt=0.005)
        mean = f[0]
        assert abs(mean) < 0.1

    def test_dominant_freq_near_10hz(self, sine_signal):
        f      = FeatureExtractor.extract(sine_signal, dt=0.005)
        dom_f  = f[7]   # dominant_freq
        assert abs(dom_f - 10.0) < 2.0

    def test_all_finite(self, sine_signal):
        f = FeatureExtractor.extract(sine_signal, dt=0.005)
        assert np.all(np.isfinite(f))

    def test_zero_signal_no_crash(self):
        f = FeatureExtractor.extract(np.zeros(100), dt=0.01)
        assert np.all(np.isfinite(f))

    def test_feature_names_length(self):
        names = FeatureExtractor.feature_names(n_channels=1)
        assert len(names) == FeatureExtractor.N_FEATURES_PER_CH

    def test_feature_names_multichannel(self):
        names = FeatureExtractor.feature_names(n_channels=3)
        assert len(names) == 3 * FeatureExtractor.N_FEATURES_PER_CH

    def test_kurtosis_high_for_impulsive(self):
        # Impulsive signal has high kurtosis
        sig       = np.zeros(200)
        sig[100]  = 10.0   # single spike
        f         = FeatureExtractor.extract(sig, dt=0.005)
        kurtosis  = f[5]
        assert kurtosis > 1.0


# ══════════════════════════════════════════════════════════════════════════════
class TestAnomalyDetector:
    @pytest.fixture
    def normal_data(self):
        np.random.seed(SEED)
        return np.random.randn(300, 10) * 0.5  # Normal operation

    @pytest.fixture
    def trained_detector(self, normal_data):
        det = AnomalyDetector(feature_dim=10, bottleneck=4)
        det.fit(normal_data, n_epochs=100, lr=0.01)
        return det

    def test_fit_sets_trained(self, trained_detector):
        assert trained_detector._trained

    def test_threshold_positive(self, trained_detector):
        assert trained_detector._threshold > 0

    def test_reconstruction_error_normal_low(self, trained_detector, normal_data):
        errors = [trained_detector.reconstruction_error(x) for x in normal_data[:20]]
        assert np.mean(errors) < trained_detector._threshold * 2

    def test_anomaly_score_range(self, trained_detector, normal_data):
        score = trained_detector.anomaly_score(normal_data[0])
        assert 0.0 <= score <= 1.0

    def test_anomaly_detected_for_extreme_input(self, trained_detector):
        np.random.seed(99)
        anomaly = np.random.randn(10) * 100   # Far from normal distribution
        assert trained_detector.is_anomaly(anomaly)

    def test_normal_not_anomaly_most_of_time(self, trained_detector, normal_data):
        # At 95th percentile threshold, ~5% of normal data flagged
        flags = [trained_detector.is_anomaly(x) for x in normal_data[:100]]
        assert sum(flags) <= 15   # Allow up to 15% false positives

    def test_untrained_no_crash(self):
        det   = AnomalyDetector(10)
        score = det.anomaly_score(np.ones(10))
        assert score == 0.0

    def test_chaining(self, normal_data):
        det = AnomalyDetector(10).fit(normal_data, n_epochs=50)
        assert det._trained

    def test_returns_finite_scores(self, trained_detector, normal_data):
        for x in normal_data[:10]:
            s = trained_detector.anomaly_score(x)
            assert math.isfinite(s)


# ══════════════════════════════════════════════════════════════════════════════
class TestHealthIndex:
    @pytest.fixture
    def hi(self):
        h = HealthIndex()
        h.update("battery",   0.90, 0.35)
        h.update("structure", 0.95, 0.25)
        h.update("vibration", 0.88, 0.25)
        h.update("temperature", 1.0, 0.15)
        return h

    def test_overall_in_range(self, hi):
        assert 0.0 <= hi.overall <= 1.0

    def test_overall_weighted_correctly(self, hi):
        expected = (0.90*0.35 + 0.95*0.25 + 0.88*0.25 + 1.0*0.15) / 1.0
        assert abs(hi.overall - expected) < 1e-6

    def test_airworthy_when_healthy(self, hi):
        assert hi.airworthy

    def test_not_airworthy_when_degraded(self):
        h = HealthIndex()
        h.update("battery",   0.30, 0.5)
        h.update("structure", 0.30, 0.5)
        assert not h.airworthy

    def test_critical_subsystems_empty_when_healthy(self, hi):
        assert len(hi.critical_subsystems) == 0

    def test_critical_subsystems_detected(self):
        h = HealthIndex()
        h.update("battery",   0.20, 0.5)   # Critical
        h.update("structure", 0.90, 0.5)
        assert "battery" in h.critical_subsystems

    def test_to_dict_keys(self, hi):
        d = hi.to_dict()
        assert "overall"    in d
        assert "airworthy"  in d
        assert "subsystems" in d

    def test_empty_index_returns_1(self):
        h = HealthIndex()
        assert h.overall == 1.0

    def test_repr(self, hi):
        assert "HealthIndex" in repr(hi)

    def test_subsystem_status_labels(self):
        h = HealthIndex()
        h.update("a", 0.95, 1.0)   # nominal
        h.update("b", 0.70, 1.0)   # degraded
        h.update("c", 0.30, 1.0)   # critical
        assert h._subsystems["a"].status == "nominal"
        assert h._subsystems["b"].status == "degraded"
        assert h._subsystems["c"].status == "critical"


# ══════════════════════════════════════════════════════════════════════════════
class TestMaintenanceScheduler:
    @pytest.fixture
    def scheduler(self):
        return MaintenanceScheduler("drone_test")

    def test_not_due_when_healthy(self, scheduler):
        scheduler.update(50.0, battery_rul_cycles=400, structural_damage=0.02, anomaly_score=0.05)
        assert not scheduler.maintenance_due

    def test_due_when_degraded(self, scheduler):
        scheduler.update(50.0, battery_rul_cycles=10, structural_damage=0.5, anomaly_score=0.8)
        assert scheduler.maintenance_due

    def test_generate_work_order_returns_wo(self, scheduler):
        scheduler.update(50.0, battery_rul_cycles=10, structural_damage=0.5)
        wo = scheduler.generate_work_order()
        assert wo.work_order_id.startswith("WO-")

    def test_work_order_has_tasks(self, scheduler):
        scheduler.update(50.0)
        wo = scheduler.generate_work_order()
        assert len(wo.tasks) >= 1

    def test_critical_battery_priority_1(self, scheduler):
        scheduler.update(50.0, battery_rul_cycles=0.1)
        wo = scheduler.generate_work_order()
        assert wo.priority == 1

    def test_critical_battery_next_flight_interval(self, scheduler):
        scheduler.update(50.0, battery_rul_cycles=0.0)
        wo = scheduler.generate_work_order()
        assert wo.interval == MaintenanceInterval.NEXT_FLIGHT

    def test_100h_routine_tasks_added(self, scheduler):
        scheduler.update(110.0)
        wo = scheduler.generate_work_order()
        assert any("100-hour" in t for t in wo.tasks)

    def test_wo_to_dict(self, scheduler):
        scheduler.update(50.0)
        d = scheduler.generate_work_order().to_dict()
        assert "work_order_id" in d
        assert "tasks"         in d
        assert "priority"      in d
        assert "interval"      in d

    def test_trend_returns_arrays(self, scheduler):
        for h in [10, 20, 30]:
            scheduler.update(float(h))
        trend = scheduler.get_trend()
        assert "flight_hours"   in trend
        assert "overall_health" in trend
        assert len(trend["flight_hours"]) == 3

    def test_save_creates_file(self, scheduler, tmp_path):
        scheduler.update(50.0)
        p = scheduler.save(str(tmp_path / "maintenance.json"))
        assert p.exists()
        import json
        d = json.loads(p.read_text())
        assert "vehicle_id" in d
        assert "health"     in d

    def test_work_order_id_increments(self, scheduler):
        scheduler.update(50.0)
        wo1 = scheduler.generate_work_order()
        wo2 = scheduler.generate_work_order()
        assert wo1.work_order_id != wo2.work_order_id


# ══════════════════════════════════════════════════════════════════════════════
class TestFlightLogAnalyser:
    @pytest.fixture
    def log(self):
        n = 500
        t = np.linspace(0, 100, n)
        return {
            "timestamp_wall": t,
            "altitude_agl":   np.abs(np.sin(t/20)) * 30,
            "groundspeed":    np.abs(np.sin(t/10)) * 8,
            "battery_soc":    np.linspace(0.90, 0.65, n),
            "ax":             np.random.randn(n) * 0.5,
            "ay":             np.random.randn(n) * 0.5,
            "az":             np.random.randn(n) * 0.5 - 9.81,
            "roll":           np.sin(t/15) * 0.2,
            "pitch":          np.sin(t/12) * 0.15,
        }

    def test_analyse_returns_dict(self, log):
        a = FlightLogAnalyser(log)
        r = a.analyse()
        assert isinstance(r, dict)

    def test_flight_duration(self, log):
        r = FlightLogAnalyser(log).analyse()
        assert abs(r["flight_duration_s"] - 100.0) < 1.0

    def test_max_altitude_positive(self, log):
        r = FlightLogAnalyser(log).analyse()
        assert r["max_altitude_agl_m"] > 0

    def test_max_speed_positive(self, log):
        r = FlightLogAnalyser(log).analyse()
        assert r["max_speed_ms"] > 0

    def test_soc_consumed_positive(self, log):
        r = FlightLogAnalyser(log).analyse()
        assert r["soc_consumed"] > 0

    def test_soc_consumed_correct(self, log):
        r = FlightLogAnalyser(log).analyse()
        assert abs(r["soc_consumed"] - 0.25) < 0.01

    def test_vibration_rms_positive(self, log):
        r = FlightLogAnalyser(log).analyse()
        assert r["vibration_rms"] >= 0

    def test_hover_efficiency_range(self, log):
        r = FlightLogAnalyser(log).analyse()
        assert 0.0 <= r["hover_efficiency"] <= 1.0

    def test_attitude_exceedances_dict(self, log):
        r = FlightLogAnalyser(log).analyse()
        assert "roll_exceed_30deg"  in r["attitude_exceedances"]
        assert "pitch_exceed_30deg" in r["attitude_exceedances"]

    def test_empty_log_no_crash(self):
        r = FlightLogAnalyser({}).analyse()
        assert r["flight_duration_s"] == 0.0
