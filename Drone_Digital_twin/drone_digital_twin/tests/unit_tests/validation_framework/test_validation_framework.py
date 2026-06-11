"""Tests for drone_sdk.validation_framework (Module 13)."""
from __future__ import annotations
import json
import math
import numpy as np
import pytest

from drone_sdk.validation_framework import (
    BenchmarkSuites, MetricCalculator, ValidationLevel,
    ValidationResult, ValidationStatus, ValidationSuite,
    ValidationTest, ValidationThreshold,
)


@pytest.fixture
def perfect_signals():
    t = np.linspace(0, 10, 1000)
    sig = np.sin(2 * math.pi * 0.5 * t)
    return sig, sig.copy()          # predicted == reference

@pytest.fixture
def noisy_signals():
    np.random.seed(42)
    t   = np.linspace(0, 10, 1000)
    ref = np.sin(2 * math.pi * 0.5 * t)
    pred = ref + np.random.normal(0, 0.05, len(ref))
    return pred, ref

@pytest.fixture
def bad_signals():
    np.random.seed(42)
    ref  = np.zeros(100)
    pred = np.ones(100) * 5.0       # large systematic error
    return pred, ref


class TestMetricCalculator:
    def test_perfect_prediction_zero_rmse(self, perfect_signals):
        p, r = perfect_signals
        m = MetricCalculator.compute(p, r)
        assert m.rmse < 1e-12
        assert m.mae  < 1e-12

    def test_perfect_prediction_r2_one(self, perfect_signals):
        p, r = perfect_signals
        m = MetricCalculator.compute(p, r)
        assert abs(m.r2 - 1.0) < 1e-6

    def test_noisy_metrics_positive(self, noisy_signals):
        p, r = noisy_signals
        m = MetricCalculator.compute(p, r)
        assert m.rmse  > 0
        assert m.mae   > 0
        assert m.max_ae > 0

    def test_noisy_r2_near_one(self, noisy_signals):
        p, r = noisy_signals
        m = MetricCalculator.compute(p, r)
        assert m.r2 > 0.99

    def test_bad_r2_low(self, bad_signals):
        p, r = bad_signals
        m = MetricCalculator.compute(p, r)
        assert m.r2 < 0.0

    def test_bias_detected(self):
        ref  = np.zeros(100)
        pred = np.ones(100) * 2.0
        m    = MetricCalculator.compute(pred, ref)
        assert abs(m.bias - 2.0) < 1e-9

    def test_nrmse_zero_for_perfect(self, perfect_signals):
        p, r = perfect_signals
        m = MetricCalculator.compute(p, r)
        assert m.nrmse < 1e-10

    def test_sample_count(self, noisy_signals):
        p, r = noisy_signals
        m = MetricCalculator.compute(p, r)
        assert m.n == 1000

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            MetricCalculator.compute(np.ones(10), np.ones(20))

    def test_metrics_to_dict(self, noisy_signals):
        p, r = noisy_signals
        d = MetricCalculator.compute(p, r).to_dict()
        for key in ("mae", "rmse", "max_ae", "nrmse", "r2", "bias", "n"):
            assert key in d

    def test_repr(self, noisy_signals):
        p, r = noisy_signals
        m = MetricCalculator.compute(p, r)
        assert "RMSE" in repr(m)

    def test_is_good_perfect(self, perfect_signals):
        p, r = perfect_signals
        m = MetricCalculator.compute(p, r)
        assert m.is_good

    def test_compute_fft(self, perfect_signals):
        dt = 0.01
        fft = MetricCalculator.compute_fft(*perfect_signals, dt=dt)
        assert "dominant_freq_predicted_hz" in fft
        assert "psd_correlation" in fft


class TestValidationThreshold:
    def test_pass_when_all_met(self, noisy_signals):
        p, r  = noisy_signals
        m     = MetricCalculator.compute(p, r)
        thr   = ValidationThreshold(rmse=1.0, r2_min=0.5)
        st, _ = thr.check(m)
        assert st == ValidationStatus.PASS

    def test_fail_rmse(self, bad_signals):
        p, r  = bad_signals
        m     = MetricCalculator.compute(p, r)
        thr   = ValidationThreshold(rmse=0.01)
        st, msg = thr.check(m)
        assert st == ValidationStatus.FAIL
        assert "RMSE" in msg

    def test_fail_r2(self, bad_signals):
        p, r = bad_signals
        m    = MetricCalculator.compute(p, r)
        thr  = ValidationThreshold(r2_min=0.99)
        st, _ = thr.check(m)
        assert st == ValidationStatus.FAIL

    def test_no_thresholds_always_pass(self, bad_signals):
        p, r = bad_signals
        m    = MetricCalculator.compute(p, r)
        thr  = ValidationThreshold()
        st, _ = thr.check(m)
        assert st == ValidationStatus.PASS

    def test_to_dict_excludes_none(self):
        thr = ValidationThreshold(rmse=0.1)
        d   = thr.to_dict()
        assert "rmse" in d
        assert "mae"  not in d


class TestValidationTest:
    @pytest.fixture
    def test_obj(self):
        return ValidationTest(
            "position_x", ValidationLevel.INTEGRATION,
            signal_name="x",
            threshold=ValidationThreshold(rmse=0.5, r2_min=0.99),
        )

    def test_passes_with_good_data(self, test_obj, noisy_signals):
        p, r   = noisy_signals
        result = test_obj.run(p, r)
        assert result.status == ValidationStatus.PASS

    def test_fails_with_bad_data(self, test_obj, bad_signals):
        p, r   = bad_signals
        result = test_obj.run(p, r)
        assert result.status == ValidationStatus.FAIL

    def test_result_has_metrics(self, test_obj, noisy_signals):
        p, r   = noisy_signals
        result = test_obj.run(p, r)
        assert result.metrics is not None

    def test_duration_recorded(self, test_obj, noisy_signals):
        p, r   = noisy_signals
        result = test_obj.run(p, r)
        assert result.duration_s >= 0

    def test_result_to_dict(self, test_obj, noisy_signals):
        p, r = noisy_signals
        result = test_obj.run(p, r)
        d = result.to_dict()
        assert "status"      in d
        assert "metrics"     in d
        assert "test_name"   in d
        assert "signal_name" in d

    def test_repr(self, test_obj, noisy_signals):
        p, r = noisy_signals
        r_ = test_obj.run(p, r)
        assert "PASS" in repr(r_) or "FAIL" in repr(r_)


class TestValidationSuite:
    @pytest.fixture
    def suite(self):
        s = ValidationSuite("Test Suite")
        for name in ["x", "y", "z"]:
            s.add_test(ValidationTest(
                f"pos_{name}", ValidationLevel.INTEGRATION,
                signal_name=name,
                threshold=ValidationThreshold(rmse=0.5),
            ))
        return s

    def test_run_all_pass(self, suite):
        np.random.seed(42)
        n = 500
        data = {k: np.sin(np.linspace(0,10,n)) for k in ["x","y","z"]}
        noisy = {k: v + np.random.normal(0,.01,n) for k,v in data.items()}
        results = suite.run_all(noisy, data)
        assert all(r.status == ValidationStatus.PASS for r in results)

    def test_run_all_skips_missing(self, suite):
        pred = {"x": np.ones(100)}
        ref  = {"x": np.ones(100)}
        results = suite.run_all(pred, ref)
        skipped = [r for r in results if r.status == ValidationStatus.SKIP]
        assert len(skipped) == 2   # y and z skipped

    def test_n_pass_count(self, suite):
        np.random.seed(0)
        n = 200
        data = {k: np.random.randn(n) for k in ["x","y","z"]}
        suite.run_all(data, data)
        assert suite.n_pass == 3

    def test_n_fail_count(self, suite):
        pred = {k: np.ones(100)*10 for k in ["x","y","z"]}
        ref  = {k: np.zeros(100)   for k in ["x","y","z"]}
        suite.run_all(pred, ref)
        assert suite.n_fail == 3

    def test_overall_pass_true(self, suite):
        data = {k: np.zeros(100) for k in ["x","y","z"]}
        suite.run_all(data, data)
        assert suite.overall_pass

    def test_overall_pass_false_on_failure(self, suite):
        pred = {k: np.ones(100)*5 for k in ["x","y","z"]}
        ref  = {k: np.zeros(100)  for k in ["x","y","z"]}
        suite.run_all(pred, ref)
        assert not suite.overall_pass

    def test_generate_report(self, suite):
        data = {k: np.zeros(100) for k in ["x","y","z"]}
        suite.run_all(data, data)
        report = suite.generate_report()
        assert "summary"  in report
        assert "results"  in report
        assert report["summary"]["total"] == 3

    def test_save_report(self, suite, tmp_path):
        data = {k: np.zeros(100) for k in ["x","y","z"]}
        suite.run_all(data, data)
        path = suite.save_report(str(tmp_path / "report.json"))
        assert path.exists()
        d = json.loads(path.read_text())
        assert "summary" in d

    def test_repr(self, suite):
        assert "ValidationSuite" in repr(suite)

    def test_chaining(self):
        s = (ValidationSuite("Chain")
             .add_test(ValidationTest("t1", ValidationLevel.UNIT, "a"))
             .add_test(ValidationTest("t2", ValidationLevel.UNIT, "b")))
        assert len(s._tests) == 2


class TestBenchmarkSuites:
    def test_ekf_suite_has_9_tests(self):
        s = BenchmarkSuites.ekf_position_suite()
        assert len(s._tests) == 9

    def test_dynamics_suite_has_6_tests(self):
        s = BenchmarkSuites.dynamics_suite()
        assert len(s._tests) == 6

    def test_battery_suite_has_3_tests(self):
        s = BenchmarkSuites.battery_suite()
        assert len(s._tests) == 3

    def test_aero_suite_has_3_tests(self):
        s = BenchmarkSuites.aero_suite()
        assert len(s._tests) == 3

    def test_ekf_suite_runs_perfectly(self):
        suite = BenchmarkSuites.ekf_position_suite()
        sigs  = ["x","y","z","vx","vy","vz","roll","pitch","yaw"]
        n = 500
        data = {s: np.sin(np.linspace(0, 10, n)) for s in sigs}
        suite.run_all(data, data)
        assert suite.overall_pass

    def test_battery_suite_fails_on_bad_data(self):
        suite = BenchmarkSuites.battery_suite()
        pred  = {"voltage": np.ones(100)*20, "soc": np.ones(100)*2,
                 "current": np.ones(100)*50}
        ref   = {"voltage": np.ones(100)*15, "soc": np.ones(100)*0.8,
                 "current": np.ones(100)*10}
        suite.run_all(pred, ref)
        assert suite.n_fail > 0
