"""
Unit tests for surrogate comparison benchmark (B28).
"""

import pytest
import numpy as np
from drone_sdk.pinn_engine.surrogate_benchmark import (
    SurrogateComparisonBenchmark,
    SurrogatePerformance,
)


def test_surrogate_benchmark_dataset_generation():
    X_train, y_train, X_test, y_test, X_ood, y_ood = SurrogateComparisonBenchmark.generate_airfoil_dataset()

    assert len(X_train) == 100
    assert len(y_train) == 100
    assert len(X_test) == 49
    assert len(X_ood) == 25
    assert not np.isnan(y_train).any()
    assert not np.isnan(y_test).any()
    assert not np.isnan(y_ood).any()


def test_surrogate_benchmark_run():
    benchmark = SurrogateComparisonBenchmark()
    results = benchmark.run_benchmark()

    assert len(results) == 4
    model_names = [r.model_name for r in results]
    assert "Bilinear Lookup Table" in model_names
    assert "Polynomial Regression (Order 2)" in model_names
    assert "Gaussian Process (RBF Kernel)" in model_names
    assert "Physics-Informed Neural Net (PINN)" in model_names

    for r in results:
        assert isinstance(r, SurrogatePerformance)
        assert r.test_rmse >= 0.0
        assert r.ood_rmse >= 0.0
        assert r.inference_latency_us > 0.0
        assert r.memory_footprint_kb > 0.0


def test_surrogate_benchmark_markdown_formatting():
    benchmark = SurrogateComparisonBenchmark()
    results = benchmark.run_benchmark()
    md = benchmark.format_results_markdown(results)

    assert "# Aerodynamic Surrogate Model Comparison Benchmark" in md
    assert "Bilinear Lookup Table" in md
    assert "PINN" in md
    assert "µs/eval" in md
