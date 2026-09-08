"""
Unit tests for Reproducible Benchmark Suite (B27).
Verifies:
- Automated execution of all 5 platform benchmarks.
- Generation of structured JSON and Markdown summary artifacts.
"""
import numpy as np
import pytest

from drone_sdk.experiments.reproducible_benchmark import (
    ReproducibleBenchmarkSuite,
)


def test_reproducible_benchmark_suite_execution(tmp_path):
    suite = ReproducibleBenchmarkSuite()
    payload = suite.execute_all()

    assert payload["suite_version"] == "3.0"
    assert payload["benchmarks_total"] == 5
    assert payload["benchmarks_passed"] >= 4
    assert len(payload["results"]) == 5

    # Check that individual benchmarks have metrics
    b_ids = [r["benchmark_id"] for r in payload["results"]]
    assert "UIUC_BEMT_APC_10x4.7" in b_ids
    assert "NASA_BATTERY_B0005" in b_ids
    assert "EUROC_MAV_V1_01" in b_ids

    # Save to directory and verify file persistence
    md_path, json_path = suite.save_package(tmp_path)
    assert md_path.is_file()
    assert json_path.is_file()
    assert md_path.stat().st_size > 100
    assert json_path.stat().st_size > 100
