"""Tests for UIUCPropellerDatasetLoader aerodynamic benchmarks."""
import numpy as np
import pytest

from drone_sdk.validation_framework import UIUCPropellerDatasetLoader, AeroBenchmarkReport


def test_uiuc_propeller_benchmark():
    loader = UIUCPropellerDatasetLoader()
    assert len(loader.advance_ratios) == 8
    assert loader.advance_ratios[0] == 0.0

    report = loader.evaluate_bemt_solver(rpm=4000.0)
    assert isinstance(report, AeroBenchmarkReport)
    assert report.num_test_points == 8
    assert report.ct_rmse < 0.06
    assert report.cp_rmse < 0.05
    assert report.ct_r2 is not None

    report_dict = report.to_dict()
    assert "ct_rmse" in report_dict
    assert "ct_r2" in report_dict
    assert report_dict["points"] == 8
