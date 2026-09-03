"""Tests for EuRoC MAV and Zurich UAV benchmark loaders."""
import numpy as np
import pytest

from drone_sdk.validation_framework import (
    BenchmarkTrajectory,
    EuRoCDatasetLoader,
    ZurichUAVDatasetLoader,
)


def test_euroc_benchmark_loader():
    loader = EuRoCDatasetLoader(sequence="V1_01_easy")
    traj = loader.trajectory

    assert isinstance(traj, BenchmarkTrajectory)
    assert traj.name == "EuRoC_V1_01_easy_Benchmark"
    assert len(traj.timestamps_s) == 400
    assert traj.pos_ned.shape == (400, 3)
    assert traj.vel_ned.shape == (400, 3)
    assert traj.quat_wxyz.shape == (400, 4)
    assert traj.duration_s > 15.0
    assert traj.total_distance_m > 10.0


def test_euroc_evaluate_estimator():
    loader = EuRoCDatasetLoader(sequence="V1_01_easy")
    traj = loader.trajectory

    # Perfect prediction test
    perfect_pos = traj.pos_ned.copy()
    perfect_vel = traj.vel_ned.copy()
    results = loader.evaluate_state_estimator(perfect_pos, perfect_vel)

    assert results["pos_rmse_m"] == 0.0
    assert results["r2"] == 1.0
    assert results["vel_rmse_ms"] == 0.0

    # Noisy prediction test
    noisy_pos = traj.pos_ned + np.random.normal(0, 0.05, traj.pos_ned.shape)
    noisy_results = loader.evaluate_state_estimator(noisy_pos)
    assert noisy_results["pos_rmse_m"] > 0.0
    assert noisy_results["r2"] > 0.90


def test_zurich_benchmark_loader():
    loader = ZurichUAVDatasetLoader()
    traj = loader.trajectory

    assert isinstance(traj, BenchmarkTrajectory)
    assert len(traj.timestamps_s) == 300
    assert traj.pos_ned.shape == (300, 3)

    metrics = loader.evaluate_canyon_tracking(traj.pos_ned)
    assert metrics["canyon_rmse_m"] == 0.0
