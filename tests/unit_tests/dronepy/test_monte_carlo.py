"""
Unit tests for Monte Carlo simulation, statistical envelopes, and sweeps.
"""
import numpy as np
import pytest

import dronepy


def test_distribution_sampling():
    norm = dronepy.Distribution.normal(mean=1.5, std_dev=0.05)
    rng = np.random.default_rng(42)
    samples = [norm.sample(rng) for _ in range(100)]
    assert np.mean(samples) == pytest.approx(1.5, abs=0.05)

    uni = dronepy.Distribution.uniform(low=0.1, high=0.3)
    s_uni = uni.sample(rng)
    assert 0.1 <= s_uni <= 0.3


def test_monte_carlo_execution():
    drone = dronepy.Drone.quadcopter(mass=1.5)
    mc = dronepy.MonteCarlo(drone=drone, num_simulations=4, seed=123)
    mc.add_parameter("mass", dronepy.Distribution.normal(1.5, 0.05))

    res = mc.run(duration=1.5, parallel=False)
    assert res.runs == 4
    assert len(res.landing_positions) == 4
    assert len(res.max_altitudes) == 4
    assert res.landing_dispersion_radius_95 >= 0.0
    summary = res.summary()
    assert "cep95_m" in summary


def test_sweep_parameter_study():
    drone = dronepy.Drone.quadcopter(mass=1.5)
    sweep = dronepy.Sweep(drone=drone, parameter_name="mass", values=[1.4, 1.5, 1.6])
    sweep_res = sweep.run(duration=1.5)
    assert len(sweep_res.results) == 3
    assert len(sweep_res.values) == 3
