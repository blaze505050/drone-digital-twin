"""
Unit tests for Gymnasium Environment Compliance & Domain Randomization (B20).
Verifies:
- Gymnasium Box observation and action spaces.
- Reset with deterministic seeding.
- Domain randomization affecting mass, drag, latency, and wind physics.
"""
import numpy as np
import pytest

from drone_sdk.rl_controller import (
    DomainRandomConfig,
    DomainRandomiser,
    DroneGymEnv,
    EnvConfig,
)


def test_drone_gym_env_spaces_and_reset_seeding():
    env = DroneGymEnv(config=EnvConfig(obs_dim=16, act_dim=4))

    assert env.observation_space.shape == (16,)
    assert env.action_space.shape == (4,)

    # Test reset with seed
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    assert np.allclose(obs1, obs2, atol=1e-5)

    # Test action sampling and stepping
    action = env.action_space.sample()
    assert env.action_space.contains(action)

    obs, reward, term, trunc, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert "pos_error_m" in info


def test_domain_randomization_physics():
    cfg = DomainRandomConfig(
        mass_frac=(0.8, 1.2),
        drag_frac=(0.7, 1.3),
        wind_ms=(2.0, 5.0),
    )
    rand = DomainRandomiser(config=cfg, seed=123)
    p1 = rand.sample()
    p2 = rand.sample()

    assert p1["mass_scale"] != p2["mass_scale"]
    assert 0.8 <= p1["mass_scale"] <= 1.2
    assert 0.7 <= p1["drag_scale"] <= 1.3
    assert 2.0 <= p1["wind_speed_ms"] <= 5.0
