"""Tests for drone_sdk.rl_controller (Module 19)."""
from __future__ import annotations
import math
import numpy as np
import pytest

from drone_sdk.rl_controller import (
    CurriculumManager, DomainRandomConfig, DomainRandomiser,
    DroneGymEnv, DroneTask, EnvConfig, RewardShaper, RewardWeights,
)


class TestDomainRandomiser:
    @pytest.fixture
    def rand(self):
        return DomainRandomiser(seed=42)

    def test_sample_returns_dict(self, rand):
        p = rand.sample()
        assert isinstance(p, dict)

    def test_sample_has_required_keys(self, rand):
        p = rand.sample()
        for k in ("mass_scale","drag_scale","motor_scale_1","latency_ms","wind_speed_ms"):
            assert k in p

    def test_mass_scale_in_range(self, rand):
        for _ in range(20):
            p = rand.sample()
            assert 0.7 <= p["mass_scale"] <= 1.4

    def test_nominal_returns_ones(self, rand):
        n = rand.nominal()
        assert n["mass_scale"] == 1.0
        assert n["drag_scale"] == 1.0

    def test_different_seeds_give_different_samples(self):
        r1 = DomainRandomiser(seed=1).sample()
        r2 = DomainRandomiser(seed=2).sample()
        assert r1["mass_scale"] != r2["mass_scale"]

    def test_all_values_finite(self, rand):
        p = rand.sample()
        assert all(math.isfinite(v) for v in p.values())


class TestRewardShaper:
    @pytest.fixture
    def shaper(self):
        return RewardShaper()

    def test_survival_bonus_positive(self, shaper):
        r = shaper.compute(0.0, 0.0, 0.0, np.ones(4)*0.5)
        assert r > 0

    def test_goal_reached_bonus(self, shaper):
        r_no  = shaper.compute(0.1, 0.0, 0.0, np.ones(4)*0.5, goal_reached=False)
        r_yes = shaper.compute(0.1, 0.0, 0.0, np.ones(4)*0.5, goal_reached=True)
        assert r_yes > r_no + 40

    def test_crash_penalty_large_negative(self, shaper):
        r = shaper.compute(0.0, 0.0, 0.0, np.ones(4)*0.5, crashed=True)
        assert r < -50

    def test_position_error_penalised(self, shaper):
        r_near = shaper.compute(0.1, 0.0, 0.0, np.ones(4)*0.5)
        r_far  = shaper.compute(5.0, 0.0, 0.0, np.ones(4)*0.5)
        assert r_near > r_far

    def test_tilt_penalised(self, shaper):
        r_level = shaper.compute(0.0, 0.0,       0.0, np.ones(4)*0.5)
        r_tilt  = shaper.compute(0.0, math.pi/4, 0.0, np.ones(4)*0.5)
        assert r_level > r_tilt

    def test_reset_clears_prev_action(self, shaper):
        shaper.compute(0.0, 0.0, 0.0, np.ones(4))
        shaper.reset()
        assert np.allclose(shaper._prev_act, np.zeros(4))

    def test_reward_finite(self, shaper):
        r = shaper.compute(1.0, 0.1, 2.0, np.array([0.5, 0.5, 0.5, 0.5]))
        assert math.isfinite(r)


class TestDroneGymEnv:
    @pytest.fixture
    def env(self):
        cfg = EnvConfig(max_episode_steps=100, dt=0.02)
        return DroneGymEnv(config=cfg)

    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_obs_shape(self, env):
        obs, _ = env.reset()
        assert obs.shape == (env.observation_dim,)

    def test_obs_dim_matches_config(self, env):
        assert env.observation_dim == 16

    def test_action_dim(self, env):
        assert env.action_dim == 4

    def test_step_returns_5_tuple(self, env):
        env.reset()
        result = env.step(np.array([0.5, 0.5, 0.5, 0.5]))
        assert len(result) == 5

    def test_step_increments_count(self, env):
        env.reset()
        env.step(np.array([0.5, 0.5, 0.5, 0.5]))
        assert env.step_count == 1

    def test_truncation_at_max_steps(self, env):
        env.reset(seed=0)
        truncated = False
        for _ in range(110):
            _, _, term, trunc, _ = env.step(np.full(4, 0.52))
            if trunc:
                truncated = True
                break
        assert truncated

    def test_reward_finite(self, env):
        env.reset()
        _, r, _, _, _ = env.step(np.full(4, 0.52))
        assert math.isfinite(r)

    def test_info_has_keys(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.full(4, 0.5))
        assert "pos_error_m" in info
        assert "step"        in info
        assert "goal_reached" in info

    def test_action_clipped_above_1(self, env):
        env.reset()
        _, r, _, _, _ = env.step(np.full(4, 5.0))   # Should not crash
        assert math.isfinite(r)

    def test_action_clipped_below_0(self, env):
        env.reset()
        _, r, _, _, _ = env.step(np.full(4, -2.0))
        assert math.isfinite(r)

    def test_obs_after_step_finite(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(np.full(4, 0.52))
        assert np.all(np.isfinite(obs))

    def test_position_property(self, env):
        env.reset()
        pos = env.position
        assert pos.shape == (3,)

    def test_deterministic_with_seed(self, env):
        obs1, _ = env.reset(seed=7)
        obs2, _ = env.reset(seed=7)
        assert np.allclose(obs1, obs2)

    def test_close_no_crash(self, env):
        env.reset()
        env.close()

    def test_multiple_episodes(self, env):
        for ep in range(3):
            obs, _ = env.reset(seed=ep)
            for _ in range(10):
                env.step(np.full(4, 0.52))


class TestCurriculumManager:
    @pytest.fixture
    def curriculum(self):
        return CurriculumManager(advance_threshold=0.75, window=10)

    def test_initial_stage_0(self, curriculum):
        assert curriculum.stage == 0

    def test_stage_name(self, curriculum):
        assert curriculum.stage_name == "hover"

    def test_success_rate_zero_initially(self, curriculum):
        assert curriculum.success_rate == 0.0

    def test_advances_on_high_success(self, curriculum):
        for _ in range(10):
            curriculum.record_episode(True, 45.0)
        assert curriculum.stage >= 1

    def test_does_not_advance_on_low_success(self, curriculum):
        for _ in range(10):
            curriculum.record_episode(False, -10.0)
        assert curriculum.stage == 0

    def test_success_rate_computed(self, curriculum):
        for _ in range(8):
            curriculum.record_episode(True, 40.0)
        for _ in range(2):
            curriculum.record_episode(False, -5.0)
        assert abs(curriculum.success_rate - 0.8) < 0.01

    def test_no_advance_beyond_max_stage(self, curriculum):
        # Max stage is len(STAGES) - 1
        max_stage = len(CurriculumManager.STAGES) - 1
        for _ in range(200):
            curriculum.record_episode(True, 50.0)
        assert curriculum.stage <= max_stage

    def test_get_env_config_returns_config(self, curriculum):
        cfg = curriculum.get_env_config()
        assert isinstance(cfg, EnvConfig)

    def test_repr(self, curriculum):
        r = repr(curriculum)
        assert "CurriculumManager" in r
        assert "stage=" in r
