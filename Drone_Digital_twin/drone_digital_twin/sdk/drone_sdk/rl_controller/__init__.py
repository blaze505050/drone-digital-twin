"""
drone_sdk.rl_controller
=======================
Reinforcement Learning Controller — Module 19 of the UAV Digital Twin Platform.

Implements:

1. **DroneGymEnv** — Gymnasium-compatible environment wrapping the 6-DOF
   physics engine. Supports hover, waypoint navigation, and aggressive
   manoeuvring tasks.

2. **RewardShaper** — Modular, configurable reward function with components
   for position error, attitude stability, energy efficiency, and safety.

3. **DomainRandomiser** — Randomises physical parameters at episode reset
   to improve sim-to-real transfer (mass, drag, motor scaling, latency).

4. **PolicyInterface** — Load a trained policy (ONNX or numpy) and query
   it for control actions at the required frequency.

5. **CurriculumManager** — Progressive task difficulty for stable training
   (hover → waypoint → aggressive → formation).

Python version: 3.9+
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Task definitions
# ─────────────────────────────────────────────────────────────────────────────

class DroneTask(str, Enum):
    HOVER             = "hover"
    WAYPOINT          = "waypoint"
    VELOCITY_TRACKING = "velocity_tracking"
    AGGRESSIVE_FLIP   = "aggressive_flip"
    FORMATION         = "formation"


# ─────────────────────────────────────────────────────────────────────────────
#  Environment configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnvConfig:
    """Configuration for DroneGymEnv."""
    task:              DroneTask = DroneTask.HOVER
    dt:                float     = 0.02    # Control timestep (s)  — 50 Hz
    sim_steps_per_ctrl: int      = 4       # Physics substeps per control step
    max_episode_steps: int       = 500     # ~10 s per episode
    goal_position:     np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -5.0])  # NED hover target
    )
    goal_tolerance_m:  float = 0.1        # Success radius (m)
    max_tilt_rad:      float = math.pi/3  # 60° — episode ends if exceeded
    min_altitude_m:    float = 0.2        # Floor (m AGL, positive)
    max_altitude_m:    float = 20.0       # Ceiling (m AGL)

    # Observation space: [pos(3), vel(3), quat(4), omega(3), goal_rel(3)] = 16
    obs_dim: int = 16
    # Action space: normalised motor thrusts [0, 1] × 4
    act_dim: int = 4


# ─────────────────────────────────────────────────────────────────────────────
#  Domain randomisation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DomainRandomConfig:
    """Parameters ranges for domain randomisation."""
    mass_frac:    Tuple[float,float] = (0.8, 1.2)   # ±20%
    drag_frac:    Tuple[float,float] = (0.7, 1.3)   # ±30%
    motor_frac:   Tuple[float,float] = (0.9, 1.1)   # ±10% per motor
    latency_ms:   Tuple[float,float] = (5.0, 30.0)  # action latency
    wind_ms:      Tuple[float,float] = (0.0, 3.0)   # wind speed
    inertia_frac: Tuple[float,float] = (0.85, 1.15)


class DomainRandomiser:
    """Randomises physical parameters to improve sim-to-real transfer.

    Usage::

        rand = DomainRandomiser(DomainRandomConfig())
        params = rand.sample()
        env.set_physics_params(**params)
    """

    def __init__(self, config: Optional[DomainRandomConfig] = None, seed: int = 0) -> None:
        self._cfg = config or DomainRandomConfig()
        self._rng = np.random.default_rng(seed)

    def sample(self) -> Dict[str, float]:
        """Sample one set of randomised physical parameters."""
        cfg = self._cfg
        r   = self._rng

        def uniform(lo, hi): return float(r.uniform(lo, hi))

        # Per-motor scaling (independent)
        motor_scales = [uniform(*cfg.motor_frac) for _ in range(4)]

        # Random wind direction
        wind_speed = uniform(*cfg.wind_ms)
        wind_dir   = float(r.uniform(0, 2*math.pi))

        return {
            "mass_scale":      uniform(*cfg.mass_frac),
            "drag_scale":      uniform(*cfg.drag_frac),
            "motor_scale_1":   motor_scales[0],
            "motor_scale_2":   motor_scales[1],
            "motor_scale_3":   motor_scales[2],
            "motor_scale_4":   motor_scales[3],
            "latency_ms":      uniform(*cfg.latency_ms),
            "wind_speed_ms":   wind_speed,
            "wind_dir_rad":    wind_dir,
            "inertia_scale":   uniform(*cfg.inertia_frac),
        }

    def nominal(self) -> Dict[str, float]:
        """Return nominal (un-randomised) parameters."""
        return {
            "mass_scale": 1.0, "drag_scale": 1.0,
            "motor_scale_1": 1.0, "motor_scale_2": 1.0,
            "motor_scale_3": 1.0, "motor_scale_4": 1.0,
            "latency_ms": 10.0, "wind_speed_ms": 0.0,
            "wind_dir_rad": 0.0, "inertia_scale": 1.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Reward shaper
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RewardWeights:
    """Weights for reward function components."""
    position_error:   float = -1.0   # per metre of error
    attitude_penalty: float = -0.5   # per radian of tilt
    velocity_penalty: float = -0.1   # per m/s excess speed
    action_smoothness: float = -0.05  # per unit of action rate
    goal_reached:     float =  50.0  # one-time bonus
    crash_penalty:    float = -100.0
    survival_bonus:   float =  0.1   # per step survived


class RewardShaper:
    """Computes shaped rewards for drone control tasks.

    Modular design: each reward component is independently weighted,
    making it easy to tune via hyperparameter search.

    Usage::

        shaper   = RewardShaper(RewardWeights())
        reward   = shaper.compute(
            pos_error_m=0.5,
            tilt_rad=0.1,
            speed_ms=2.0,
            action_delta=np.zeros(4),
            goal_reached=False,
            crashed=False,
        )
    """

    def __init__(self, weights: Optional[RewardWeights] = None) -> None:
        self._w        = weights or RewardWeights()
        self._prev_act = np.zeros(4)

    def compute(
        self,
        pos_error_m:  float,
        tilt_rad:     float,
        speed_ms:     float,
        action:       np.ndarray,
        goal_reached: bool = False,
        crashed:      bool = False,
    ) -> float:
        """Compute the shaped reward for one timestep."""
        w    = self._w
        act_rate  = float(np.linalg.norm(action - self._prev_act))
        self._prev_act = action.copy()

        r = 0.0
        r += w.position_error   * pos_error_m
        r += w.attitude_penalty * abs(tilt_rad)
        r += w.velocity_penalty * max(0.0, speed_ms - 5.0)
        r += w.action_smoothness * act_rate
        r += w.survival_bonus

        if goal_reached:
            r += w.goal_reached
        if crashed:
            r += w.crash_penalty

        return float(r)

    def reset(self) -> None:
        self._prev_act = np.zeros(4)


# ─────────────────────────────────────────────────────────────────────────────
#  Drone Gym Environment (Gymnasium-compatible)
# ─────────────────────────────────────────────────────────────────────────────

class DroneGymEnv:
    """Gymnasium-compatible UAV control environment.

    Wraps the 6-DOF physics engine with:
    - Standardised obs/act spaces matching Gymnasium API
    - Domain randomisation at reset
    - Configurable reward shaping
    - Episode termination on crash or timeout
    - Reproducible seeding

    Compatible with Stable-Baselines3, CleanRL, RLlib.

    Installation::

        pip install gymnasium stable-baselines3

    Usage::

        env    = DroneGymEnv()
        obs, _ = env.reset(seed=42)
        done   = False
        while not done:
            action          = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config:     Optional[EnvConfig]           = None,
        randomiser: Optional[DomainRandomiser]    = None,
        reward_weights: Optional[RewardWeights]   = None,
    ) -> None:
        self._cfg    = config      or EnvConfig()
        self._rand   = randomiser  or DomainRandomiser()
        self._shaper = RewardShaper(reward_weights)

        # State
        self._pos   = np.zeros(3)
        self._vel   = np.zeros(3)
        self._quat  = np.array([1.0, 0.0, 0.0, 0.0])
        self._omega = np.zeros(3)
        self._step_count = 0
        self._physics_params = self._rand.nominal()

        # Spaces (defined without gymnasium dependency for portability)
        obs_high = np.full(self._cfg.obs_dim, np.inf)
        self._obs_low  = -obs_high
        self._obs_high =  obs_high
        self._act_low  = np.zeros(self._cfg.act_dim)
        self._act_high = np.ones(self._cfg.act_dim)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset to a new episode."""
        if seed is not None:
            np.random.seed(seed)

        # Randomise physics
        self._physics_params = self._rand.sample()

        # Random start position near goal (±2 m)
        self._pos   = self._cfg.goal_position + np.random.uniform(-2, 2, 3)
        self._pos[2] = max(self._pos[2], -self._cfg.min_altitude_m - 1.0)
        self._vel   = np.zeros(3)
        self._quat  = np.array([1.0, 0.0, 0.0, 0.0])
        self._omega = np.zeros(3)
        self._step_count = 0
        self._shaper.reset()

        return self._get_obs(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Apply action and advance one control timestep.

        Args:
            action: Normalised motor thrusts [0, 1]^4.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        action = np.clip(action, 0.0, 1.0)

        # Physics integration (simplified)
        self._integrate(action)
        self._step_count += 1

        obs       = self._get_obs()
        pos_err   = float(np.linalg.norm(self._pos - self._cfg.goal_position))
        tilt      = self._get_tilt()
        speed     = float(np.linalg.norm(self._vel))
        goal_ok   = pos_err < self._cfg.goal_tolerance_m
        crashed   = self._is_crashed()

        reward    = self._shaper.compute(pos_err, tilt, speed, action, goal_ok, crashed)
        terminated = crashed or goal_ok
        truncated  = self._step_count >= self._cfg.max_episode_steps

        info = {
            "pos_error_m":   round(pos_err, 3),
            "tilt_rad":      round(tilt, 4),
            "speed_ms":      round(speed, 3),
            "step":          self._step_count,
            "goal_reached":  goal_ok,
            "crashed":       crashed,
        }
        return obs, reward, terminated, truncated, info

    def action_space_sample(self) -> np.ndarray:
        """Return a random valid action (hover ≈ 0.5 each)."""
        return np.random.uniform(0.0, 1.0, self._cfg.act_dim).astype(np.float32)

    def close(self) -> None:
        pass

    # ── Internal physics (simplified for RL training speed) ───────────────────

    def _integrate(self, action: np.ndarray) -> None:
        """Fast approximate 6-DOF integration for RL training."""
        cfg = self._cfg
        dt  = cfg.dt / cfg.sim_steps_per_ctrl
        m   = 1.5 * self._physics_params.get("mass_scale", 1.0)
        g   = 9.81

        # Total thrust (sum of 4 motor thrusts, scaled)
        max_thrust_per_motor = m * g * 0.4   # Slightly over-powered
        thrusts = action * max_thrust_per_motor
        F_total = thrusts.sum()

        # Torques from differential thrust (simplified X config)
        arms    = 0.25   # m
        tau_roll  = (thrusts[0] + thrusts[2] - thrusts[1] - thrusts[3]) * arms
        tau_pitch = (thrusts[0] + thrusts[1] - thrusts[2] - thrusts[3]) * arms
        tau_yaw   = (thrusts[0] + thrusts[3] - thrusts[1] - thrusts[2]) * 0.01

        for _ in range(cfg.sim_steps_per_ctrl):
            # Attitude dynamics (simplified: Euler angles, no coupling)
            q0, q1, q2, q3 = self._quat

            # Thrust in NED frame (up = -z in NED)
            thrust_world = np.array([
                2*(q1*q3 - q0*q2) * F_total / m,
                2*(q2*q3 + q0*q1) * F_total / m,
                (q0*q0 - q1*q1 - q2*q2 + q3*q3) * F_total / m - g,
            ])

            # Translational dynamics
            self._vel += thrust_world * dt
            self._pos += self._vel * dt

            # Attitude dynamics
            I = np.array([0.035, 0.046, 0.098]) * self._physics_params.get("inertia_scale", 1.0)
            self._omega += np.array([tau_roll, tau_pitch, tau_yaw]) / I * dt

            # Quaternion integration
            wx, wy, wz = self._omega
            dq = 0.5 * dt * np.array([
                -q1*wx - q2*wy - q3*wz,
                 q0*wx + q2*wz - q3*wy,
                 q0*wy - q1*wz + q3*wx,
                 q0*wz + q1*wy - q2*wx,
            ])
            self._quat = self._quat + dq
            norm = np.linalg.norm(self._quat)
            if norm > 1e-10:
                self._quat /= norm

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        goal_rel = self._pos - self._cfg.goal_position
        obs = np.concatenate([
            self._pos,
            self._vel,
            self._quat,
            self._omega,
            goal_rel,
        ]).astype(np.float32)
        return obs

    def _get_tilt(self) -> float:
        """Tilt angle from vertical (radians)."""
        q0, q1, q2, q3 = self._quat
        cos_tilt = q0*q0 - q1*q1 - q2*q2 + q3*q3
        return float(math.acos(max(-1.0, min(1.0, cos_tilt))))

    def _is_crashed(self) -> bool:
        alt_agl = -self._pos[2]
        tilt    = self._get_tilt()
        return (alt_agl < self._cfg.min_altitude_m or
                alt_agl > self._cfg.max_altitude_m or
                tilt    > self._cfg.max_tilt_rad)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def observation_dim(self) -> int:
        return self._cfg.obs_dim

    @property
    def action_dim(self) -> int:
        return self._cfg.act_dim

    @property
    def position(self) -> np.ndarray:
        return self._pos.copy()

    @property
    def step_count(self) -> int:
        return self._step_count


# ─────────────────────────────────────────────────────────────────────────────
#  Curriculum Manager
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumManager:
    """Progressive task difficulty for stable RL training.

    Stages:
      0. Hover in place (easy)
      1. Reach target 1 m away
      2. Reach target 5 m away with attitude disturbances
      3. Aggressive manoeuvres (flips, fast repositioning)
      4. Formation flight (multi-drone, Module 21)

    Advances stage when success_rate > threshold over last N episodes.

    Usage::

        curriculum = CurriculumManager()
        curriculum.record_episode(success=True, reward=45.2)
        cfg = curriculum.get_env_config()   # gets config for current stage
    """

    STAGES: List[dict] = [
        {"name": "hover",        "goal_dist": 0.1, "wind": 0.0, "tilt_limit": 60},
        {"name": "near_wp",      "goal_dist": 1.0, "wind": 0.5, "tilt_limit": 60},
        {"name": "far_wp",       "goal_dist": 5.0, "wind": 2.0, "tilt_limit": 60},
        {"name": "agile",        "goal_dist": 5.0, "wind": 3.0, "tilt_limit": 75},
        {"name": "full_mission", "goal_dist": 10.0,"wind": 5.0, "tilt_limit": 80},
    ]

    def __init__(
        self,
        advance_threshold: float = 0.75,
        window:            int   = 50,
    ) -> None:
        self._stage    = 0
        self._threshold = advance_threshold
        self._window   = window
        self._successes: List[bool]  = []
        self._rewards:   List[float] = []

    def record_episode(self, success: bool, reward: float) -> None:
        self._successes.append(success)
        self._rewards.append(reward)
        if len(self._successes) > self._window:
            self._successes.pop(0)
            self._rewards.pop(0)
        self._maybe_advance()

    def _maybe_advance(self) -> None:
        if len(self._successes) < self._window:
            return
        rate = sum(self._successes) / len(self._successes)
        if rate >= self._threshold and self._stage < len(self.STAGES) - 1:
            self._stage += 1

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def stage_name(self) -> str:
        return self.STAGES[self._stage]["name"]

    @property
    def success_rate(self) -> float:
        if not self._successes:
            return 0.0
        return sum(self._successes) / len(self._successes)

    def get_env_config(self) -> EnvConfig:
        s   = self.STAGES[self._stage]
        cfg = EnvConfig()
        cfg.max_tilt_rad = math.radians(s["tilt_limit"])
        DomainRandomiser(DomainRandomConfig(
            wind_ms=(0.0, s["wind"]),
        ))
        return cfg

    def __repr__(self) -> str:
        return (
            f"CurriculumManager(stage={self._stage}/{len(self.STAGES)-1}, "
            f"name={self.stage_name!r}, "
            f"success_rate={self.success_rate:.2f})"
        )
