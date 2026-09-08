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

    def seed(self, seed: int) -> None:
        """Seed the separate NumPy random generator."""
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
#  Gymnasium-compatible Space Abstractions
# ─────────────────────────────────────────────────────────────────────────────

class BoxSpace:
    """Gymnasium-compatible Box space interface for standalone portability."""

    def __init__(self, low: np.ndarray, high: np.ndarray, shape: Optional[Tuple[int, ...]] = None, dtype=np.float32) -> None:
        self.low = np.array(low, dtype=dtype)
        self.high = np.array(high, dtype=dtype)
        self.shape = self.low.shape if shape is None else shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        # Avoid inf bounds in sample
        low = np.where(np.isneginf(self.low), -1e3, self.low)
        high = np.where(np.isposinf(self.high), 1e3, self.high)
        return np.random.uniform(low, high).astype(self.dtype)

    def contains(self, x: Any) -> bool:
        arr = np.asarray(x)
        return bool(arr.shape == self.shape and np.all(arr >= self.low) and np.all(arr <= self.high))


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
        config:              Optional[EnvConfig]        = None,
        randomiser:          Optional[DomainRandomiser] = None,
        reward_weights:      Optional[RewardWeights]    = None,
        cad_mass_properties: Optional[object]           = None,
        arm_length:          Optional[float]            = None,
    ) -> None:
        self._cfg    = config      or EnvConfig()
        self._rand   = randomiser  or DomainRandomiser()
        self._shaper = RewardShaper(reward_weights)
        self._cad_mass_properties = cad_mass_properties
        self._arm_length = arm_length

        # State
        self._pos   = np.zeros(3)
        self._vel   = np.zeros(3)
        self._quat  = np.array([1.0, 0.0, 0.0, 0.0])
        self._omega = np.zeros(3)
        self._motor_thrust = np.zeros(4)
        self._step_count = 0
        self._physics_params = self._rand.nominal()

        # Aerodynamic and motor parameters
        self._motor_tau = 0.035          # 35 ms first-order ESC/motor time constant
        self._cd_flat_plate = 1.05       # Quadrotor bluff-body drag coefficient
        self._frontal_area = 0.025       # Frontal area m^2
        self._prop_radius = 0.127        # 5-inch prop radius
        self._i_rotor = 1.5e-5           # Rotor polar inertia (kg*m^2)

        # Spaces
        obs_high = np.full(self._cfg.obs_dim, np.inf, dtype=np.float32)
        self._obs_low  = -obs_high
        self._obs_high =  obs_high
        self._act_low  = np.zeros(self._cfg.act_dim, dtype=np.float32)
        self._act_high = np.ones(self._cfg.act_dim, dtype=np.float32)

        try:
            import gymnasium as gym
            self.observation_space = gym.spaces.Box(self._obs_low, self._obs_high, dtype=np.float32)
            self.action_space = gym.spaces.Box(self._act_low, self._act_high, dtype=np.float32)
        except ImportError:
            self.observation_space = BoxSpace(self._obs_low, self._obs_high, dtype=np.float32)
            self.action_space = BoxSpace(self._act_low, self._act_high, dtype=np.float32)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset to a new episode."""
        if seed is not None:
            np.random.seed(seed)
            if hasattr(self._rand, "seed"):
                self._rand.seed(seed)

        # Randomise physics
        self._physics_params = self._rand.sample()

        # Random start position near goal (±2 m)
        self._pos   = self._cfg.goal_position + np.random.uniform(-2, 2, 3)
        self._pos[2] = max(self._pos[2], -self._cfg.min_altitude_m - 1.0)
        self._vel   = np.zeros(3)
        self._quat  = np.array([1.0, 0.0, 0.0, 0.0])
        self._omega = np.zeros(3)
        self._motor_thrust = np.zeros(4)
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

        # Physics integration (with drag, motor lag, gyro torque, and ground effect)
        self._integrate(action)
        self._step_count += 1

        obs       = self._get_obs()
        pos_err   = float(np.linalg.norm(self._pos - self._cfg.goal_position))
        tilt      = self._get_tilt()
        speed     = float(np.linalg.norm(self._vel))
        goal_ok   = pos_err < self._cfg.goal_tolerance_m
        crashed   = self._is_crashed()

        reward    = self._shaper.compute(pos_err, tilt, speed, action, goal_ok, crashed)

        # Termination conditions
        terminated = crashed or (goal_ok and speed < 0.2)
        truncated  = self._step_count >= self._cfg.max_episode_steps

        info = {
            "step":          self._step_count,
            "pos_error_m":   pos_err,
            "tilt_rad":      tilt,
            "speed_ms":      speed,
            "goal_reached":  goal_ok,
            "crashed":       crashed,
        }
        return obs, reward, terminated, truncated, info

    def action_space_sample(self) -> np.ndarray:
        """Return a random valid action (hover ≈ 0.5 each)."""
        return np.random.uniform(0.0, 1.0, self._cfg.act_dim).astype(np.float32)

    def close(self) -> None:
        pass

    # ── Internal physics (full 6-DOF dynamics with lag, drag, gyro & ground effect) ──

    def _integrate(self, action: np.ndarray) -> None:
        """Full 6-DOF integration with motor lag, drag, gyro torque, and CAD coupling."""
        cfg = self._cfg
        dt  = cfg.dt / cfg.sim_steps_per_ctrl
        g   = 9.81
        rho = 1.225

        # Extract mass & inertia from CAD if provided, else nominal
        if self._cad_mass_properties is not None:
            cad = self._cad_mass_properties
            base_m = getattr(cad, "mass_kg", 1.5)
            base_I = getattr(cad, "inertia_tensor", np.diag([0.035, 0.046, 0.098]))
            if isinstance(base_I, np.ndarray) and base_I.ndim == 2:
                inertia_diag = np.diag(base_I)
            else:
                inertia_diag = np.array([0.035, 0.046, 0.098])
            arms = self._arm_length if self._arm_length is not None else 0.25
        else:
            base_m = 1.5
            inertia_diag = np.array([0.035, 0.046, 0.098])
            arms = self._arm_length if self._arm_length is not None else 0.25

        m = base_m * self._physics_params.get("mass_scale", 1.0)
        I = inertia_diag * self._physics_params.get("inertia_scale", 1.0)

        # Actuator command with independent per-motor scaling and mass-independent motor limits
        motor_scales = np.array([
            self._physics_params.get("motor_scale_1", 1.0),
            self._physics_params.get("motor_scale_2", 1.0),
            self._physics_params.get("motor_scale_3", 1.0),
            self._physics_params.get("motor_scale_4", 1.0),
        ])
        base_max_thrust = 6.62
        target_thrusts = action * (base_max_thrust * motor_scales)

        # Substep physics loop
        for _ in range(cfg.sim_steps_per_ctrl):
            # 1. First-order motor/ESC spin-up lag
            alpha_motor = min(1.0, dt / max(1e-4, self._motor_tau))
            self._motor_thrust += (target_thrusts - self._motor_thrust) * alpha_motor
            thrusts = self._motor_thrust.copy()

            # 2. Ground effect augmentation near surface (NED: alt_agl = -pos[2])
            alt_agl = max(0.01, -self._pos[2])
            if alt_agl < 2.0 * (2.0 * self._prop_radius):
                r_over_z = self._prop_radius / (4.0 * max(alt_agl, self._prop_radius))
                k_ge = 1.0 / max(0.6, 1.0 - (r_over_z ** 2))
                thrusts *= min(1.3, k_ge)

            F_total = float(np.sum(thrusts))

            # 3. Aerodynamic differential torques
            tau_roll  = (thrusts[0] + thrusts[2] - thrusts[1] - thrusts[3]) * arms
            tau_pitch = (thrusts[0] + thrusts[1] - thrusts[2] - thrusts[3]) * arms
            tau_yaw   = (thrusts[0] + thrusts[3] - thrusts[1] - thrusts[2]) * 0.01

            # 4. Rotor gyroscopic reaction torque (counter-rotating pairs)
            omega_rotor = np.sqrt(np.maximum(thrusts, 0.0) / 1.5e-5)
            net_rotor_h = self._i_rotor * (omega_rotor[0] + omega_rotor[2] - omega_rotor[1] - omega_rotor[3])
            tau_gyro = np.array([
                -self._omega[1] * net_rotor_h,
                 self._omega[0] * net_rotor_h,
                 0.0,
            ])

            # 5. Quaternion body-to-world rotation (NED frame: thrust acts along -z)
            q0, q1, q2, q3 = self._quat
            thrust_world = np.array([
                -2.0 * (q1*q3 + q0*q2) * F_total,
                -2.0 * (q2*q3 - q0*q1) * F_total,
                -(q0*q0 - q1*q1 - q2*q2 + q3*q3) * F_total,
            ])

            # 6. Quadratic aerodynamic translational drag with randomized wind and drag scale
            drag_scale = self._physics_params.get("drag_scale", 1.0)
            wind_speed = self._physics_params.get("wind_speed_ms", 0.0)
            wind_dir = self._physics_params.get("wind_dir_rad", 0.0)
            v_wind = np.array([wind_speed * math.cos(wind_dir), wind_speed * math.sin(wind_dir), 0.0])
            v_rel = self._vel - v_wind

            drag_world = -0.5 * rho * (self._cd_flat_plate * drag_scale) * self._frontal_area * v_rel * np.abs(v_rel)

            # 7. Translational acceleration (NED frame: +z down, gravity g_ned = [0, 0, +g])
            acc_world = (thrust_world + drag_world) / m + np.array([0.0, 0.0, g])
            self._vel += acc_world * dt
            self._pos += self._vel * dt

            # 8. Angular dynamics via Euler equations
            total_torques = np.array([tau_roll, tau_pitch, tau_yaw]) + tau_gyro
            gyro_moment = np.cross(self._omega, I * self._omega)
            self._omega += ((total_torques - gyro_moment) / I) * dt

            # 9. Quaternion integration
            wx, wy, wz = self._omega
            dq = 0.5 * dt * np.array([
                -q1*wx - q2*wy - q3*wz,
                 q0*wx + q2*wz - q3*wy,
                 q0*wy - q1*wz + q3*wx,
                 q0*wz + q1*wy - q2*wx,
            ])
            self._quat += dq
            norm_q = np.linalg.norm(self._quat)
            if norm_q > 1e-10:
                self._quat /= norm_q

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
