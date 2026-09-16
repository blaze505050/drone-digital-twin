"""
dronepy.monte_carlo
===================
Stochastic Monte Carlo Simulation Engine for DronePy.
Provides parameter uncertainty distributions, parallel execution, and statistical envelopes.
"""
from __future__ import annotations

import copy
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .drone import Drone
from .environment import Environment, Wind
from .flight import FlightSimulatorEngine
from .results import FlightResult


@dataclass
class Distribution:
    """Statistical distribution specification for Monte Carlo sampling."""
    dist_type: str  # "normal", "uniform", "triangular", "discrete"
    p1: float       # mean / lower bound
    p2: float       # std / upper bound
    p3: float = 0.0 # mode (for triangular)

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one random sample from this distribution."""
        if self.dist_type == "normal":
            return float(rng.normal(self.p1, self.p2))
        elif self.dist_type == "uniform":
            return float(rng.uniform(self.p1, self.p2))
        elif self.dist_type == "triangular":
            return float(rng.triangular(self.p1, self.p3, self.p2))
        return self.p1

    @classmethod
    def normal(cls, mean: float, std_dev: float) -> "Distribution":
        return cls("normal", mean, std_dev)

    @classmethod
    def uniform(cls, low: float, high: float) -> "Distribution":
        return cls("uniform", low, high)

    @classmethod
    def triangular(cls, low: float, mode: float, high: float) -> "Distribution":
        return cls("triangular", low, high, mode)


@dataclass
class MonteCarloResult:
    """Aggregated statistical outcome across multiple stochastic flight runs."""
    runs: int
    landing_positions_ned: np.ndarray      # (N, 3)
    max_altitudes_m: np.ndarray            # (N,)
    max_velocities_mps: np.ndarray         # (N,)
    total_energies_wh: np.ndarray          # (N,)
    flight_durations_s: np.ndarray         # (N,)
    parameters_sampled: List[Dict[str, float]]
    trajectories_sample: List[np.ndarray]   # Sample of trajectory NED points for envelope plotting

    @property
    def landing_positions(self) -> np.ndarray:
        return self.landing_positions_ned

    @property
    def max_altitudes(self) -> np.ndarray:
        return self.max_altitudes_m

    @property
    def max_velocities(self) -> np.ndarray:
        return self.max_velocities_mps

    @property
    def total_energies(self) -> np.ndarray:
        return self.total_energies_wh

    @property
    def flight_durations(self) -> np.ndarray:
        return self.flight_durations_s

    @property
    def landing_dispersion_radius_95(self) -> float:
        """95% Circular Error Probable (CEP95) radius in metres from nominal landing."""
        radii = np.linalg.norm(self.landing_positions_ned[:, :2], axis=1)
        return float(np.percentile(radii, 95.0))

    def summary(self) -> Dict[str, Any]:
        """Summary metrics and confidence intervals."""
        r_2d = np.linalg.norm(self.landing_positions_ned[:, :2], axis=1)
        return {
            "total_runs": self.runs,
            "cep50_radius_m": float(np.percentile(r_2d, 50.0)),
            "cep95_radius_m": float(np.percentile(r_2d, 95.0)),
            "cep50_m": float(np.percentile(r_2d, 50.0)),
            "cep95_m": float(np.percentile(r_2d, 95.0)),
            "mean_energy_wh": float(np.mean(self.total_energies_wh)),
            "std_energy_wh": float(np.std(self.total_energies_wh)),
            "energy_ci95": [float(np.percentile(self.total_energies_wh, 2.5)), float(np.percentile(self.total_energies_wh, 97.5))],
            "mean_max_vel_mps": float(np.mean(self.max_velocities_mps)),
            "max_vel_ci95": [float(np.percentile(self.max_velocities_mps, 2.5)), float(np.percentile(self.max_velocities_mps, 97.5))],
        }

    def plot_dispersion(self, **kwargs: Any) -> Any:
        from .dispersion import plot_landing_dispersion
        return plot_landing_dispersion(self, **kwargs)

    def plot_envelopes(self, **kwargs: Any) -> Any:
        from .dispersion import plot_trajectory_envelopes
        return plot_trajectory_envelopes(self, **kwargs)


def _single_run_worker(args: Tuple[Any, Any, Any, Any, float, float, Dict[str, float]]) -> Tuple[np.ndarray, float, float, float, float, np.ndarray]:
    drone_factory, env_factory, ctrl_factory, mission, duration, dt, param_sample = args

    # Instantiate vehicle
    drone = drone_factory()
    env = env_factory()
    ctrl = ctrl_factory(drone.mass) if ctrl_factory else None

    # Apply sampled uncertainties
    if "mass" in param_sample:
        drone.base_mass = param_sample["mass"]
    if "cd" in param_sample and hasattr(drone.aerodynamics, "cd_x"):
        drone.aerodynamics.cd_x = param_sample["cd"]
        drone.aerodynamics.cd_y = param_sample["cd"]
    if "motor_thrust" in param_sample:
        for m in drone.motors:
            m.thrust_coefficient = param_sample["motor_thrust"]
    if "battery_voltage" in param_sample and hasattr(drone.battery, "state"):
        drone.battery.state.v_terminal = param_sample["battery_voltage"]
    if "wind_speed" in param_sample:
        env.wind.steady_wind_ned = np.array([param_sample["wind_speed"], 0.0, 0.0])

    sim = FlightSimulatorEngine(drone=drone, environment=env, controller=ctrl, mission=mission)
    res = sim.run(duration=duration, dt=dt)

    landing_pos = res.pos_ned[-1]
    max_alt = float(np.max(res.altitude_agl))
    max_vel = float(np.max(res.airspeed))
    energy_wh = float(res.battery_energy_wh[-1])
    dur = float(res.time[-1])
    # Downsample trajectory for memory efficiency
    traj_step = max(1, len(res.pos_ned) // 50)
    traj_sample = res.pos_ned[::traj_step]

    return landing_pos, max_alt, max_vel, energy_wh, dur, traj_sample


class MonteCarlo:
    """Stochastic Monte Carlo simulation engine across parameter uncertainty distributions."""

    def __init__(
        self,
        drone: Drone,
        environment: Optional[Environment] = None,
        controller: Optional[Any] = None,
        mission: Optional[Any] = None,
        uncertainties: Optional[Dict[str, Distribution]] = None,
        runs: int = 100,
        num_simulations: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> None:
        self.drone = drone
        self.environment = environment or Environment.standard_atmosphere()
        self.controller = controller
        self.mission = mission
        self.uncertainties = dict(uncertainties) if uncertainties else {}
        self.runs = int(num_simulations if num_simulations is not None else runs)
        self.seed = seed

    def add_parameter(self, name: str, distribution: Distribution) -> None:
        """Add or update an uncertain parameter distribution."""
        self.uncertainties[name] = distribution

    def run(
        self,
        duration: float = 30.0,
        dt: float = 0.005,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> MonteCarloResult:
        """Execute all Monte Carlo iterations and aggregate results."""
        rng = np.random.default_rng(self.seed)

        # Draw all parameter samples up front
        param_samples: List[Dict[str, float]] = []
        for _ in range(self.runs):
            sample = {k: dist.sample(rng) for k, dist in self.uncertainties.items()}
            param_samples.append(sample)

        landings = np.zeros((self.runs, 3), dtype=np.float64)
        max_alts = np.zeros(self.runs, dtype=np.float64)
        max_vels = np.zeros(self.runs, dtype=np.float64)
        energies = np.zeros(self.runs, dtype=np.float64)
        durations = np.zeros(self.runs, dtype=np.float64)
        trajectories: List[np.ndarray] = []

        if parallel and self.runs > 10:
            # Parallel execution
            tasks = [
                (
                    lambda: copy.deepcopy(self.drone),
                    lambda: copy.deepcopy(self.environment),
                    (lambda m: copy.deepcopy(self.controller)) if self.controller else None,
                    copy.deepcopy(self.mission),
                    duration,
                    dt,
                    param_samples[i],
                )
                for i in range(self.runs)
            ]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_single_run_worker, task) for task in tasks]
                for idx, fut in enumerate(as_completed(futures)):
                    l_pos, m_alt, m_vel, e_wh, dur, traj = fut.result()
                    landings[idx] = l_pos
                    max_alts[idx] = m_alt
                    max_vels[idx] = m_vel
                    energies[idx] = e_wh
                    durations[idx] = dur
                    if len(trajectories) < 30:
                        trajectories.append(traj)
        else:
            # Sequential execution
            for idx in range(self.runs):
                d_copy = copy.deepcopy(self.drone)
                e_copy = copy.deepcopy(self.environment)
                c_copy = copy.deepcopy(self.controller) if self.controller else None
                m_copy = copy.deepcopy(self.mission) if self.mission else None

                p_sample = param_samples[idx]
                if "mass" in p_sample:
                    d_copy.base_mass = p_sample["mass"]
                if "cd" in p_sample and hasattr(d_copy.aerodynamics, "cd_x"):
                    d_copy.aerodynamics.cd_x = p_sample["cd"]
                    d_copy.aerodynamics.cd_y = p_sample["cd"]
                if "motor_thrust" in p_sample:
                    for m in d_copy.motors:
                        m.thrust_coefficient = p_sample["motor_thrust"]
                if "wind_speed" in p_sample:
                    e_copy.wind.steady_wind_ned = np.array([p_sample["wind_speed"], 0.0, 0.0])

                sim = FlightSimulatorEngine(drone=d_copy, environment=e_copy, controller=c_copy, mission=m_copy)
                res = sim.run(duration=duration, dt=dt)

                landings[idx] = res.pos_ned[-1]
                max_alts[idx] = float(np.max(res.altitude_agl))
                max_vels[idx] = float(np.max(res.airspeed))
                energies[idx] = float(res.battery_energy_wh[-1])
                durations[idx] = float(res.time[-1])

                if len(trajectories) < 30:
                    traj_step = max(1, len(res.pos_ned) // 50)
                    trajectories.append(res.pos_ned[::traj_step])

        return MonteCarloResult(
            runs=self.runs,
            landing_positions_ned=landings,
            max_altitudes_m=max_alts,
            max_velocities_mps=max_vels,
            total_energies_wh=energies,
            flight_durations_s=durations,
            parameters_sampled=param_samples,
            trajectories_sample=trajectories,
        )
