"""
dronepy.sweep
=============
Parametric Study and Sensitivity Sweep Engine for DronePy.
Sweeps multirotor design variables (mass, propeller diameter, battery, arm length)
and evaluates engineering trade-offs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .drone import Drone
from .environment import Environment
from .flight import FlightSimulatorEngine
from .results import FlightResult


@dataclass
class SweepResult:
    """Outcomes of a multi-variable parametric design sweep."""
    parameter_name: str
    values: List[Any]
    flight_results: List[FlightResult]

    @property
    def energies_wh(self) -> List[float]:
        return [float(res.battery_energy_wh[-1]) for res in self.flight_results]

    @property
    def max_velocities_mps(self) -> List[float]:
        return [float(np.max(res.airspeed)) for res in self.flight_results]

    @property
    def max_altitudes_m(self) -> List[float]:
        return [float(np.max(res.altitude_agl)) for res in self.flight_results]

    @property
    def results(self) -> List[FlightResult]:
        """Alias for flight_results."""
        return self.flight_results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter": self.parameter_name,
            "values": self.values,
            "energy_consumed_wh": self.energies_wh,
            "max_velocity_mps": self.max_velocities_mps,
        }

    def plot(self, metric: str = "energy", show: bool = True) -> Any:
        """Plot the swept trade-off curve."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for SweepResult.plot().")

        fig, ax = plt.subplots(figsize=(8, 5))

        if metric == "energy":
            y_vals = self.energies_wh
            y_label = "Energy Consumed (Wh)"
            color = "#00d4ff"
        elif metric == "velocity":
            y_vals = self.max_velocities_mps
            y_label = "Max Velocity (m/s)"
            color = "#ff8a3d"
        else:
            y_vals = self.max_altitudes_m
            y_label = "Max Altitude (m)"
            color = "#00ff88"

        ax.plot(self.values, y_vals, marker="o", color=color, linewidth=2.0, markersize=7)
        ax.set_xlabel(f"Swept Parameter: {self.parameter_name}")
        ax.set_ylabel(y_label)
        ax.set_title(f"Parametric Trade Study: {metric.title()} vs {self.parameter_name}")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        if show:
            plt.show()
        return fig


class Sweep:
    """Executes parametric trade studies across drone configuration parameters."""

    def __init__(
        self,
        parameter: Optional[str] = None,
        values: Optional[List[Any]] = None,
        drone_factory: Optional[Callable[[Any], Drone]] = None,
        base_drone: Optional[Drone] = None,
        drone: Optional[Drone] = None,
        parameter_name: Optional[str] = None,
        environment: Optional[Environment] = None,
        duration: float = 30.0,
        dt: float = 0.005,
    ) -> None:
        self.parameter = str(parameter_name if parameter_name is not None else parameter)
        self.values = list(values if values is not None else [])
        self.drone_factory = drone_factory
        self.base_drone = drone if drone is not None else base_drone
        self.environment = environment or Environment.standard_atmosphere()
        self.duration = duration
        self.dt = dt

    def run(self, duration: Optional[float] = None, dt: Optional[float] = None) -> SweepResult:
        """Execute simulation for each parameter value."""
        run_dur = duration if duration is not None else self.duration
        run_dt = dt if dt is not None else self.dt
        results: List[FlightResult] = []

        for val in self.values:
            if self.drone_factory is not None:
                drone_instance = self.drone_factory(val)
            elif self.base_drone is not None:
                drone_instance = copy.deepcopy(self.base_drone)
                if hasattr(drone_instance, self.parameter):
                    setattr(drone_instance, self.parameter, val)
                elif self.parameter == "mass":
                    drone_instance.base_mass = float(val)
                    drone_instance.arm_length = float(val)
                elif self.parameter in ("cd", "drag"):
                    drone_instance.aerodynamics.cd_x = float(val)
                    drone_instance.aerodynamics.cd_y = float(val)
            else:
                raise ValueError("Must supply either drone_factory or base_drone to Sweep.")

            sim = FlightSimulatorEngine(drone=drone_instance, environment=self.environment)
            res = sim.run(duration=self.duration, dt=self.dt)
            results.append(res)

        return SweepResult(
            parameter_name=self.parameter,
            values=self.values,
            flight_results=results,
        )
