"""
dronepy.failures
================
Standardized Failure and Scenario Injection Architecture for DronePy.
Simulates motor failure, propeller damage, sensor loss, and wind disturbances.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .environment import Wind
from .events import FlightEvent


class ScenarioBuilder:
    """Builder for scheduling failure and environmental events at specific timestamps."""

    def __init__(self, scenario: "Scenario", time_s: float) -> None:
        self.scenario = scenario
        self.time = float(time_s)

    def motor_failure(self, motor_index: int = 0) -> "Scenario":
        """Schedule complete motor failure at this timestamp."""
        def _action(sim: Any) -> None:
            sim.drone.motors.failure(motor_index)
            sim.log_event(f"FAILURE: Motor {motor_index + 1} failed completely.")
        self.scenario.add_event(FlightEvent(
            time=self.time,
            action=_action,
            name=f"MotorFailure_{motor_index}",
            description=f"Motor {motor_index + 1} shut down",
        ))
        return self.scenario

    def motor_efficiency(self, motor_index: int, efficiency: float) -> "Scenario":
        """Schedule motor degradation/efficiency drop at this timestamp."""
        def _action(sim: Any) -> None:
            sim.drone.motors[motor_index].set_efficiency(efficiency)
            sim.log_event(f"DEGRADATION: Motor {motor_index + 1} efficiency dropped to {efficiency*100:.1f}%.")
        self.scenario.add_event(FlightEvent(
            time=self.time,
            action=_action,
            name=f"MotorEfficiency_{motor_index}",
            description=f"Motor {motor_index + 1} efficiency = {efficiency}",
        ))
        return self.scenario

    def wind_change(
        self,
        wind: Optional[Union[Wind, np.ndarray, List[float]]] = None,
        speed_mps: Optional[float] = None,
        direction_deg: float = 0.0,
        north: Optional[float] = None,
        east: Optional[float] = None,
        down: Optional[float] = None,
    ) -> "Scenario":
        """Schedule an environmental wind vector or turbulence update."""
        if wind is not None:
            w_obj = wind if isinstance(wind, Wind) else Wind(steady_wind_ned=np.asarray(wind, dtype=np.float64))
        elif north is not None or east is not None or down is not None:
            w_obj = Wind.constant(north=north, east=east, down=down)
        elif speed_mps is not None:
            w_obj = Wind.constant(speed=speed_mps, heading_deg=direction_deg)
        else:
            w_obj = Wind()

        def _action(sim: Any) -> None:
            sim.environment.wind = w_obj
            sim.log_event(f"ENVIRONMENT: Wind shifted to {w_obj.steady_wind_ned} m/s.")
        self.scenario.add_event(FlightEvent(
            time=self.time,
            action=_action,
            name="WindShift",
            description="Atmospheric wind vector updated",
        ))
        return self.scenario

    def gps_loss(self) -> "Scenario":
        """Schedule GPS sensor fix loss at this timestamp."""
        def _action(sim: Any) -> None:
            sim.gps_active = False
            sim.log_event("SENSOR: GPS signal lost (0 satellites).")
        self.scenario.add_event(FlightEvent(
            time=self.time,
            action=_action,
            name="GPSLoss",
            description="GPS loss of signal",
        ))
        return self.scenario

    def payload_release(self, mass_kg: Optional[float] = None) -> "Scenario":
        """Schedule payload drop at this timestamp."""
        def _action(sim: Any) -> None:
            sim.drone.release_payload(mass_kg)
            sim.log_event(f"EVENT: Payload dropped ({mass_kg if mass_kg is not None else 'all'} kg).")
        self.scenario.add_event(FlightEvent(
            time=self.time,
            action=_action,
            name="PayloadDrop",
            description="Payload release",
        ))
        return self.scenario


class Scenario:
    """Timeline scenario manager for scheduling multiple failure and operational events.

    Example
    -------
    >>> scenario = Scenario()
    >>> scenario.at(15.0).motor_failure(1)
    >>> scenario.at(25.0).wind_change([6.0, 3.0, 0.0])
    >>> flight = drone.simulate(duration=40.0, events=scenario)
    """

    def __init__(self) -> None:
        self.events: List[FlightEvent] = []

    def at(self, time_s: float) -> ScenarioBuilder:
        """Specify time in seconds for the next scheduled event."""
        return ScenarioBuilder(self, time_s)

    def add_event(self, event: FlightEvent) -> None:
        self.events.append(event)
        self.events.sort(key=lambda e: e.time)

    def to_events(self) -> List[FlightEvent]:
        return list(self.events)

    def __iter__(self):
        return iter(self.events)
