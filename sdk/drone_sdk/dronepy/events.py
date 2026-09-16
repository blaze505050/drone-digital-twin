"""
dronepy.events
==============
Discrete Event Simulation Architecture for DronePy.
Supports payload release, mass & CG shifts, discrete wind changes, and custom event callbacks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class FlightEvent:
    """Discrete event triggered during a simulation at a specified time."""
    time: float
    action: Callable[..., None]
    name: str = "FlightEvent"
    description: str = ""
    executed: bool = False

    def trigger(self, sim: Any) -> None:
        """Execute the event action against the simulation instance."""
        if not self.executed:
            self.action(sim)
            self.executed = True


def payload_release(mass_kg: Optional[float] = None) -> Callable[[Any], None]:
    """Action callback to release payload mass from the drone."""
    def _action(sim: Any) -> None:
        sim.drone.release_payload(mass_released=mass_kg)
        sim.log_event(f"Payload released ({mass_kg if mass_kg is not None else 'all'} kg)")
    return _action


def payload_add(mass_kg: float, offset_body: Optional[np.ndarray] = None) -> Callable[[Any], None]:
    """Action callback to add payload mass to the drone."""
    def _action(sim: Any) -> None:
        sim.drone.add_payload(mass_added=mass_kg, location_body=offset_body)
        sim.log_event(f"Payload added ({mass_kg} kg)")
    return _action


def set_target(target_pos_ned: np.ndarray, target_yaw_rad: float = 0.0) -> Callable[[Any], None]:
    """Action callback to switch the flight controller waypoint target."""
    def _action(sim: Any) -> None:
        sim.target_pos = np.asarray(target_pos_ned, dtype=np.float64)
        sim.target_yaw = float(target_yaw_rad)
        sim.log_event(f"Target waypoint updated to NED {sim.target_pos}")
    return _action
