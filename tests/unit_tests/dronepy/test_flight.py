"""
Unit tests for 6-DOF Flight simulation, Mission planning, and Events.
"""
import numpy as np
import pytest

import dronepy


def test_flight_hover_simulation():
    drone = dronepy.Drone.quadcopter(mass=1.5)
    env = dronepy.Environment.standard_atmosphere()
    flight = dronepy.Flight(drone=drone, environment=env, duration=2.0)
    res = flight.result

    assert isinstance(res, dronepy.FlightResult)
    assert len(res.time) > 10
    assert res.pos_ned.shape == (len(res.time), 3)
    assert res.vel_ned.shape == (len(res.time), 3)
    assert res.total_thrust.shape == (len(res.time),)


def test_mission_execution():
    drone = dronepy.Drone.quadcopter(mass=1.5)
    mission = dronepy.Mission()
    mission.takeoff(altitude=5.0)
    mission.goto(x=10.0, y=0.0, z=-5.0)
    mission.land()

    flight = dronepy.Flight(drone=drone, mission=mission, duration=3.0)
    assert flight.result is not None
    assert len(flight.events_log) > 0


def test_scenario_event_injection():
    drone = dronepy.Drone.quadcopter(mass=1.5)
    scenario = dronepy.Scenario()
    scenario.at(1.0).motor_failure(motor_index=0)
    scenario.at(1.5).wind_change(speed_mps=8.0, direction_deg=90.0)

    flight = dronepy.Flight(drone=drone, events=scenario, duration=2.5)
    assert flight.result is not None
    # Motor 1 should be failed after t=1.0
    assert drone.motors[0].is_failed is True
