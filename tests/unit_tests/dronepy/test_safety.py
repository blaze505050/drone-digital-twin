"""
Unit tests for SafetyGateway, execution modes, and LaptopController.
"""
import numpy as np
import pytest

import dronepy


def test_safety_gateway_default_simulation_mode():
    gw = dronepy.SafetyGateway()
    assert gw.mode == dronepy.SystemExecutionMode.SIMULATION_ONLY

    # Requesting hardware flight requires confirmation token
    denied = gw.request_hardware_enable("random_token")
    assert denied is False
    assert gw.mode == dronepy.SystemExecutionMode.SIMULATION_ONLY

    allowed = gw.request_hardware_enable("CONFIRM_HARDWARE_FLIGHT_SAFETY_CHECKED")
    assert allowed is True
    assert gw.mode == dronepy.SystemExecutionMode.HARDWARE_ACTIVE_PILOT


def test_safety_gateway_arming_interlock():
    gw = dronepy.SafetyGateway()
    pos = np.zeros(3)
    vel = np.zeros(3)
    eul = np.zeros(3)

    intent = dronepy.CommandIntent(timestamp=1.0, mode="POSITION", throttle_norm=0.5)
    report = gw.validate_command(intent, pos, vel, eul)
    assert report.passed is False
    assert report.violation == dronepy.SafetyViolationType.DISARMED

    # Arm vehicle
    gw.arm(1234)
    report_armed = gw.validate_command(intent, pos, vel, eul)
    assert report_armed.passed is True
    assert report_armed.violation == dronepy.SafetyViolationType.NONE


def test_safety_gateway_geofence_enforcement():
    gw = dronepy.SafetyGateway(
        geofence=dronepy.GeofenceCylinder(max_radius_m=100.0, max_altitude_m=50.0)
    )
    gw.arm(1234)

    # Position inside geofence
    pos_inside = np.array([20.0, 30.0, -20.0])  # NED down is positive, alt=20m
    intent = dronepy.CommandIntent(timestamp=1.0)
    rep_inside = gw.validate_command(intent, pos_inside, np.zeros(3), np.zeros(3))
    assert rep_inside.passed is True

    # Position breached outside
    pos_outside = np.array([120.0, 0.0, -20.0])
    rep_outside = gw.validate_command(intent, pos_outside, np.zeros(3), np.zeros(3))
    assert rep_outside.passed is False
    assert rep_outside.violation == dronepy.SafetyViolationType.GEOFENCE_EXCEEDED


def test_laptop_controller_inputs():
    controller = dronepy.LaptopController()
    controller.safety_gateway.arm(1234)

    # Simulate keyboard forward pitch
    controller.update_from_keyboard({"w": True})
    pos = np.zeros(3)
    vel = np.zeros(3)
    eul = np.zeros(3)

    report = controller.process_and_validate(pos, vel, eul)
    assert report.passed is True
    assert report.filtered_intent is not None
    assert report.filtered_intent.target_velocity_ned[0] > 0.0  # North / Forward velocity
