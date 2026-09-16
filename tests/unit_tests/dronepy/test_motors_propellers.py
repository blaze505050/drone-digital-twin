"""
Unit tests for Motor dynamics, failures, and Propeller models.
"""
import numpy as np
import pytest

import dronepy


def test_motor_first_order_lag():
    motor = dronepy.Motor(max_rpm=10000.0, time_constant=0.04)
    assert motor.current_rpm == 0.0

    # Step response towards 5000 RPM over dt=0.02s
    thrust, torque, curr, pwr = motor.step(cmd_rpm=5000.0, dt=0.02, bus_voltage=16.0)
    assert motor.current_rpm > 0.0
    assert motor.current_rpm < 5000.0
    assert thrust > 0.0
    assert pwr > 0.0


def test_motor_failure_injection():
    motor = dronepy.Motor(max_rpm=10000.0)
    motor.step(cmd_rpm=8000.0, dt=0.1)
    assert motor.current_rpm > 1000.0

    # Inject failure
    motor.fail(rpm_limit=0.0)
    assert motor.is_failed is True
    thrust, torque, curr, pwr = motor.step(cmd_rpm=8000.0, dt=0.05)
    assert thrust == 0.0


def test_propeller_models():
    prop = dronepy.Propeller(diameter_in=10.0, pitch_in=4.5)
    thrust, torque, pwr = prop.compute_thrust_and_torque(rpm=6000.0)
    assert thrust > 0.0
    assert torque > 0.0
    assert pwr > 0.0

    bemt_prop = dronepy.BEMTPropeller(diameter_in=10.0, pitch_in=4.5)
    b_thrust, b_torque, b_pwr = bemt_prop.compute_thrust_and_torque(rpm=6000.0)
    assert b_thrust > 0.0
    assert b_torque > 0.0
