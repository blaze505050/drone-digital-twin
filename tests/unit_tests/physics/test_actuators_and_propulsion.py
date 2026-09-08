"""
Unit tests for Installed Propulsion & Actuator Dynamics (B11).
Verifies:
- UIUC wind tunnel propeller polar interpolation (CT, CP, efficiency).
- Brushless DC motor back-EMF, winding resistance, and ESC first-order lag response.
- Multirotor propulsion net force and moment balance in FRD body frame.
"""
import numpy as np
import pytest

from drone_sdk.configuration.schema import MotorConfig, PropellerConfig, RotorPlacement
from drone_sdk.physics.actuators import (
    MotorPropellerUnit,
    PropellerAerodynamics,
    PropulsionSystem,
)


def test_uiuc_propeller_interpolation():
    prop_cfg = PropellerConfig(
        name="APC 10x4.5 MR",
        diameter_m=0.254,
        pitch_m=0.1143,
        polar_dataset="uiuc_apc_10x4.5",
    )
    aero = PropellerAerodynamics(prop_cfg)

    # At static condition J = 0
    ct0, cp0, eta0 = aero.get_coefficients(0.0)
    assert np.isclose(ct0, 0.112, atol=1e-3)
    assert np.isclose(cp0, 0.048, atol=1e-3)
    assert eta0 == 0.0  # Efficiency is zero at zero forward speed

    # At J = 0.30 (moderate forward flight)
    ct_j, cp_j, eta_j = aero.get_coefficients(0.30)
    assert 0.07 < ct_j < 0.10
    assert 0.03 < cp_j < 0.05
    assert eta_j > 0.40  # Positive efficiency in forward flight

    # Thrust and Power scaling with RPM: T ~ RPM^2, P ~ RPM^3
    t1, q1, p1 = aero.compute_aerodynamics(rpm=3000.0)
    t2, q2, p2 = aero.compute_aerodynamics(rpm=6000.0)
    assert np.isclose(t2 / t1, 4.0, rtol=0.05)  # 2^2 = 4
    assert np.isclose(p2 / p1, 8.0, rtol=0.05)  # 2^3 = 8


def test_bldc_motor_and_esc_lag():
    m_cfg = MotorConfig(kv=880.0, r_m=0.12, i_0=0.6, tau_m=0.030)
    p_cfg = PropellerConfig(diameter_m=0.254, pitch_m=0.1143)
    unit = MotorPropellerUnit(m_cfg, p_cfg)

    # Initial step from 0 to 50% throttle at 16.0V
    t_n, q_nm, i_a, p_w = unit.step(throttle_cmd=0.50, bus_voltage_v=16.0, dt_sec=0.01)
    rpm_step1 = unit.current_rpm
    assert rpm_step1 > 0.0

    # Advance several time steps to reach steady state
    for _ in range(50):
        t_n, q_nm, i_a, p_w = unit.step(throttle_cmd=0.50, bus_voltage_v=16.0, dt_sec=0.01)

    rpm_ss = unit.current_rpm
    assert rpm_ss > rpm_step1  # Lag caused gradual spin-up
    assert t_n > 1.0           # Substantial thrust produced
    assert i_a > 0.5           # Current draw > idle current
    assert p_w > 5.0           # Positive electrical power


def test_propulsion_system_frd_forces_and_moments():
    # 4-rotor Quad-X setup
    rotors = [
        RotorPlacement(rotor_id=1, position_b=np.array([+0.18, +0.18, 0.0]), axis_b=np.array([0, 0, -1]), spin_direction=-1),
        RotorPlacement(rotor_id=2, position_b=np.array([-0.18, -0.18, 0.0]), axis_b=np.array([0, 0, -1]), spin_direction=-1),
        RotorPlacement(rotor_id=3, position_b=np.array([+0.18, -0.18, 0.0]), axis_b=np.array([0, 0, -1]), spin_direction=+1),
        RotorPlacement(rotor_id=4, position_b=np.array([-0.18, +0.18, 0.0]), axis_b=np.array([0, 0, -1]), spin_direction=+1),
    ]
    motors = [MotorConfig(kv=880.0, tau_m=0.01)]
    props = [PropellerConfig(diameter_m=0.254)]
    propulsion = PropulsionSystem(rotors, motors, props)

    # Symmetrical throttle [0.6, 0.6, 0.6, 0.6]
    cmds = np.array([0.6, 0.6, 0.6, 0.6])
    # Run to steady-state
    for _ in range(30):
        force_b, torque_b, current_a, power_w, rpms = propulsion.step(
            motor_commands=cmds, bus_voltage_v=15.0, dt_sec=0.01
        )

    # Upward thrust is -Z in FRD body frame!
    assert force_b[2] < -5.0
    assert np.isclose(force_b[0], 0.0, atol=1e-3)
    assert np.isclose(force_b[1], 0.0, atol=1e-3)

    # Symmetric throttle should produce near-zero net roll, pitch, yaw moments
    assert np.isclose(torque_b[0], 0.0, atol=1e-3)
    assert np.isclose(torque_b[1], 0.0, atol=1e-3)
    assert np.isclose(torque_b[2], 0.0, atol=1e-3)

    # Positive power and current
    assert current_a > 2.0
    assert power_w > 20.0
