"""
Unit tests for Drone vehicle model, factories, and mass properties.
"""
import numpy as np
import pytest

import dronepy


def test_drone_quadcopter_factory():
    drone = dronepy.Drone.quadcopter(mass=1.5, arm_length=0.25)
    assert drone.num_motors == 4
    assert drone.mass == pytest.approx(1.5)
    assert drone.mass_kg == pytest.approx(1.5)
    assert drone.arm_length == pytest.approx(0.25)
    assert len(drone.motors) == 4
    assert len(drone.propellers) == 4
    assert drone.inertia_tensor.shape == (3, 3)
    assert np.allclose(drone.motor_positions.shape, (4, 3))


def test_drone_hexacopter_and_octocopter_factories():
    hexa = dronepy.Drone.hexacopter(mass=2.5, arm_length=0.35)
    assert hexa.num_motors == 6
    assert len(hexa.motors) == 6
    assert hexa.mass == pytest.approx(2.5)

    octo = dronepy.Drone.octocopter(mass=4.0, arm_length=0.45)
    assert octo.num_motors == 8
    assert len(octo.motors) == 8
    assert octo.mass == pytest.approx(4.0)


def test_payload_addition_and_release():
    drone = dronepy.Drone.quadcopter(mass=1.5)
    initial_inertia = drone.inertia_tensor.copy()

    # Add payload 0.5 kg offset by 0.1m along X
    drone.add_payload(mass_kg=0.5, offset_m=np.array([0.1, 0.0, 0.05]))
    assert drone.mass == pytest.approx(2.0)
    assert drone.payload_mass == pytest.approx(0.5)
    # CG should have shifted forward
    assert drone.center_of_gravity[0] > 0.0
    # Inertia tensor should have increased
    assert drone.inertia_tensor[0, 0] >= initial_inertia[0, 0]

    # Release payload
    drone.release_payload()
    assert drone.mass == pytest.approx(1.5)
    assert drone.payload_mass == pytest.approx(0.0)


def test_drone_copy():
    drone = dronepy.Drone.quadcopter(mass=1.5)
    drone_copy = drone.copy()
    assert drone_copy is not drone
    assert drone_copy.mass == drone.mass
    drone_copy.mass_kg = 2.0
    assert drone.mass == pytest.approx(1.5)
    assert drone_copy.mass == pytest.approx(2.0)
