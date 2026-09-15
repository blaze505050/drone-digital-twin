"""Tests for math_models.hexarotor — dynamics, mixing, and failure tolerance."""
import math
import numpy as np
import pytest

from drone_sdk.math_models.hexarotor import (
    HexarotorModel,
    HexarotorParams,
    HexConfig,
    mixing_matrix_hex_flat,
    mixing_matrix_coaxial_y6,
)
from drone_sdk.math_models.rigid_body import (
    RigidBodyState,
    GRAVITY_MPS2,
)


class TestHexarotorMixingMatrix:
    def test_flat_hex_shape(self):
        M = mixing_matrix_hex_flat(arm=0.30, c_tau=0.015)
        assert M.shape == (4, 6)
        # All thrust coefficients are 1
        np.testing.assert_allclose(M[0, :], 1.0)
        # Yaw coefficients alternate signs
        assert M[3, 0] > 0
        assert M[3, 1] < 0
        assert M[3, 2] > 0

    def test_coaxial_y6_shape_and_scaling(self):
        M = mixing_matrix_coaxial_y6(arm=0.30, c_tau=0.015, coax_factor=0.85)
        assert M.shape == (4, 6)
        # Upper rotors are 1.0, lower rotors are 0.85
        assert M[0, 0] == pytest.approx(1.0)
        assert M[0, 1] == pytest.approx(0.85)
        assert M[0, 2] == pytest.approx(1.0)
        assert M[0, 3] == pytest.approx(0.85)


class TestHexarotorModel:
    def test_properties(self):
        model = HexarotorModel()
        assert model.uav_type == "hexarotor"
        assert model.num_motors == 6

    def test_hover_equilibrium(self):
        model = HexarotorModel()
        state = RigidBodyState(pos_ned=np.array([0, 0, -10]))
        hover_cmd = model.hover_command()

        # Step through motor lag to settle
        for _ in range(50):
            wrench = model.compute_wrench(state, hover_cmd, dt=0.01)

        weight = model.params.mass_kg * GRAVITY_MPS2
        assert abs(wrench.force_body[2] + weight) < weight * 0.05

    def test_single_motor_failure_and_recovery(self):
        model = HexarotorModel()
        assert model.can_sustain_hover_with_failure()

        # Fail motor index 2
        model.fail_motor(2)
        assert 2 in model._failed_motors
        assert model.can_sustain_hover_with_failure()

        # Check allocated command for failed motor is 0
        cmd = model.allocate_motors(desired_thrust=model.params.mass_kg * GRAVITY_MPS2, desired_torques=np.zeros(3))
        assert cmd[2] == pytest.approx(0.0)

        # Recover motor
        model.recover_motor(2)
        assert 2 not in model._failed_motors
        cmd_recovered = model.allocate_motors(desired_thrust=model.params.mass_kg * GRAVITY_MPS2, desired_torques=np.zeros(3))
        assert cmd_recovered[2] > 0.0

    def test_coaxial_config_instantiation(self):
        params = HexarotorParams(config=HexConfig.COAXIAL_Y6)
        model = HexarotorModel(params)
        assert model.params.config == HexConfig.COAXIAL_Y6
        assert model.num_motors == 6
