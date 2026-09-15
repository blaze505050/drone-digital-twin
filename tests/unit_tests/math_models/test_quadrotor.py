"""Tests for math_models.quadrotor — mixing matrix, hover, ground effect."""
import math
import numpy as np
import pytest

from drone_sdk.math_models.quadrotor import (
    QuadrotorModel,
    QuadrotorParams,
    QuadConfig,
    mixing_matrix_x,
    mixing_matrix_plus,
    ground_effect_factor,
)
from drone_sdk.math_models.rigid_body import (
    RigidBodyState,
    RigidBodySimulator,
    InertiaParams,
    GRAVITY_MPS2,
)


class TestMixingMatrix:
    def test_x_config_total_thrust(self):
        """Equal motor thrusts → total thrust = sum, zero torques."""
        M = mixing_matrix_x(0.25, 0.015)
        thrusts = np.array([5.0, 5.0, 5.0, 5.0])
        result = M @ thrusts
        assert abs(result[0] - 20.0) < 1e-10  # Total thrust
        assert abs(result[1]) < 1e-10          # Roll torque = 0
        assert abs(result[2]) < 1e-10          # Pitch torque = 0
        assert abs(result[3]) < 1e-10          # Yaw torque = 0

    def test_x_config_roll_sign(self):
        """Higher right motors → positive roll (right roll in FRD)."""
        M = mixing_matrix_x(0.25, 0.015)
        thrusts = np.array([3.0, 5.0, 3.0, 5.0])  # Right motors higher
        result = M @ thrusts
        assert result[1] > 0  # Positive roll torque

    def test_plus_config_pitch(self):
        """Front motor higher → positive pitch."""
        M = mixing_matrix_plus(0.25, 0.015)
        thrusts = np.array([6.0, 4.0, 4.0, 4.0])  # Front motor higher
        result = M @ thrusts
        assert result[2] > 0  # Positive pitch torque


class TestGroundEffect:
    def test_no_ground_effect_high(self):
        """At altitude >> 4R, ground effect factor = 1.0."""
        assert ground_effect_factor(10.0, 0.127) == 1.0

    def test_ground_effect_at_half_radius(self):
        """At altitude = R/2, ground effect should significantly augment thrust."""
        factor = ground_effect_factor(0.0635, 0.127)
        assert factor > 1.05
        assert factor <= 1.3

    def test_ground_effect_clamped(self):
        """Ground effect factor should be clamped to ≤ 1.3."""
        factor = ground_effect_factor(0.01, 0.127)
        assert factor <= 1.3


class TestQuadrotorModel:
    def test_hover_equilibrium(self):
        """At hover command, total thrust should equal weight → zero vertical acceleration."""
        model = QuadrotorModel()
        state = RigidBodyState(pos_ned=np.array([0, 0, -5]))

        hover_cmd = model.hover_command()
        # Step through motor lag to reach steady-state thrust
        for _ in range(50):
            wrench = model.compute_wrench(state, hover_cmd, dt=0.01)

        # Force in body z should be close to -mg (upward)
        weight = model.params.mass_kg * GRAVITY_MPS2
        assert abs(wrench.force_body[2] + weight) < weight * 0.05  # Within 5%

    def test_thrust_to_weight_ratio(self):
        """Default quadrotor should have T/W > 1 (able to hover)."""
        model = QuadrotorModel()
        assert model.thrust_to_weight_ratio > 1.0

    def test_hover_command_symmetry(self):
        """All 4 hover commands should be equal."""
        model = QuadrotorModel()
        hover = model.hover_command()
        assert all(abs(hover[i] - hover[0]) < 1e-10 for i in range(4))

    def test_allocate_and_recover(self):
        """Control allocation round-trips through mixing matrix."""
        model = QuadrotorModel()
        desired_thrust = model.params.mass_kg * GRAVITY_MPS2
        desired_torques = np.array([0.1, -0.05, 0.02])
        cmds = model.allocate_motors(desired_thrust, desired_torques)
        assert cmds.shape == (4,)
        assert all(0 <= c <= 1 for c in cmds)

    def test_uav_type_and_motors(self):
        model = QuadrotorModel()
        assert model.uav_type == "quadrotor"
        assert model.num_motors == 4


class TestQuadrotorSimulation:
    def test_hover_simulation_no_crash(self):
        """Run a hover simulation for 5s without NaN or crash."""
        params = QuadrotorParams()
        model = QuadrotorModel(params)
        sim = RigidBodySimulator(
            inertia=params.to_inertia_params,
            state=RigidBodyState(pos_ned=np.array([0, 0, -5])),
        )

        hover_cmd = model.hover_command()
        for _ in range(1000):  # 5s at 200 Hz
            wrench = model.compute_wrench(sim.state, hover_cmd, dt=0.005)
            sim.step(wrench, 0.005, n_substeps=2)

        assert not np.any(np.isnan(sim.state.pos_ned))
        assert not np.any(np.isnan(sim.state.quat))
        # Should still be near 5m altitude
        alt = sim.state.altitude_agl
        assert alt > 2.0
        assert alt < 15.0
