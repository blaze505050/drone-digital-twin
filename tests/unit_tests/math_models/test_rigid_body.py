"""Tests for math_models.rigid_body — 6-DOF dynamics and integrators."""
import math
import numpy as np
import pytest

from drone_sdk.math_models.rigid_body import (
    RigidBodyState,
    InertiaParams,
    ExternalWrench,
    RigidBodySimulator,
    compute_derivatives,
    integrate_semi_implicit_euler,
    GRAVITY_MPS2,
)
from drone_sdk.math_models.frames import quat_to_euler


class TestRigidBodyState:
    def test_default_state_is_identity(self):
        s = RigidBodyState()
        np.testing.assert_allclose(s.pos_ned, [0, 0, 0])
        np.testing.assert_allclose(s.quat, [1, 0, 0, 0])
        assert s.altitude_agl == 0.0

    def test_to_from_array_roundtrip(self):
        s = RigidBodyState(
            pos_ned=[1, 2, -5],
            vel_body=[3, 0, -1],
            quat=[1, 0, 0, 0],
            omega_body=[0.1, -0.2, 0.3],
        )
        arr = s.to_array()
        assert arr.shape == (13,)
        s2 = RigidBodyState.from_array(arr)
        np.testing.assert_allclose(s2.pos_ned, s.pos_ned)
        np.testing.assert_allclose(s2.vel_body, s.vel_body)

    def test_altitude_agl_from_ned(self):
        s = RigidBodyState(pos_ned=[0, 0, -10])
        assert abs(s.altitude_agl - 10.0) < 0.01


class TestFreefall:
    """A body with zero thrust should accelerate at g downward."""

    def test_free_fall_acceleration(self):
        sim = RigidBodySimulator(
            inertia=InertiaParams(mass_kg=1.0),
            state=RigidBodyState(pos_ned=np.array([0, 0, -100])),
        )
        dt = 0.01
        wrench = ExternalWrench()  # No forces

        for _ in range(100):  # 1 second
            sim.step(wrench, dt, n_substeps=1)

        # After 1s free-fall: vz ≈ +g ≈ 9.81 m/s (downward in NED)
        vel_ned = sim.state.vel_ned
        assert vel_ned[2] > 9.0  # Close to g
        assert vel_ned[2] < 11.0

    def test_free_fall_position(self):
        sim = RigidBodySimulator(
            inertia=InertiaParams(mass_kg=2.0),
            state=RigidBodyState(pos_ned=np.array([0, 0, -100])),
        )
        dt = 0.01
        wrench = ExternalWrench()

        for _ in range(100):  # 1 second
            sim.step(wrench, dt, n_substeps=1)

        # z should have moved ~0.5*g*t² ≈ 4.9 m downward (NED: z increases)
        dz = sim.state.pos_ned[2] - (-100)
        assert dz > 4.0
        assert dz < 6.0


class TestTorqueFreeBody:
    """A symmetric torque-free body should spin at constant rate."""

    def test_constant_spin_rate(self):
        omega_init = np.array([0.0, 0.0, 5.0])  # 5 rad/s about z
        sim = RigidBodySimulator(
            inertia=InertiaParams(
                mass_kg=1.0,
                inertia_tensor=np.diag([0.1, 0.1, 0.1]),  # Symmetric
            ),
            state=RigidBodyState(
                pos_ned=np.array([0, 0, -10]),
                omega_body=omega_init,
            ),
        )
        wrench = ExternalWrench()

        for _ in range(1000):
            sim.step(wrench, 0.01, n_substeps=2)

        # Spin rate should remain ~5 rad/s about z
        assert abs(sim.state.omega_body[2] - 5.0) < 0.5
        # Off-axis rates should stay near zero
        assert abs(sim.state.omega_body[0]) < 0.5
        assert abs(sim.state.omega_body[1]) < 0.5


class TestQuaternionNormPreservation:
    """Quaternion norm should be preserved over many integration steps."""

    def test_quat_norm_over_10000_steps(self):
        sim = RigidBodySimulator(
            state=RigidBodyState(
                pos_ned=np.array([0, 0, -10]),
                omega_body=np.array([1.0, -0.5, 0.3]),
            ),
        )
        wrench = ExternalWrench()

        for _ in range(10000):
            sim.step(wrench, 0.001, n_substeps=1)

        norm = np.linalg.norm(sim.state.quat)
        assert abs(norm - 1.0) < 1e-6


class TestInertiaParams:
    def test_diagonal_inertia(self):
        p = InertiaParams(inertia_tensor=np.array([1, 2, 3]))
        np.testing.assert_allclose(p.inertia_tensor, np.diag([1, 2, 3]))
        inv = p.inertia_inv
        np.testing.assert_allclose(inv, np.diag([1, 0.5, 1/3]), atol=1e-10)
