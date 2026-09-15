"""Tests for math_models.fixed_wing — aerodynamics, trim solver, and linearisation."""
import math
import numpy as np
import pytest

from drone_sdk.math_models.fixed_wing import (
    FixedWingModel,
    FixedWingParams,
    ControlSurfaces,
)
from drone_sdk.math_models.rigid_body import (
    RigidBodyState,
    GRAVITY_MPS2,
)


class TestFixedWingModel:
    def test_properties(self):
        model = FixedWingModel()
        assert model.uav_type == "fixed_wing"
        assert model.num_motors == 1

    def test_trim_solver_convergence(self):
        """Trim solver should converge for level cruise at 15 m/s."""
        model = FixedWingModel()
        alpha, de, dt, converged = model.trim_solver(airspeed_target=15.0)
        assert converged is True
        assert 0.0 < alpha < math.radians(15.0)
        assert abs(de) < math.radians(20.0)
        assert 0.0 < dt <= 1.0

    def test_lift_at_trim(self):
        """At trim condition, lift generated should approximately equal aircraft weight."""
        model = FixedWingModel()
        alpha, de, dt, converged = model.trim_solver(airspeed_target=15.0)
        assert converged is True

        CL, CD, CY, Cl, Cm, Cn = model.compute_aero_coefficients(
            alpha=alpha, beta=0.0, p_hat=0.0, q_hat=0.0, r_hat=0.0,
            da=0.0, de=de, dr=0.0,
        )
        q_bar = 0.5 * 1.225 * 15.0 ** 2
        lift = q_bar * model.params.wing_area_m2 * CL
        weight = model.params.mass_kg * GRAVITY_MPS2
        assert abs(lift - weight) / weight < 0.01  # Within 1%

    def test_linearisation_stability(self):
        """Longitudinal A matrix should yield stable (non-positive real part) eigenvalues."""
        model = FixedWingModel()
        alpha, de, dt, converged = model.trim_solver(airspeed_target=15.0)
        A_lon, B_lon = model.linearise_at_trim(alpha_trim=alpha, airspeed=15.0)

        assert A_lon.shape == (4, 4)
        assert B_lon.shape == (4, 2)

        eigenvals = np.linalg.eigvals(A_lon)
        # Real parts of phugoid and short-period should be stable (< 0.05 margin)
        for ev in eigenvals:
            assert ev.real < 0.05

    def test_stall_model(self):
        """Alpha beyond stall angle should trigger post-stall model."""
        model = FixedWingModel()
        alpha_high = math.radians(20.0)  # > 15 deg stall
        CL, CD, _, _, _, _ = model.compute_aero_coefficients(
            alpha=alpha_high, beta=0.0, p_hat=0.0, q_hat=0.0, r_hat=0.0,
            da=0.0, de=0.0, dr=0.0,
        )
        assert CL > 0.0
        assert CD > model.params.CD0
