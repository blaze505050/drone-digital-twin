"""Tests for math_models.vtol_tiltrotor — hybrid hover, transition, and cruise dynamics."""
import math
import numpy as np
import pytest

from drone_sdk.math_models.vtol_tiltrotor import (
    VTOLTiltrotorModel,
    VTOLParams,
    VTOLFlightPhase,
)
from drone_sdk.math_models.rigid_body import (
    RigidBodyState,
    GRAVITY_MPS2,
)


class TestVTOLTiltrotorModel:
    def test_properties(self):
        model = VTOLTiltrotorModel()
        assert model.uav_type == "vtol_tiltrotor"
        assert model.num_motors == 4
        assert model.flight_phase == VTOLFlightPhase.HOVER
        assert model.transition_progress == pytest.approx(0.0)

    def test_hover_equilibrium(self):
        """In hover mode (tilt=0), hover motor commands should balance aircraft weight."""
        model = VTOLTiltrotorModel()
        state = RigidBodyState(pos_ned=np.array([0, 0, -10]))
        hover_cmd = model.hover_command()

        # Step through motor lag to settle
        for _ in range(50):
            wrench = model.compute_wrench(state, hover_cmd, dt=0.01)

        weight = model.params.mass_kg * GRAVITY_MPS2
        assert abs(wrench.force_body[2] + weight) < weight * 0.05
        # Forward force should be negligible in hover
        assert abs(wrench.force_body[0]) < 0.1

    def test_transition_corridor(self):
        """Initiating transition should smoothly rotate tilt angle from 0 to 90 degrees."""
        model = VTOLTiltrotorModel()
        model.begin_transition()
        assert model._target_tilt_rad == pytest.approx(math.pi / 2.0)

        # Advance 1.5 seconds (at 30 deg/s max rate -> 45 deg tilt)
        state = RigidBodyState()
        hover_cmd = model.hover_command()
        for _ in range(150):
            model.compute_wrench(state, hover_cmd, dt=0.01)

        assert model.flight_phase == VTOLFlightPhase.TRANSITION
        assert 0.3 < model.transition_progress < 0.7

        # Advance another 2.5 seconds (total 4.0s > 3.0s needed for 90 deg)
        for _ in range(250):
            model.compute_wrench(state, hover_cmd, dt=0.01)

        assert model.flight_phase == VTOLFlightPhase.CRUISE
        assert model.transition_progress == pytest.approx(1.0, abs=1e-3)

    def test_thrust_vectoring_in_cruise(self):
        """At 90 deg tilt, rotor thrust should act along body +x (forward)."""
        model = VTOLTiltrotorModel()
        model.set_tilt_target(90.0)
        state = RigidBodyState()
        hover_cmd = model.hover_command()

        # Step until fully tilted and motors settled
        for _ in range(350):
            wrench = model.compute_wrench(state, hover_cmd, dt=0.01)

        weight = model.params.mass_kg * GRAVITY_MPS2
        # Rotor thrust should now be along +x
        assert wrench.force_body[0] == pytest.approx(weight, rel=0.05)
        # Vertical force from rotors should be near 0
        assert abs(wrench.force_body[2]) < 0.5

    def test_back_transition(self):
        """begin_back_transition should command tilt target back to 0."""
        model = VTOLTiltrotorModel()
        model.set_tilt_target(90.0)
        state = RigidBodyState()
        for _ in range(350):
            model.compute_wrench(state, model.hover_command(), dt=0.01)

        assert model.flight_phase == VTOLFlightPhase.CRUISE
        model.begin_back_transition()
        assert model._target_tilt_rad == pytest.approx(0.0)

        for _ in range(350):
            model.compute_wrench(state, model.hover_command(), dt=0.01)

        assert model.flight_phase == VTOLFlightPhase.HOVER
        assert model.transition_progress == pytest.approx(0.0, abs=1e-3)
