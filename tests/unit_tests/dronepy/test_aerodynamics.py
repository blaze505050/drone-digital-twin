"""
Unit tests for Aerodynamics models (AnalyticalDrag, OpenFOAMAeroDatabase, NeuralAeroModel).
"""
import numpy as np
import pytest

import dronepy


def test_analytical_drag():
    aero = dronepy.AnalyticalDrag(cd_x=0.85, cd_y=0.85, cd_z=1.25)
    v_air = np.array([10.0, 0.0, 0.0])  # 10 m/s forward
    omega = np.zeros(3)

    forces, moments = aero.compute_forces_and_moments(v_air, omega, air_density=1.225)
    # Drag acts opposite to relative airspeed
    assert forces[0] < 0.0
    assert forces[1] == pytest.approx(0.0)
    assert forces[2] == pytest.approx(0.0)
    assert np.allclose(moments, 0.0)


def test_openfoam_aero_database():
    aero_cfd = dronepy.OpenFOAMAeroDatabase()
    v_air = np.array([5.0, 0.0, 2.0])
    omega = np.zeros(3)
    forces, moments = aero_cfd.compute_forces_and_moments(v_air, omega)
    assert len(forces) == 3
    assert len(moments) == 3


def test_neural_aero_model_fallback():
    surrogate = dronepy.NeuralAeroModel()
    v_air = np.array([8.0, 2.0, -1.0])
    omega = np.array([0.1, 0.0, 0.0])
    forces, moments = surrogate.compute_forces_and_moments(v_air, omega)
    assert len(forces) == 3
    assert len(moments) == 3
