"""
Unit tests for Environment, atmosphere, and wind models.
"""
import numpy as np
import pytest

import dronepy


def test_standard_atmosphere():
    env = dronepy.Environment.standard_atmosphere()
    assert env.sea_level_pressure_pa == pytest.approx(101325.0)
    assert env.sea_level_temperature_k == pytest.approx(288.15)
    assert env.density_at(0.0) == pytest.approx(1.225, rel=0.01)

    # At 1000m altitude, density should be lower
    rho_1000 = env.density_at(1000.0)
    assert rho_1000 < 1.225
    assert rho_1000 > 1.0


def test_constant_wind():
    wind = dronepy.Wind.constant(north=3.0, east=4.0, down=0.0)
    env = dronepy.Environment(wind=wind)
    w_vec = env.get_wind_ned(altitude_agl=10.0, time=0.0)
    assert w_vec[0] == pytest.approx(3.0)
    assert w_vec[1] == pytest.approx(4.0)
    assert w_vec[2] == pytest.approx(0.0)


def test_wind_gust():
    gust = dronepy.Wind.gust(
        magnitude=5.0,
        direction_deg=0.0,
        start_time=2.0,
        duration=1.0,
    )
    # Before gust
    w_before = gust.get_wind(altitude_agl=10.0, time=1.0)
    assert np.allclose(w_before, 0.0)

    # At peak of 1-cosine gust (start + duration/2 = 2.5s)
    w_peak = gust.get_wind(altitude_agl=10.0, time=2.5)
    assert w_peak[0] == pytest.approx(5.0, rel=0.05)

    # After gust
    w_after = gust.get_wind(altitude_agl=10.0, time=4.0)
    assert np.allclose(w_after, 0.0)
