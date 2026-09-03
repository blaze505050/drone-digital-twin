"""Tests for Blade Element Momentum Theory (BEMT) propeller analysis."""
import math
import numpy as np
import pytest

from drone_sdk.cad_engine import (
    BladeElementMomentumSolver,
    DroneGeometryBuilder,
    PropellerGeometry,
    BEMTResult,
)


def test_propeller_geometry_defaults():
    geom = PropellerGeometry()
    assert geom.radius == 0.127
    assert geom.diameter == 0.254
    assert geom.num_blades == 2
    assert geom.disk_area > 0.05
    assert geom.chord_at(geom.hub_radius) > geom.chord_at(geom.radius)
    assert geom.twist_rad_at(geom.hub_radius) > geom.twist_rad_at(geom.radius)
    assert geom.polar_moment_of_inertia > 0.0


def test_bemt_hover_solve():
    solver = BladeElementMomentumSolver()
    # Typical multirotor hover RPM ~5000 RPM
    res = solver.solve(rpm=5200.0, v_inf=0.0)

    assert isinstance(res, BEMTResult)
    assert res.rpm == 5200.0
    assert res.thrust_n > 1.0   # Expected ~1.2 to 5 N for 10" prop at 5200 RPM
    assert res.torque_nm > 0.01
    assert res.power_w > 5.0
    assert res.ct > 0.0
    assert res.cp > 0.0
    assert 0.0 <= res.figure_of_merit <= 1.0
    assert len(res.r_stations) == 30
    assert len(res.dthrust_dr) == 30
    assert res.to_dict()["thrust_n"] == round(res.thrust_n, 3)


def test_bemt_zero_rpm():
    solver = BladeElementMomentumSolver()
    res = solver.solve(rpm=0.0)
    assert res.thrust_n == 0.0
    assert res.torque_nm == 0.0
    assert res.power_w == 0.0


def test_bemt_polar_sweep():
    solver = BladeElementMomentumSolver()
    rpms = np.array([3000.0, 4500.0, 6000.0])
    sweep = solver.polar_sweep(rpms)

    assert len(sweep["thrust_n"]) == 3
    # Thrust increases monotonically with RPM squared
    assert sweep["thrust_n"][0] < sweep["thrust_n"][1] < sweep["thrust_n"][2]
    assert sweep["power_w"][0] < sweep["power_w"][1] < sweep["power_w"][2]


def test_cad_to_frame_config():
    cfg = DroneGeometryBuilder.to_frame_config(arm_length=0.28, arm_diameter=0.016)
    assert getattr(cfg, "arm_length") == 0.28
    assert getattr(cfg, "arm_diameter_o") == 0.016
    assert getattr(cfg, "n_arms") == 4
