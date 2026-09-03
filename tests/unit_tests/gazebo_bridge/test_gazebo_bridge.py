"""
Tests for drone_sdk.gazebo_bridge (Module 7).

Tests SDF generation, world building, and config validation.
All tests work without Gazebo or gz-transport installed.
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from drone_sdk.gazebo_bridge import (
    DroneModelConfig,
    GazeboSource,
    GpsSensorConfig,
    ImuSensorConfig,
    RotorConfig,
    SDFBuilder,
    WorldConfig,
    WorldManager,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def builder():
    return SDFBuilder()


@pytest.fixture
def default_drone():
    return DroneModelConfig(name="test_iris")


@pytest.fixture
def world_cfg():
    return WorldConfig(name="test_world", wind_speed=0.0)


@pytest.fixture
def windy_world():
    return WorldConfig(name="windy_world", wind_speed=5.0, wind_direction=90.0)


# ══════════════════════════════════════════════════════════════════════════════
#  RotorConfig defaults
# ══════════════════════════════════════════════════════════════════════════════

class TestRotorConfig:

    def test_default_quadrotor_has_4_rotors(self, default_drone):
        assert len(default_drone.rotors) == 4

    def test_rotor_indices_sequential(self, default_drone):
        indices = [r.index for r in default_drone.rotors]
        assert sorted(indices) == [0, 1, 2, 3]

    def test_x_config_alternating_directions(self, default_drone):
        dirs = [r.direction for r in default_drone.rotors]
        cw  = dirs.count(-1)
        ccw = dirs.count( 1)
        assert cw == 2 and ccw == 2, "X-config needs 2 CW and 2 CCW rotors"

    def test_rotor_positions_symmetric(self, default_drone):
        """Sum of rotor positions should be ~zero (symmetric layout)."""
        xs = sum(r.x for r in default_drone.rotors)
        ys = sum(r.y for r in default_drone.rotors)
        assert abs(xs) < 1e-9
        assert abs(ys) < 1e-9

    def test_custom_rotor_config(self):
        rotors = [
            RotorConfig(0,  0.15,  0.15, direction=-1),
            RotorConfig(1, -0.15, -0.15, direction=-1),
            RotorConfig(2,  0.15, -0.15, direction= 1),
            RotorConfig(3, -0.15,  0.15, direction= 1),
        ]
        cfg = DroneModelConfig(name="micro", arm_length=0.12, rotors=rotors)
        assert len(cfg.rotors) == 4


# ══════════════════════════════════════════════════════════════════════════════
#  SDFBuilder — drone model
# ══════════════════════════════════════════════════════════════════════════════

class TestSDFBuilderDroneModel:

    def test_returns_sdf_element(self, builder, default_drone):
        sdf = builder.build_drone_model(default_drone)
        assert sdf.tag == "sdf"
        assert sdf.get("version") == "1.9"

    def test_model_name_set(self, builder, default_drone):
        sdf = builder.build_drone_model(default_drone)
        model = sdf.find("model")
        assert model is not None
        assert model.get("name") == "test_iris"

    def test_base_link_exists(self, builder, default_drone):
        sdf   = builder.build_drone_model(default_drone)
        model = sdf.find("model")
        link  = model.find("link[@name='base_link']")
        assert link is not None

    def test_mass_in_inertial(self, builder, default_drone):
        sdf   = builder.build_drone_model(default_drone)
        model = sdf.find("model")
        link  = model.find("link[@name='base_link']")
        inert = link.find("inertial")
        mass  = inert.find("mass")
        assert mass is not None
        assert abs(float(mass.text) - default_drone.mass_kg) < 1e-9

    def test_4_rotor_joints_for_quadrotor(self, builder, default_drone):
        sdf    = builder.build_drone_model(default_drone)
        model  = sdf.find("model")
        joints = model.findall("joint")
        assert len(joints) == 4

    def test_imu_sensor_present(self, builder, default_drone):
        sdf   = builder.build_drone_model(default_drone)
        model = sdf.find("model")
        link  = model.find("link[@name='base_link']")
        imu   = link.find("sensor[@name='imu_sensor']")
        assert imu is not None
        assert imu.get("type") == "imu"

    def test_gps_sensor_present(self, builder, default_drone):
        sdf   = builder.build_drone_model(default_drone)
        model = sdf.find("model")
        link  = model.find("link[@name='base_link']")
        gps   = link.find("sensor[@name='gps_sensor']")
        assert gps is not None
        assert gps.get("type") == "gps"

    def test_imu_update_rate(self, builder):
        cfg = DroneModelConfig(name="fast_imu",
                               imu=ImuSensorConfig(update_rate=400.0))
        sdf   = builder.build_drone_model(cfg)
        model = sdf.find("model")
        link  = model.find("link[@name='base_link']")
        imu   = link.find("sensor[@name='imu_sensor']")
        rate  = imu.find("update_rate")
        assert abs(float(rate.text) - 400.0) < 1e-9

    def test_gps_update_rate(self, builder):
        cfg = DroneModelConfig(name="fast_gps",
                               gps=GpsSensorConfig(update_rate=10.0))
        sdf   = builder.build_drone_model(cfg)
        model = sdf.find("model")
        link  = model.find("link[@name='base_link']")
        gps   = link.find("sensor[@name='gps_sensor']")
        rate  = gps.find("update_rate")
        assert abs(float(rate.text) - 10.0) < 1e-9

    def test_mavlink_plugin_present(self, builder, default_drone):
        sdf    = builder.build_drone_model(default_drone)
        model  = sdf.find("model")
        plugin = model.find("plugin[@name='gz_mavlink_plugin']")
        assert plugin is not None

    def test_hexarotor_has_6_joints(self, builder):
        rotors = [
            RotorConfig(i, math.cos(i*math.pi/3)*0.3,
                           math.sin(i*math.pi/3)*0.3,
                           direction=(-1)**i)
            for i in range(6)
        ]
        cfg   = DroneModelConfig(name="hexa", rotors=rotors)
        sdf   = builder.build_drone_model(cfg)
        model = sdf.find("model")
        joints = model.findall("joint")
        assert len(joints) == 6


# ══════════════════════════════════════════════════════════════════════════════
#  SDFBuilder — world
# ══════════════════════════════════════════════════════════════════════════════

class TestSDFBuilderWorld:

    def test_returns_sdf_element(self, builder, world_cfg, default_drone):
        sdf = builder.build_world(world_cfg, [default_drone])
        assert sdf.tag == "sdf"

    def test_world_name_set(self, builder, world_cfg):
        sdf = builder.build_world(world_cfg)
        world = sdf.find("world")
        assert world.get("name") == "test_world"

    def test_physics_step_set(self, builder, world_cfg):
        sdf    = builder.build_world(world_cfg)
        world  = sdf.find("world")
        physics = world.find("physics")
        step   = physics.find("max_step_size")
        assert abs(float(step.text) - world_cfg.physics_step) < 1e-9

    def test_gravity_set(self, builder, world_cfg):
        sdf   = builder.build_world(world_cfg)
        world = sdf.find("world")
        grav  = world.find("gravity")
        assert "-9.81" in grav.text

    def test_ground_plane_included(self, builder, world_cfg):
        sdf   = builder.build_world(world_cfg)
        world = sdf.find("world")
        gp    = world.find("model[@name='ground_plane']")
        assert gp is not None

    def test_sun_light_included(self, builder, world_cfg):
        sdf   = builder.build_world(world_cfg)
        world = sdf.find("world")
        sun   = world.find("light[@name='sun']")
        assert sun is not None

    def test_wind_element_added_when_nonzero(self, builder, windy_world):
        sdf   = builder.build_world(windy_world)
        world = sdf.find("world")
        wind  = world.find("wind")
        assert wind is not None
        vel   = wind.find("linear_velocity")
        # wind_direction=90° → East → vy≈5, vx≈0
        parts = [float(x) for x in vel.text.split()]
        assert abs(parts[0]) < 0.01        # vx ≈ 0 (cos 90° = 0)
        assert abs(parts[1] - 5.0) < 0.01  # vy ≈ 5

    def test_no_wind_element_when_zero(self, builder, world_cfg):
        sdf   = builder.build_world(world_cfg)
        world = sdf.find("world")
        wind  = world.find("wind")
        assert wind is None

    def test_drone_include_embedded(self, builder, world_cfg, default_drone):
        sdf   = builder.build_world(world_cfg, [default_drone], [(0, 0, 0.3)])
        world = sdf.find("world")
        inc   = world.find("include")
        assert inc is not None
        uri   = inc.find("uri")
        assert "test_iris" in uri.text

    def test_multiple_drones_embedded(self, builder, world_cfg):
        drones = [
            DroneModelConfig(name=f"drone_{i}") for i in range(3)
        ]
        positions = [(i * 5.0, 0, 0.3) for i in range(3)]
        sdf   = builder.build_world(world_cfg, drones, positions)
        world = sdf.find("world")
        incs  = world.findall("include")
        assert len(incs) == 3

    def test_real_time_factor(self, builder):
        cfg = WorldConfig(name="fast_world", real_time_factor=2.0)
        sdf   = builder.build_world(cfg)
        world = sdf.find("world")
        phys  = world.find("physics")
        rtf   = phys.find("real_time_factor")
        assert abs(float(rtf.text) - 2.0) < 1e-9


# ══════════════════════════════════════════════════════════════════════════════
#  SDFBuilder — file write
# ══════════════════════════════════════════════════════════════════════════════

class TestSDFBuilderWrite:

    def test_write_creates_file(self, builder, world_cfg, tmp_path):
        sdf  = builder.build_world(world_cfg)
        path = builder.write(sdf, str(tmp_path / "test.world"))
        assert path.exists()

    def test_written_file_is_valid_xml(self, builder, world_cfg, tmp_path):
        sdf  = builder.build_world(world_cfg)
        path = builder.write(sdf, str(tmp_path / "valid.world"))
        # Should parse without error
        tree = ET.parse(str(path))
        root = tree.getroot()
        assert root.tag == "sdf"

    def test_write_creates_parent_dirs(self, builder, world_cfg, tmp_path):
        sdf  = builder.build_world(world_cfg)
        deep = tmp_path / "nested" / "dir" / "world.sdf"
        builder.write(sdf, str(deep))
        assert deep.exists()

    def test_write_includes_xml_declaration(self, builder, world_cfg, tmp_path):
        sdf  = builder.build_world(world_cfg)
        path = builder.write(sdf, str(tmp_path / "decl.world"))
        content = path.read_text()
        assert content.startswith("<?xml")

    def test_roundtrip_drone_model(self, builder, default_drone, tmp_path):
        sdf   = builder.build_drone_model(default_drone)
        path  = builder.write(sdf, str(tmp_path / "drone.sdf"))
        tree  = ET.parse(str(path))
        root  = tree.getroot()
        model = root.find("model")
        assert model.get("name") == "test_iris"


# ══════════════════════════════════════════════════════════════════════════════
#  GazeboSource lifecycle (no gz-transport required)
# ══════════════════════════════════════════════════════════════════════════════

class TestGazeboSource:

    def test_initial_status_disconnected(self):
        from drone_sdk.telemetry_engine.source import SourceStatus
        src = GazeboSource("gz_0", "drone_0")
        assert src.status == SourceStatus.DISCONNECTED

    def test_connect_without_gz_transport_raises(self):
        import sys
        src = GazeboSource("gz_0", "drone_0")
        original = sys.modules.pop("gz.transport13", None)
        try:
            with pytest.raises(ImportError, match="gz-transport13"):
                src.connect()
        finally:
            if original:
                sys.modules["gz.transport13"] = original

    def test_poll_when_disconnected_returns_empty(self):
        src = GazeboSource("gz_0", "drone_0")
        assert src.poll() == []

    def test_disconnect_sets_disconnected(self):
        from drone_sdk.telemetry_engine.source import SourceStatus
        src = GazeboSource("gz_0", "drone_0")
        src.disconnect()
        assert src.status == SourceStatus.DISCONNECTED


# ══════════════════════════════════════════════════════════════════════════════
#  WorldManager (no gz on PATH)
# ══════════════════════════════════════════════════════════════════════════════

class TestWorldManager:

    def test_spawn_without_gz_returns_false(self):
        """If gz not on PATH, spawn_model returns False gracefully."""
        import shutil
        from unittest.mock import patch
        wm = WorldManager("test_world")
        with patch.object(WorldManager, "_gz_available", return_value=False):
            result = wm.spawn_model("model.sdf", "drone_0", 0, 0, 0.3)
        assert result is False

    def test_remove_without_gz_returns_false(self):
        wm = WorldManager()
        with __import__("unittest.mock", fromlist=["patch"]).patch.object(
            WorldManager, "_gz_available", return_value=False
        ):
            assert wm.remove_model("drone_0") is False

    def test_gz_available_returns_bool(self):
        result = WorldManager._gz_available()
        assert isinstance(result, bool)

    def test_world_name_set(self):
        wm = WorldManager("my_world")
        assert wm._world == "my_world"

    def test_default_world_name(self):
        wm = WorldManager()
        assert wm._world == "drone_world"
