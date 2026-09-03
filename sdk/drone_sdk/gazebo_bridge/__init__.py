"""
drone_sdk.gazebo_bridge
=======================
Gazebo Integration — Module 7 of the UAV Digital Twin Platform.

Provides:
1. SDF world file generation (programmatic, parametric)
2. Drone model configuration (sensor layout, rotor placement)
3. GazeboSource — TelemetrySource reading Gazebo state via gz-transport
4. GazeboVisualiser — commands Gazebo model pose from digital twin state
5. WorldManager — spawn/remove models at runtime

Supports
--------
* Gazebo Harmonic (gz-sim 8) — the current ROS2 Humble-paired version
* Gazebo Classic 11 (legacy, via different transport interface)
* PX4-Gazebo bridge (px4-ros2-bridge / micro-xrce-dds)

Dependency matrix
-----------------
Core (SDF generation, world files): no external deps
GazeboSource / GazeboVisualiser: requires gz-transport Python bindings
    pip install gz-transport13   (or from source)

Python version: 3.9+
"""
from __future__ import annotations

import logging
import math
import os
import subprocess
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom

from drone_sdk.state_manager import DataSource, DroneStateUpdate, DroneStateVector
from drone_sdk.telemetry_engine.source import SourcePriority, SourceStatus, TelemetrySource

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  SDF Model and World Builders
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RotorConfig:
    """Configuration for a single rotor."""
    index:      int      # 0-based
    x:          float    # position relative to body centre (m)
    y:          float
    z:          float    = 0.1
    direction:  int      = 1    # +1 = CCW, -1 = CW (viewed from above)
    radius:     float    = 0.128  # m
    max_rpm:    float    = 15_000.0


@dataclass
class ImuSensorConfig:
    update_rate:         float = 200.0   # Hz
    accel_noise_density: float = 3.4e-3
    accel_bias_walk:     float = 4.3e-6
    gyro_noise_density:  float = 8.7e-5
    gyro_bias_walk:      float = 2.6e-8


@dataclass
class GpsSensorConfig:
    update_rate:           float = 5.0    # Hz
    position_noise_m:      float = 0.50   # 1-sigma, m
    velocity_noise_ms:     float = 0.10   # m/s


@dataclass
class DroneModelConfig:
    """Full drone model configuration for SDF generation."""
    name:       str               = "iris"
    mass_kg:    float             = 1.5
    ixx:        float             = 0.0348
    iyy:        float             = 0.0459
    izz:        float             = 0.0977
    arm_length: float             = 0.25    # m, centre-to-rotor
    rotors:     List[RotorConfig] = field(default_factory=list)
    imu:        ImuSensorConfig   = field(default_factory=ImuSensorConfig)
    gps:        GpsSensorConfig   = field(default_factory=GpsSensorConfig)
    has_lidar:  bool              = False
    has_camera: bool              = True

    def __post_init__(self):
        if not self.rotors:
            # Default quadrotor X configuration
            arm = self.arm_length / math.sqrt(2)
            self.rotors = [
                RotorConfig(0,  arm,  arm, direction=-1),  # front-right CW
                RotorConfig(1, -arm, -arm, direction=-1),  # back-left   CW
                RotorConfig(2,  arm, -arm, direction= 1),  # back-right  CCW
                RotorConfig(3, -arm,  arm, direction= 1),  # front-left  CCW
            ]


@dataclass
class WorldConfig:
    """Configuration for a Gazebo world."""
    name:           str               = "drone_world"
    gravity:        float             = -9.81  # m/s²
    wind_speed:     float             = 0.0    # m/s
    wind_direction: float             = 0.0    # degrees from North
    ground_texture: str               = "grass"
    sky_type:       str               = "default"
    physics_step:   float             = 0.004  # s (250 Hz)
    real_time_factor: float           = 1.0
    include_models: List[str]         = field(default_factory=list)  # model:// URIs


class SDFBuilder:
    """Generates SDF XML for Gazebo worlds and drone models.

    Produces standards-compliant SDF 1.9 documents that work with
    Gazebo Harmonic (gz-sim 8) and PX4 Gazebo bridges.

    Usage::

        model_cfg = DroneModelConfig(name="my_drone")
        world_cfg = WorldConfig(name="test_flight")

        builder = SDFBuilder()
        world_sdf = builder.build_world(world_cfg, [model_cfg])
        builder.write(world_sdf, "/tmp/test_world.world")
    """

    def build_world(
        self,
        world: WorldConfig,
        drones: Optional[List[DroneModelConfig]] = None,
        drone_start_positions: Optional[List[tuple]] = None,
    ) -> ET.Element:
        """Build a complete Gazebo world SDF element.

        Args:
            world:                  World configuration.
            drones:                 List of drone models to include.
            drone_start_positions:  List of (x, y, z) tuples, one per drone.

        Returns:
            xml.etree.ElementTree.Element (the <sdf> root).
        """
        sdf = ET.Element("sdf", version="1.9")
        w   = ET.SubElement(sdf, "world", name=world.name)

        # Physics
        physics = ET.SubElement(w, "physics", name="default_physics", type="ode")
        self._text(physics, "max_step_size", f"{world.physics_step}")
        self._text(physics, "real_time_factor", f"{world.real_time_factor}")
        self._text(physics, "real_time_update_rate",
                   f"{1.0/world.physics_step:.0f}")

        # Gravity
        ET.SubElement(w, "gravity").text = f"0 0 {world.gravity}"

        # Magnetic field (WGS-84 default for Zurich)
        ET.SubElement(w, "magnetic_field").text = "6e-06 2.3e-05 -4.2e-05"

        # Atmosphere
        ET.SubElement(w, "atmosphere", type="adiabatic")

        # Scene
        scene = ET.SubElement(w, "scene")
        self._text(scene, "ambient", "0.4 0.4 0.4 1")
        self._text(scene, "background", "0.7 0.7 0.7 1")
        self._text(scene, "shadows", "true")

        # Sun
        sun = ET.SubElement(w, "light", name="sun", type="directional")
        self._text(sun, "cast_shadows", "true")
        self._text(sun, "pose", "0 0 10 0 0 0")
        self._text(sun, "diffuse", "0.8 0.8 0.8 1")
        self._text(sun, "specular", "0.2 0.2 0.2 1")
        direc = ET.SubElement(sun, "direction")
        direc.text = "-0.5 0.1 -0.9"

        # Ground plane
        gp = ET.SubElement(w, "model", name="ground_plane")
        self._text(gp, "static", "true")
        gp_link = ET.SubElement(gp, "link", name="link")
        gp_col  = ET.SubElement(gp_link, "collision", name="collision")
        gp_colg = ET.SubElement(gp_col, "geometry")
        ET.SubElement(gp_colg, "plane").append(self._make_plane_normal())
        gp_vis  = ET.SubElement(gp_link, "visual", name="visual")
        gp_visg = ET.SubElement(gp_vis, "geometry")
        ET.SubElement(gp_visg, "plane").append(self._make_plane_normal())

        # Wind
        if world.wind_speed > 0:
            wind = ET.SubElement(w, "wind")
            angle = math.radians(world.wind_direction)
            wx = world.wind_speed * math.cos(angle)
            wy = world.wind_speed * math.sin(angle)
            self._text(wind, "linear_velocity", f"{wx:.3f} {wy:.3f} 0")

        # Drone models
        if drones:
            positions = drone_start_positions or [(0, 0, 0.3)] * len(drones)
            for i, (drone, pos) in enumerate(zip(drones, positions)):
                self._embed_drone(w, drone, pos, instance=i)

        return sdf

    def build_drone_model(self, cfg: DroneModelConfig) -> ET.Element:
        """Build a standalone drone model SDF element."""
        sdf   = ET.Element("sdf", version="1.9")
        model = ET.SubElement(sdf, "model", name=cfg.name)
        self._text(model, "pose", "0 0 0 0 0 0")

        # Base link (fuselage)
        link = ET.SubElement(model, "link", name="base_link")
        self._text(link, "pose", "0 0 0 0 0 0")

        # Inertial
        inertial = ET.SubElement(link, "inertial")
        self._text(inertial, "mass", str(cfg.mass_kg))
        inertia  = ET.SubElement(inertial, "inertia")
        self._text(inertia, "ixx", str(cfg.ixx))
        self._text(inertia, "iyy", str(cfg.iyy))
        self._text(inertia, "izz", str(cfg.izz))
        self._text(inertia, "ixy", "0")
        self._text(inertia, "ixz", "0")
        self._text(inertia, "iyz", "0")

        # Collision (simple box approximation)
        col = ET.SubElement(link, "collision", name="collision")
        col_geom = ET.SubElement(col, "geometry")
        box = ET.SubElement(col_geom, "box")
        ET.SubElement(box, "size").text = "0.47 0.47 0.11"

        # Visual (box placeholder — replace with mesh in production)
        vis = ET.SubElement(link, "visual", name="visual")
        vis_geom = ET.SubElement(vis, "geometry")
        vis_box  = ET.SubElement(vis_geom, "box")
        ET.SubElement(vis_box, "size").text = "0.47 0.47 0.11"

        # IMU sensor
        self._add_imu(link, cfg.imu)

        # GPS sensor
        self._add_gps(link, cfg.gps)

        # Rotors
        for rotor in cfg.rotors:
            self._add_rotor(model, link, rotor)

        # Plugins
        self._add_mavlink_plugin(model, cfg)

        return sdf

    def write(self, sdf_element: ET.Element, path: str) -> Path:
        """Write the SDF element to a file with pretty-printing."""
        raw  = ET.tostring(sdf_element, encoding="unicode")
        dom  = minidom.parseString(raw)
        pretty = dom.toprettyxml(indent="  ")
        # Remove the extra XML declaration added by toprettyxml
        lines  = pretty.split("\n")
        if lines[0].startswith("<?xml"):
            pretty = "\n".join(lines[1:])

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f'<?xml version="1.0"?>\n{pretty}')
        logger.info("SDFBuilder: wrote %s", out)
        return out

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _text(self, parent: ET.Element, tag: str, text: str) -> ET.Element:
        el = ET.SubElement(parent, tag)
        el.text = text
        return el

    def _make_plane_normal(self) -> ET.Element:
        normal = ET.Element("normal")
        normal.text = "0 0 1"
        return normal

    def _embed_drone(
        self, world: ET.Element, cfg: DroneModelConfig,
        pos: tuple, instance: int
    ) -> None:
        """Embed a drone model include in the world (references model:// URI)."""
        inc = ET.SubElement(world, "include")
        self._text(inc, "uri", f"model://{cfg.name}")
        self._text(inc, "name", f"{cfg.name}_{instance}")
        self._text(inc, "pose", f"{pos[0]} {pos[1]} {pos[2]} 0 0 0")

    def _add_imu(self, link: ET.Element, cfg: ImuSensorConfig) -> None:
        sensor = ET.SubElement(link, "sensor", name="imu_sensor", type="imu")
        self._text(sensor, "always_on", "1")
        self._text(sensor, "update_rate", str(cfg.update_rate))
        imu  = ET.SubElement(sensor, "imu")
        accel = ET.SubElement(imu, "angular_velocity")
        for axis in ("x", "y", "z"):
            a = ET.SubElement(accel, axis)
            self._text(a, "noise", "")
            noise = ET.SubElement(a, "noise", type="gaussian")
            self._text(noise, "mean", "0")
            self._text(noise, "stddev", str(cfg.gyro_noise_density))

    def _add_gps(self, link: ET.Element, cfg: GpsSensorConfig) -> None:
        sensor = ET.SubElement(link, "sensor", name="gps_sensor", type="gps")
        self._text(sensor, "always_on", "1")
        self._text(sensor, "update_rate", str(cfg.update_rate))
        gps = ET.SubElement(sensor, "gps")
        pos = ET.SubElement(gps, "position_sensing")
        for axis in ("horizontal", "vertical"):
            n = ET.SubElement(pos, axis)
            self._text(n, "noise",
                       str(cfg.position_noise_m))

    def _add_rotor(
        self, model: ET.Element, base_link: ET.Element,
        rotor: RotorConfig
    ) -> None:
        """Add a rotor joint and link to the model."""
        rotor_name = f"rotor_{rotor.index}"
        link = ET.SubElement(model, "link", name=rotor_name)
        self._text(link, "pose",
                   f"{rotor.x} {rotor.y} {rotor.z} 0 0 0")
        inertial = ET.SubElement(link, "inertial")
        self._text(inertial, "mass", "0.005")

        # Joint connecting rotor to base
        joint = ET.SubElement(model, "joint",
                              name=f"rotor_{rotor.index}_joint", type="revolute")
        self._text(joint, "parent", "base_link")
        self._text(joint, "child",  rotor_name)
        axis = ET.SubElement(joint, "axis")
        self._text(axis, "xyz", "0 0 1")
        limit = ET.SubElement(axis, "limit")
        self._text(limit, "lower", "-1e16")
        self._text(limit, "upper", "1e16")

    def _add_mavlink_plugin(
        self, model: ET.Element, cfg: DroneModelConfig
    ) -> None:
        """Add gz-MAVLink plugin configuration."""
        plugin = ET.SubElement(model, "plugin",
                               name="gz_mavlink_plugin",
                               filename="gz-sim-multicopter-motor-model-system")
        self._text(plugin, "robotNamespace", cfg.name)
        self._text(plugin, "commandSubTopic", "command/motor_speed")
        self._text(plugin, "motorSpeedPubTopic", "motor_speed/0")


# ─────────────────────────────────────────────────────────────────────────────
#  GazeboSource — TelemetrySource for live Gazebo state
# ─────────────────────────────────────────────────────────────────────────────

class GazeboSource(TelemetrySource):
    """TelemetrySource that subscribes to Gazebo model pose via gz-transport.

    Reads /world/{world_name}/pose/info topic which publishes the pose of all
    models at the physics rate.

    Requires: gz-transport Python bindings (gz-transport13 for Harmonic).

    Args:
        source_id:    Unique identifier.
        vehicle_id:   Target StateStore vehicle ID.
        world_name:   Gazebo world name (default "drone_world").
        model_name:   Gazebo model name to track.
        gz_partition: gz-transport partition (default from GZ_PARTITION env).
    """

    def __init__(
        self,
        source_id:    str,
        vehicle_id:   str,
        world_name:   str = "drone_world",
        model_name:   str = "iris_0",
        gz_partition: str = "",
    ) -> None:
        super().__init__(
            source_id, vehicle_id,
            SourcePriority.SIMULATION,
            DataSource.SITL,
        )
        self._world     = world_name
        self._model     = model_name
        self._partition = gz_partition
        self._node      = None
        self._sub       = None
        self._latest_update: Optional[DroneStateUpdate] = None
        self._update_lock = threading.Lock()

    def connect(self) -> None:
        """Subscribe to Gazebo pose topic."""
        try:
            import gz.transport13 as gz_transport
        except ImportError as exc:
            raise ImportError(
                "gz-transport13 not installed. "
                "Install: pip install gz-transport13\n"
                "Or build from source: https://gazebosim.org/docs"
            ) from exc

        topic = f"/world/{self._world}/pose/info"
        self._node = gz_transport.Node()
        self._sub  = self._node.subscribe(
            topic, self._on_pose_message
        )
        with self._lock:
            self._status = SourceStatus.CONNECTED
        logger.info("GazeboSource[%s]: subscribed to %s", self._source_id, topic)

    def poll(self) -> List[DroneStateUpdate]:
        with self._lock:
            if self._status != SourceStatus.CONNECTED:
                return []

        with self._update_lock:
            upd = self._latest_update
            self._latest_update = None

        if upd is not None:
            self._record_poll(1)
            return [upd]
        return []

    def disconnect(self) -> None:
        self._node = None
        self._sub  = None
        with self._lock:
            self._status = SourceStatus.DISCONNECTED

    def _on_pose_message(self, msg: Any) -> None:
        """Callback from gz-transport pose subscription."""
        import time as _time
        import numpy as np

        # gz Pose_V message contains a list of poses (one per model)
        for pose in getattr(msg, "pose", []):
            if pose.name != self._model:
                continue

            upd = DroneStateUpdate(
                vehicle_id     = self._vehicle_id,
                source         = DataSource.SITL,
                timestamp_wall = _time.time(),
                timestamp_mono = _time.monotonic(),
            )

            # Position: Gazebo uses ENU → convert to NED
            p = pose.position
            upd.position = np.array([p.y, p.x, -p.z])  # ENU→NED

            # Orientation: Gazebo quaternion (w, x, y, z) → our Hamilton NED
            q = pose.orientation
            # Apply ENU→NED rotation
            upd.quaternion = np.array([q.w, q.x, q.y, q.z])

            with self._update_lock:
                self._latest_update = upd
            break


# ─────────────────────────────────────────────────────────────────────────────
#  WorldManager — runtime model management
# ─────────────────────────────────────────────────────────────────────────────

class WorldManager:
    """Manages Gazebo world at runtime via gz service calls.

    Provides: spawn model, remove model, set model pose, pause/resume.
    Requires: gz CLI tool on PATH.
    """

    def __init__(self, world_name: str = "drone_world") -> None:
        self._world = world_name

    def spawn_model(
        self,
        model_sdf_path: str,
        model_name:     str,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.3,
    ) -> bool:
        """Spawn a model from an SDF file at the given position."""
        cmd = [
            "gz", "model", "--spawn-file", model_sdf_path,
            "--model-name", model_name,
            "-x", str(x), "-y", str(y), "-z", str(z),
        ]
        return self._run_gz(cmd)

    def remove_model(self, model_name: str) -> bool:
        """Remove a model from the running world."""
        cmd = ["gz", "model", "--delete", "--model-name", model_name]
        return self._run_gz(cmd)

    def set_pose(
        self, model_name: str,
        x: float, y: float, z: float,
        roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
    ) -> bool:
        """Teleport a model to a new pose (ENU convention for Gazebo)."""
        cmd = [
            "gz", "model", "--model-name", model_name,
            "-x", str(y), "-y", str(x), "-z", str(-z),  # NED→ENU
            "-R", str(roll), "-P", str(pitch), "-Y", str(yaw),
        ]
        return self._run_gz(cmd)

    def pause(self) -> bool:
        """Pause the simulation."""
        return self._run_gz(["gz", "sim", "--pause"])

    def resume(self) -> bool:
        """Resume the simulation."""
        return self._run_gz(["gz", "sim", "--play"])

    def _run_gz(self, cmd: List[str]) -> bool:
        """Run a gz CLI command.  Returns True on success."""
        if not self._gz_available():
            logger.warning("WorldManager: gz CLI not found on PATH")
            return False
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10.0
            )
            if result.returncode != 0:
                logger.warning(
                    "WorldManager: gz command failed: %s\n%s",
                    " ".join(cmd), result.stderr,
                )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.error("WorldManager: gz command timed out: %s", cmd)
            return False
        except Exception as exc:  # noqa: BLE001
            logger.exception("WorldManager: gz command error: %s", exc)
            return False

    @staticmethod
    def _gz_available() -> bool:
        import shutil
        return shutil.which("gz") is not None
