"""
sitl.config
===========
PX4 SITL configuration — vehicle types, simulation worlds, port allocation,
and environment setup.

Supported vehicle types
-----------------------
    iris            Standard quadrotor (X configuration)
    iris_opt_flow   Iris with optical flow sensor
    hexarotor       6-rotor hexacopter
    standard_vtol   Fixed-wing VTOL
    plane           Fixed-wing aircraft
    rover           Ground vehicle
    boat            Aquatic vehicle

Supported simulation backends
------------------------------
    none        PX4 jMAVSim (built-in, no external sim required)
    gazebo      Gazebo (requires ros-humble-gazebo-ros)
    gazebo-classic  Gazebo Classic 11

Port conventions
----------------
Multiple SITL instances use the base port + (instance_id * port_step):

    MAVLink GCS:  14550 + 10 * instance
    MAVLink onboard: 14540 + 10 * instance
    RTPS/DDS:     2019  + instance
    SITL UDP:     4560  + instance
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class VehicleType(str, Enum):
    """PX4 vehicle model identifiers."""
    IRIS            = "iris"
    IRIS_OPT_FLOW   = "iris_opt_flow"
    HEXAROTOR       = "hexarotor_6dof"
    STANDARD_VTOL   = "standard_vtol"
    TAILSITTER_VTOL = "tailsitter"
    PLANE           = "plane"
    ROVER           = "rover"
    BOAT            = "boat"


class SimBackend(str, Enum):
    """External simulation backend for PX4 SITL."""
    NONE            = "none"        # jMAVSim built-in
    GAZEBO          = "gz"          # Gazebo Harmonic (default in PX4 v1.15+)
    GAZEBO_CLASSIC  = "gazebo"      # Gazebo Classic 11
    JSBSIM          = "jsbsim"


class WorldType(str, Enum):
    """Pre-defined simulation worlds."""
    EMPTY       = "empty"
    BAYLANDS    = "baylands"
    MCMILLAN    = "mcmillan_airfield"
    SONOMA      = "sonoma_raceway"
    IRLOCK      = "irlock"


@dataclass
class SITLPortConfig:
    """MAVLink port configuration for one SITL instance.

    Port scheme (PX4 convention):
        GCS UDP in   (QGC connects here):  14550 + 10 * instance
        MAVLink API  (onboard/MAVSDK):     14540 + 10 * instance
        SITL UDP     (sim connection):      4560 + instance
    """
    instance:        int   = 0
    gcs_port:        int   = 14550   # Ground station (QGroundControl)
    api_port:        int   = 14540   # API (pymavlink / MAVSDK)
    sitl_port:       int   = 4560    # Internal sim connection

    @classmethod
    def for_instance(cls, instance: int) -> "SITLPortConfig":
        """Create port config for a specific SITL instance number."""
        return cls(
            instance  = instance,
            gcs_port  = 14550 + instance * 10,
            api_port  = 14540 + instance * 10,
            sitl_port = 4560  + instance,
        )

    @property
    def connection_string(self) -> str:
        """pymavlink connection string for the API port."""
        return f"udp:127.0.0.1:{self.api_port}"

    @property
    def qgc_connection_string(self) -> str:
        """QGroundControl connection string."""
        return f"udp:127.0.0.1:{self.gcs_port}"


@dataclass
class HomePosition:
    """GPS home position for SITL."""
    lat:     float = 47.397742   # PX4 default: Zurich, Switzerland
    lon:     float = 8.545594
    alt:     float = 488.0       # m MSL
    heading: float = 0.0         # degrees (true north)

    # Common locations for testing
    @classmethod
    def bengaluru_hal(cls) -> "HomePosition":
        """HAL Airport, Bengaluru, India."""
        return cls(lat=12.9600, lon=77.4173, alt=902.0, heading=0.0)

    @classmethod
    def px4_default(cls) -> "HomePosition":
        """PX4 default: Zurich, Switzerland."""
        return cls(lat=47.397742, lon=8.545594, alt=488.0, heading=0.0)

    @classmethod
    def isro_shar(cls) -> "HomePosition":
        """ISRO Satish Dhawan Space Centre, Sriharikota."""
        return cls(lat=13.7199, lon=80.2304, alt=10.0, heading=0.0)

    def to_env_string(self) -> str:
        """Format as PX4_HOME_LAT/LON/ALT environment variable string."""
        return (
            f"{self.lat:.6f},{self.lon:.6f},{self.alt:.1f},{self.heading:.1f}"
        )


@dataclass
class SITLConfig:
    """Complete configuration for one PX4 SITL instance.

    Args:
        instance:       Instance number (0–9 for multi-vehicle).
        vehicle:        PX4 vehicle model.
        backend:        External simulation backend.
        world:          Simulation world (backend-dependent).
        home:           GPS home position.
        headless:       Run without display (for CI/servers).
        baudrate:       MAVLink baudrate (for serial connections).
        extra_env:      Additional environment variables for PX4.

    Example (single drone)::

        cfg = SITLConfig(
            instance = 0,
            vehicle  = VehicleType.IRIS,
            home     = HomePosition.bengaluru_hal(),
        )

    Example (second drone in a swarm)::

        cfg = SITLConfig(
            instance = 1,
            vehicle  = VehicleType.IRIS,
        )
        # api_port = 14550, connection_str = "udp:127.0.0.1:14550"
    """
    instance:   int          = 0
    vehicle:    VehicleType  = VehicleType.IRIS
    backend:    SimBackend   = SimBackend.NONE
    world:      WorldType    = WorldType.EMPTY
    home:       HomePosition = field(default_factory=HomePosition.px4_default)
    headless:   bool         = True
    extra_env:  Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._ports = SITLPortConfig.for_instance(self.instance)

    @property
    def ports(self) -> SITLPortConfig:
        return self._ports

    @property
    def connection_string(self) -> str:
        """pymavlink connection string for this instance."""
        return self._ports.connection_string

    @property
    def gcs_port(self) -> int:
        return self._ports.gcs_port

    @property
    def api_port(self) -> int:
        return self._ports.api_port

    def to_env(self) -> Dict[str, str]:
        """Build environment variables dict for PX4 SITL process."""
        env: Dict[str, str] = {
            "PX4_SIM_HOST_ADDR":  "127.0.0.1",
            "PX4_HOME_LAT":       str(self.home.lat),
            "PX4_HOME_LON":       str(self.home.lon),
            "PX4_HOME_ALT":       str(self.home.alt),
            "PX4_HOME_HEADING":   str(self.home.heading),
            "PX4_INSTANCE":       str(self.instance),
            "MAV_SYS_ID":         str(self.instance + 1),   # 1-based sysid
        }
        if self.headless:
            env["HEADLESS"] = "1"
        env.update(self.extra_env)
        return env

    def to_docker_env_args(self) -> List[str]:
        """Format env vars as Docker --env flags."""
        return [f"--env {k}={v}" for k, v in self.to_env().items()]

    def to_px4_command(self, px4_src_dir: str = "/px4") -> str:
        """Build the PX4 SITL launch command string."""
        make_target = f"px4_sitl_{self.backend.value}_{self.vehicle.value}"
        return (
            f"cd {px4_src_dir} && "
            f"INSTANCE={self.instance} "
            f"make {make_target}"
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"SITLConfig(instance={self.instance}, "
            f"vehicle={self.vehicle.value}, "
            f"backend={self.backend.value}, "
            f"api_port={self.api_port})"
        )


# ── Multi-vehicle fleet configuration ────────────────────────────────────────

@dataclass
class FleetConfig:
    """Configuration for a multi-vehicle SITL fleet.

    Creates N instances with sequential port allocation and
    configurable formation start positions.

    Args:
        n_vehicles:   Number of drones.
        vehicle_type: Vehicle model for all drones.
        backend:      Simulation backend.
        home:         Home position of the first drone.
        spacing_m:    Spacing between drones in the X (North) direction, m.

    Example::

        fleet = FleetConfig(n_vehicles=3)
        for cfg in fleet.configs:
            print(cfg.connection_string)
            # udp:127.0.0.1:14540
            # udp:127.0.0.1:14550
            # udp:127.0.0.1:14560
    """
    n_vehicles:   int          = 3
    vehicle_type: VehicleType  = VehicleType.IRIS
    backend:      SimBackend   = SimBackend.NONE
    home:         HomePosition = field(default_factory=HomePosition.px4_default)
    spacing_m:    float        = 5.0

    @property
    def configs(self) -> List[SITLConfig]:
        """Return one SITLConfig per vehicle."""
        result = []
        for i in range(self.n_vehicles):
            # Offset each vehicle in the North (X) direction
            offset_lat = (self.spacing_m * i) / 111_111.0   # approx deg/m
            home_i = HomePosition(
                lat     = self.home.lat + offset_lat,
                lon     = self.home.lon,
                alt     = self.home.alt,
                heading = self.home.heading,
            )
            cfg = SITLConfig(
                instance  = i,
                vehicle   = self.vehicle_type,
                backend   = self.backend,
                home      = home_i,
                headless  = True,
            )
            result.append(cfg)
        return result

    @property
    def connection_strings(self) -> List[str]:
        return [cfg.connection_string for cfg in self.configs]

    @property
    def vehicle_ids(self) -> List[str]:
        return [f"drone_{i}" for i in range(self.n_vehicles)]
