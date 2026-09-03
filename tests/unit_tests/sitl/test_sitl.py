"""
Tests for drone_sdk.sitl (Module 6).

All process-spawning tests use subprocess.Popen mocks so no PX4 installation
or Docker daemon is required.  Config and port arithmetic are tested purely.
"""
from __future__ import annotations

import subprocess
import time
from io import StringIO
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List

import pytest

from drone_sdk.sitl import (
    FleetConfig,
    HomePosition,
    LaunchMode,
    LaunchResult,
    ProcessState,
    SimBackend,
    SITLConfig,
    SITLLauncher,
    SITLPortConfig,
    SwarmLauncher,
    VehicleType,
    WorldType,
    PX4_READY_MARKER,
)


# ══════════════════════════════════════════════════════════════════════════════
#  SITLPortConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestSITLPortConfig:

    def test_instance_0_default_ports(self):
        cfg = SITLPortConfig.for_instance(0)
        assert cfg.gcs_port == 14550
        assert cfg.api_port == 14540
        assert cfg.sitl_port == 4560

    def test_instance_1_ports(self):
        cfg = SITLPortConfig.for_instance(1)
        assert cfg.gcs_port == 14560
        assert cfg.api_port == 14550
        assert cfg.sitl_port == 4561

    def test_instance_2_ports(self):
        cfg = SITLPortConfig.for_instance(2)
        assert cfg.gcs_port == 14570
        assert cfg.api_port == 14560
        assert cfg.sitl_port == 4562

    def test_connection_string_format(self):
        cfg = SITLPortConfig.for_instance(0)
        cs  = cfg.connection_string
        assert cs.startswith("udp:")
        assert "127.0.0.1" in cs
        assert "14540" in cs

    def test_no_port_collision_for_10_instances(self):
        """All 10 instances must have unique API and GCS ports."""
        api_ports = [SITLPortConfig.for_instance(i).api_port for i in range(10)]
        gcs_ports = [SITLPortConfig.for_instance(i).gcs_port for i in range(10)]
        assert len(set(api_ports)) == 10
        assert len(set(gcs_ports)) == 10


# ══════════════════════════════════════════════════════════════════════════════
#  HomePosition
# ══════════════════════════════════════════════════════════════════════════════

class TestHomePosition:

    def test_px4_default_is_zurich(self):
        home = HomePosition.px4_default()
        assert abs(home.lat - 47.397742) < 0.001
        assert abs(home.lon - 8.545594)  < 0.001

    def test_bengaluru_hal(self):
        home = HomePosition.bengaluru_hal()
        assert abs(home.lat - 12.96) < 0.1
        assert abs(home.lon - 77.42) < 0.1

    def test_isro_shar(self):
        home = HomePosition.isro_shar()
        # Sriharikota: ~13.72°N, 80.23°E
        assert abs(home.lat - 13.7199) < 0.01
        assert abs(home.lon - 80.2304) < 0.01

    def test_to_env_string_format(self):
        home = HomePosition(lat=12.96, lon=77.42, alt=900.0, heading=0.0)
        s = home.to_env_string()
        assert "12.96" in s
        assert "77.42" in s
        assert "900.0" in s


# ══════════════════════════════════════════════════════════════════════════════
#  SITLConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestSITLConfig:

    def test_default_vehicle_is_iris(self):
        cfg = SITLConfig()
        assert cfg.vehicle == VehicleType.IRIS

    def test_connection_string_uses_instance_port(self):
        cfg = SITLConfig(instance=2)
        assert "14560" in cfg.connection_string   # api_port for instance 2

    def test_gcs_port(self):
        cfg = SITLConfig(instance=0)
        assert cfg.gcs_port == 14550

    def test_api_port(self):
        cfg = SITLConfig(instance=0)
        assert cfg.api_port == 14540

    def test_env_vars_include_home(self):
        home = HomePosition(lat=13.0, lon=77.5, alt=920.0, heading=90.0)
        cfg  = SITLConfig(instance=0, home=home)
        env  = cfg.to_env()
        assert env["PX4_HOME_LAT"] == "13.0"
        assert env["PX4_HOME_LON"] == "77.5"
        assert env["PX4_HOME_ALT"] == "920.0"

    def test_env_vars_include_instance(self):
        cfg = SITLConfig(instance=3)
        env = cfg.to_env()
        assert env["PX4_INSTANCE"] == "3"
        assert env["MAV_SYS_ID"]   == "4"   # 1-based

    def test_headless_sets_env(self):
        cfg = SITLConfig(headless=True)
        assert cfg.to_env()["HEADLESS"] == "1"

    def test_not_headless_no_env(self):
        cfg = SITLConfig(headless=False)
        assert "HEADLESS" not in cfg.to_env()

    def test_extra_env_merged(self):
        cfg = SITLConfig(extra_env={"MY_VAR": "hello"})
        assert cfg.to_env()["MY_VAR"] == "hello"

    def test_to_docker_env_args(self):
        cfg  = SITLConfig(instance=0)
        args = cfg.to_docker_env_args()
        assert any("PX4_HOME_LAT" in a for a in args)

    def test_to_px4_command_includes_vehicle(self):
        cfg = SITLConfig(vehicle=VehicleType.HEXAROTOR)
        cmd = cfg.to_px4_command()
        assert VehicleType.HEXAROTOR.value in cmd

    def test_repr(self):
        cfg = SITLConfig(instance=1)
        r   = repr(cfg)
        assert "instance=1" in r
        assert "iris" in r

    def test_different_vehicles(self):
        for vt in VehicleType:
            cfg = SITLConfig(vehicle=vt)
            assert cfg.vehicle == vt

    def test_different_backends(self):
        for sb in SimBackend:
            cfg = SITLConfig(backend=sb)
            assert cfg.backend == sb


# ══════════════════════════════════════════════════════════════════════════════
#  FleetConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestFleetConfig:

    def test_generates_correct_n_configs(self):
        fleet = FleetConfig(n_vehicles=5)
        assert len(fleet.configs) == 5

    def test_instances_sequential(self):
        fleet = FleetConfig(n_vehicles=4)
        instances = [c.instance for c in fleet.configs]
        assert instances == [0, 1, 2, 3]

    def test_no_port_collisions(self):
        fleet = FleetConfig(n_vehicles=6)
        api_ports = [c.api_port for c in fleet.configs]
        assert len(set(api_ports)) == 6, "API port collision detected"

    def test_vehicles_offset_position(self):
        """Each drone should have a slightly different home latitude."""
        fleet = FleetConfig(n_vehicles=3, spacing_m=10.0)
        lats  = [c.home.lat for c in fleet.configs]
        assert lats[0] < lats[1] < lats[2]

    def test_connection_strings(self):
        fleet = FleetConfig(n_vehicles=3)
        cs = fleet.connection_strings
        assert len(cs) == 3
        for s in cs:
            assert s.startswith("udp:")

    def test_vehicle_ids(self):
        fleet = FleetConfig(n_vehicles=3)
        ids = fleet.vehicle_ids
        assert ids == ["drone_0", "drone_1", "drone_2"]

    def test_single_vehicle_fleet(self):
        fleet = FleetConfig(n_vehicles=1)
        assert len(fleet.configs) == 1
        assert fleet.configs[0].instance == 0

    def test_bengaluru_home(self):
        fleet = FleetConfig(
            n_vehicles=2,
            home=HomePosition.bengaluru_hal(),
        )
        assert abs(fleet.configs[0].home.lat - 12.96) < 0.1


# ══════════════════════════════════════════════════════════════════════════════
#  SITLLauncher (mocked subprocess)
# ══════════════════════════════════════════════════════════════════════════════

def _make_mock_process(ready: bool = True, exit_code: int = None):
    """Create a mock subprocess.Popen that simulates PX4 SITL output."""
    proc = MagicMock()
    proc.pid = 12345
    proc.returncode = exit_code

    if ready:
        # Stream simulates PX4 boot sequence then ready marker
        lines = [
            "INFO  [px4] startup script: /bin/sh etc/init.d-posix/rcS\n",
            "INFO  [simulator] Simulator listening on TCP port 4560\n",
            f"{PX4_READY_MARKER}\n",
            "",   # EOF sentinel
        ]
    else:
        lines = [
            "ERROR [px4] Failed to start\n",
            "",
        ]

    line_iter = iter(lines)
    proc.stdout.readline.side_effect = lambda: next(line_iter, "")
    proc.poll.return_value = exit_code   # None = running, int = exited

    return proc


class TestSITLLauncherLifecycle:

    def test_initial_state_idle(self):
        launcher = SITLLauncher(SITLConfig(instance=0))
        assert launcher.state == ProcessState.IDLE
        assert not launcher.is_running

    def test_connection_string_property(self):
        launcher = SITLLauncher(SITLConfig(instance=0))
        assert "14540" in launcher.connection_string

    def test_start_docker_no_docker_returns_failure(self):
        """If Docker is not on PATH, start() returns failure immediately."""
        import shutil
        launcher = SITLLauncher(SITLConfig(), mode=LaunchMode.DOCKER)

        with patch("shutil.which", return_value=None):
            result = launcher.start(timeout_s=5.0)

        assert result.success is False
        assert "Docker" in result.error_message

    def test_start_native_missing_src_returns_failure(self):
        """Missing PX4 source dir → failure with helpful message."""
        launcher = SITLLauncher(
            SITLConfig(), mode=LaunchMode.NATIVE, px4_src_dir="/nonexistent/px4"
        )
        result = launcher.start(timeout_s=5.0)
        assert result.success is False
        assert "PX4 source" in result.error_message

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")   # Mock the `docker rm -f` cleanup call
    def test_successful_docker_start(self, mock_run, mock_which, mock_popen):
        """Mock a successful Docker PX4 SITL startup."""
        mock_proc = _make_mock_process(ready=True)
        mock_popen.return_value = mock_proc

        launcher = SITLLauncher(SITLConfig(instance=0), mode=LaunchMode.DOCKER)
        result   = launcher.start(timeout_s=10.0)

        assert result.success is True
        assert result.instance == 0
        assert "14540" in result.connection_str
        assert launcher.state == ProcessState.RUNNING
        assert launcher.is_running

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")
    def test_failed_docker_start_ready_marker_not_found(
        self, mock_run, mock_which, mock_popen
    ):
        """Process produces no ready marker within timeout → failure."""
        proc = MagicMock()
        proc.pid = 9999
        proc.returncode = None
        proc.poll.return_value = None
        # Never produce the ready marker
        proc.stdout.readline.return_value = "INFO  [px4] Still booting...\n"

        mock_popen.return_value = proc

        launcher = SITLLauncher(SITLConfig(), mode=LaunchMode.DOCKER)
        result   = launcher.start(timeout_s=0.1)   # Very short timeout

        assert result.success is False
        assert "Timeout" in result.error_message

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")
    def test_process_exits_before_ready(
        self, mock_run, mock_which, mock_popen
    ):
        """Process exits with non-zero code before ready marker → failure."""
        proc = MagicMock()
        proc.pid = 8888
        proc.returncode = 1
        proc.poll.return_value = 1   # Already exited
        proc.stdout.readline.return_value = ""
        proc.stdout.read.return_value = "Fatal error"
        mock_popen.return_value = proc

        launcher = SITLLauncher(SITLConfig(), mode=LaunchMode.DOCKER)
        result   = launcher.start(timeout_s=5.0)

        assert result.success is False
        assert result.startup_time_s >= 0

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")
    def test_stop_terminates_process(
        self, mock_run, mock_which, mock_popen
    ):
        """stop() calls process.terminate() and waits."""
        mock_proc = _make_mock_process(ready=True)
        mock_proc.wait = MagicMock(return_value=0)
        mock_popen.return_value = mock_proc

        launcher = SITLLauncher(SITLConfig(), mode=LaunchMode.DOCKER)
        launcher.start(timeout_s=5.0)
        stopped = launcher.stop(timeout_s=3.0)

        mock_proc.terminate.assert_called_once()
        assert launcher.state == ProcessState.STOPPED

    def test_stop_when_not_started(self):
        """stop() when never started should not raise."""
        launcher = SITLLauncher(SITLConfig())
        result = launcher.stop()
        assert result is True   # Nothing to stop

    def test_get_stdout_tail_empty(self):
        launcher = SITLLauncher(SITLConfig())
        assert launcher.get_stdout_tail() == []

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")
    def test_get_mavlink_source(
        self, mock_run, mock_which, mock_popen
    ):
        """get_mavlink_source returns a properly configured MAVLinkSource."""
        mock_popen.return_value = _make_mock_process(ready=True)
        launcher = SITLLauncher(SITLConfig(instance=0), mode=LaunchMode.DOCKER)
        launcher.start(timeout_s=5.0)

        source = launcher.get_mavlink_source("drone_0")
        assert source.vehicle_id == "drone_0"
        assert source.source_id  == "sitl_0"
        assert "14540" in source._connection_str

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")
    def test_launch_result_has_startup_time(
        self, mock_run, mock_which, mock_popen
    ):
        mock_popen.return_value = _make_mock_process(ready=True)
        launcher = SITLLauncher(SITLConfig(), mode=LaunchMode.DOCKER)
        result   = launcher.start(timeout_s=5.0)
        assert result.startup_time_s >= 0


# ══════════════════════════════════════════════════════════════════════════════
#  SwarmLauncher
# ══════════════════════════════════════════════════════════════════════════════

class TestSwarmLauncher:

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")
    def test_starts_all_instances(
        self, mock_run, mock_which, mock_popen
    ):
        # Each Popen call gets a fresh mock process (iterator not shared)
        mock_popen.side_effect = lambda *a, **kw: _make_mock_process(ready=True)
        fleet  = FleetConfig(n_vehicles=3)
        swarm  = SwarmLauncher(fleet.configs, mode=LaunchMode.DOCKER, stagger_s=0.0)
        results = swarm.start(timeout_per_instance=5.0)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert swarm.all_running()

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")
    def test_stops_all_instances(
        self, mock_run, mock_which, mock_popen
    ):
        def fresh_proc(*a, **kw):
            p = _make_mock_process(ready=True)
            p.wait = MagicMock(return_value=0)
            return p
        mock_popen.side_effect = fresh_proc

        fleet = FleetConfig(n_vehicles=2)
        swarm = SwarmLauncher(fleet.configs, mode=LaunchMode.DOCKER, stagger_s=0.0)
        swarm.start(timeout_per_instance=5.0)
        swarm.stop()

        for launcher in swarm.launchers:
            assert launcher.state == ProcessState.STOPPED

    def test_status_keys(self):
        fleet = FleetConfig(n_vehicles=2)
        swarm = SwarmLauncher(fleet.configs)
        status = swarm.status()
        assert len(status) == 2
        for s in status:
            assert "instance"   in s
            assert "state"      in s
            assert "connection" in s
            assert "restarts"   in s

    def test_not_running_initially(self):
        fleet = FleetConfig(n_vehicles=2)
        swarm = SwarmLauncher(fleet.configs)
        assert not swarm.all_running()

    def test_failure_stops_further_launches(self):
        """If one instance fails, swarm aborts remaining launches."""
        configs = FleetConfig(n_vehicles=3).configs

        # All launchers will return failure (no Docker)
        with patch("shutil.which", return_value=None):
            swarm   = SwarmLauncher(configs, mode=LaunchMode.DOCKER, stagger_s=0.0)
            results = swarm.start(timeout_per_instance=2.0)

        # At least the first one failed
        assert any(not r.success for r in results)


# ══════════════════════════════════════════════════════════════════════════════
#  Integration: SITLConfig → MAVLinkSource connection string
# ══════════════════════════════════════════════════════════════════════════════

class TestSITLToMAVLinkIntegration:

    def test_config_connection_string_matches_mavlink_source(self):
        """Connection string from SITLConfig must match what MAVLinkSource receives."""
        cfg = SITLConfig(instance=0)
        launcher = SITLLauncher(cfg)
        source   = launcher.get_mavlink_source("drone_0")
        assert source._connection_str == cfg.connection_string

    def test_multi_instance_sources_have_unique_ports(self):
        """Each SITL instance must produce MAVLinkSources on different ports."""
        configs  = FleetConfig(n_vehicles=5).configs
        launchers = [SITLLauncher(c) for c in configs]
        sources  = [l.get_mavlink_source(f"drone_{i}") for i, l in enumerate(launchers)]
        conn_strs = [s._connection_str for s in sources]
        assert len(set(conn_strs)) == 5, "Duplicate connection strings!"
