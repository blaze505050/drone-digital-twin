"""
sitl.launcher
=============
SITLLauncher — manages PX4 SITL process lifecycle.

Two launch modes
----------------
1. **Docker mode** (default, recommended): Pulls the official PX4 Docker
   image and runs PX4 SITL in a container.  No native PX4 build required.
   Requires: Docker Engine running.

2. **Native mode**: Runs PX4 from a local source checkout.
   Requires: PX4-Autopilot source + build dependencies.

Workflow
--------
    launcher = SITLLauncher(config)
    launcher.start()             # Launches process, waits for MAVLink
    source = launcher.get_source()  # MAVLinkSource ready to use
    # … fly …
    launcher.stop()

Multi-vehicle
-------------
    SwarmLauncher([cfg0, cfg1, cfg2]).start()
    # Launches 3 PX4 instances with staggered startup

Health monitoring
-----------------
The launcher monitors the process health in a background thread and
attempts automatic restart on crash (configurable max_restarts).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

from .config import SITLConfig

logger = logging.getLogger(__name__)


class LaunchMode(Enum):
    DOCKER = auto()
    NATIVE = auto()


class ProcessState(Enum):
    IDLE       = auto()
    STARTING   = auto()
    RUNNING    = auto()
    CRASHED    = auto()
    STOPPED    = auto()


# ── PX4 Docker images ─────────────────────────────────────────────────────────

PX4_DOCKER_IMAGE = "px4io/px4-dev-simulation-focal:latest"

# Ready string in PX4 SITL stdout — indicates simulator is accepting MAVLink
PX4_READY_MARKER = "INFO  [mavlink] mode: Normal"
PX4_READY_TIMEOUT_S = 60.0


@dataclass
class LaunchResult:
    """Result of a SITL launch attempt."""
    success:         bool
    instance:        int
    pid:             Optional[int] = None
    container_id:    Optional[str] = None
    connection_str:  str           = ""
    error_message:   str           = ""
    startup_time_s:  float         = 0.0


class SITLLauncher:
    """Manages one PX4 SITL process (Docker or native).

    Args:
        config:       SITLConfig for this instance.
        mode:         DOCKER (default) or NATIVE.
        px4_src_dir:  Path to PX4-Autopilot source (NATIVE mode only).
        max_restarts: Auto-restart count on unexpected crash.
        on_crash:     Optional callback fired on process crash.

    Usage::

        launcher = SITLLauncher(SITLConfig(instance=0))
        result   = launcher.start()
        if result.success:
            source = launcher.get_mavlink_source("drone_0")
            source.connect()
    """

    def __init__(
        self,
        config:       SITLConfig,
        mode:         LaunchMode          = LaunchMode.DOCKER,
        px4_src_dir:  str                 = "/px4",
        max_restarts: int                 = 3,
        on_crash:     Optional[Callable]  = None,
    ) -> None:
        self._config       = config
        self._mode         = mode
        self._px4_src      = px4_src_dir
        self._max_restarts = max_restarts
        self._on_crash     = on_crash

        self._process: Optional[subprocess.Popen] = None
        self._state    = ProcessState.IDLE
        self._lock     = threading.RLock()
        self._n_restarts = 0
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._stdout_lines: List[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, timeout_s: float = PX4_READY_TIMEOUT_S) -> LaunchResult:
        """Launch the PX4 SITL process and wait for it to be ready.

        Args:
            timeout_s: Maximum time to wait for the MAVLink ready marker.

        Returns:
            LaunchResult with success=True if PX4 started and MAVLink is live.
        """
        with self._lock:
            if self._state == ProcessState.RUNNING:
                return LaunchResult(
                    success=True,
                    instance=self._config.instance,
                    connection_str=self._config.connection_string,
                )
            self._state = ProcessState.STARTING

        t0 = time.time()
        try:
            if self._mode == LaunchMode.DOCKER:
                result = self._start_docker(timeout_s)
            else:
                result = self._start_native(timeout_s)

            if result.success:
                with self._lock:
                    self._state = ProcessState.RUNNING
                self._start_monitor()
                logger.info(
                    "SITLLauncher: instance %d ready in %.1f s — %s",
                    self._config.instance,
                    time.time() - t0,
                    self._config.connection_string,
                )
            else:
                with self._lock:
                    self._state = ProcessState.CRASHED

            result.startup_time_s = time.time() - t0
            return result

        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._state = ProcessState.CRASHED
            logger.exception(
                "SITLLauncher: failed to start instance %d: %s",
                self._config.instance, exc,
            )
            return LaunchResult(
                success=False,
                instance=self._config.instance,
                error_message=str(exc),
                startup_time_s=time.time() - t0,
            )

    def stop(self, timeout_s: float = 10.0) -> bool:
        """Terminate the SITL process.

        Returns:
            True if process was successfully stopped.
        """
        self._stop_evt.set()

        with self._lock:
            process = self._process
            self._state = ProcessState.STOPPED

        if process:
            try:
                process.terminate()
                process.wait(timeout=timeout_s)
                logger.info(
                    "SITLLauncher: instance %d stopped", self._config.instance
                )
                return True
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(
                    "SITLLauncher: instance %d killed (timeout)", self._config.instance
                )
                return False
            except Exception:  # noqa: BLE001
                logger.exception(
                    "SITLLauncher: error stopping instance %d", self._config.instance
                )
                return False

        # Docker mode: kill container
        container_id = getattr(self, "_container_id", None)
        if container_id:
            try:
                subprocess.run(
                    ["docker", "stop", container_id],
                    timeout=timeout_s,
                    capture_output=True,
                )
                return True
            except Exception:  # noqa: BLE001
                return False

        return True

    def get_mavlink_source(self, vehicle_id: str):
        """Return a configured MAVLinkSource ready to connect.

        Returns:
            MAVLinkSource pointing at this SITL instance's API port.
        """
        from drone_sdk.mavlink_bridge.mavlink_source import MAVLinkSource
        from drone_sdk.telemetry_engine.source import SourcePriority

        return MAVLinkSource(
            source_id      = f"sitl_{self._config.instance}",
            vehicle_id     = vehicle_id,
            connection_str = self._config.connection_string,
            priority       = SourcePriority.SITL,
        )

    @property
    def state(self) -> ProcessState:
        with self._lock:
            return self._state

    @property
    def is_running(self) -> bool:
        return self.state == ProcessState.RUNNING

    @property
    def connection_string(self) -> str:
        return self._config.connection_string

    def get_stdout_tail(self, n: int = 20) -> List[str]:
        """Return the last N lines of PX4 stdout (for diagnostics)."""
        return self._stdout_lines[-n:]

    # ── Docker launcher ───────────────────────────────────────────────────────

    def _start_docker(self, timeout_s: float) -> LaunchResult:
        """Launch PX4 SITL in a Docker container."""
        if not shutil.which("docker"):
            return LaunchResult(
                success=False,
                instance=self._config.instance,
                error_message=(
                    "Docker not found. Install Docker Engine: "
                    "https://docs.docker.com/engine/install/"
                ),
            )

        env = self._config.to_env()
        env_args = []
        for k, v in env.items():
            env_args.extend(["-e", f"{k}={v}"])

        container_name = f"px4_sitl_{self._config.instance}"

        # Remove any stale container with same name
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
        )

        cmd = [
            "docker", "run",
            "--rm",
            "--name",    container_name,
            "--network", "host",          # Use host networking for UDP ports
            *env_args,
            PX4_DOCKER_IMAGE,
            "/bin/bash", "-c",
            (
                f"cd /src/PX4-Autopilot && "
                f"HEADLESS=1 make px4_sitl_default none_{self._config.vehicle.value} "
                f"2>&1"
            ),
        ]

        logger.info(
            "SITLLauncher: starting Docker container '%s'", container_name
        )

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._process = process
        self._container_id = container_name

        return self._wait_for_ready(process, timeout_s)

    # ── Native launcher ───────────────────────────────────────────────────────

    def _start_native(self, timeout_s: float) -> LaunchResult:
        """Launch PX4 SITL from a local source checkout."""
        if not os.path.isdir(self._px4_src):
            return LaunchResult(
                success=False,
                instance=self._config.instance,
                error_message=(
                    f"PX4 source directory not found: {self._px4_src}\n"
                    "Clone PX4-Autopilot: "
                    "git clone https://github.com/PX4/PX4-Autopilot.git --recursive"
                ),
            )

        env = {**os.environ, **self._config.to_env()}
        cmd = [
            "bash", "-c",
            (
                f"cd {self._px4_src} && "
                f"make px4_sitl_default none_{self._config.vehicle.value}"
            ),
        ]

        logger.info(
            "SITLLauncher[native]: starting instance %d vehicle=%s",
            self._config.instance, self._config.vehicle.value,
        )

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=self._px4_src,
            bufsize=1,
        )
        self._process = process
        return self._wait_for_ready(process, timeout_s)

    # ── Readiness detection ───────────────────────────────────────────────────

    def _wait_for_ready(
        self, process: subprocess.Popen, timeout_s: float
    ) -> LaunchResult:
        """Read stdout until PX4_READY_MARKER or timeout."""
        deadline = time.time() + timeout_s
        pid = process.pid

        while time.time() < deadline:
            if process.poll() is not None:
                # Process exited unexpectedly
                remaining = process.stdout.read() if process.stdout else ""
                return LaunchResult(
                    success=False,
                    instance=self._config.instance,
                    pid=pid,
                    error_message=(
                        f"PX4 process exited with code {process.returncode}. "
                        f"Last output: {remaining[-200:]}"
                    ),
                )

            if process.stdout:
                line = process.stdout.readline()
                if line:
                    line = line.rstrip()
                    self._stdout_lines.append(line)
                    # Keep only last 200 lines
                    if len(self._stdout_lines) > 200:
                        self._stdout_lines.pop(0)

                    if PX4_READY_MARKER in line:
                        return LaunchResult(
                            success=True,
                            instance=self._config.instance,
                            pid=pid,
                            connection_str=self._config.connection_string,
                        )
            else:
                time.sleep(0.1)

        return LaunchResult(
            success=False,
            instance=self._config.instance,
            pid=pid,
            error_message=(
                f"Timeout ({timeout_s}s) waiting for PX4 ready marker. "
                f"Last lines: {self._stdout_lines[-5:]}"
            ),
        )

    # ── Process monitor ───────────────────────────────────────────────────────

    def _start_monitor(self) -> None:
        """Start background thread to watch for process crashes."""
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name=f"SITLMonitor-{self._config.instance}",
            daemon=True,
        )
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        """Watch the SITL process and restart on crash if configured."""
        while not self._stop_evt.is_set():
            time.sleep(1.0)
            with self._lock:
                process = self._process
                if self._state != ProcessState.RUNNING:
                    break

            if process and process.poll() is not None:
                # Process has exited unexpectedly
                with self._lock:
                    self._state = ProcessState.CRASHED

                logger.error(
                    "SITLLauncher: instance %d CRASHED (exit=%d)",
                    self._config.instance, process.returncode,
                )

                if self._on_crash:
                    try:
                        self._on_crash(self._config.instance)
                    except Exception:  # noqa: BLE001
                        pass

                if self._n_restarts < self._max_restarts:
                    self._n_restarts += 1
                    logger.info(
                        "SITLLauncher: restarting instance %d (attempt %d/%d)",
                        self._config.instance, self._n_restarts, self._max_restarts,
                    )
                    time.sleep(2.0)
                    result = self.start(timeout_s=30.0)
                    if not result.success:
                        logger.error(
                            "SITLLauncher: restart failed for instance %d",
                            self._config.instance,
                        )
                        break
                else:
                    logger.error(
                        "SITLLauncher: max restarts (%d) reached for instance %d",
                        self._max_restarts, self._config.instance,
                    )
                    break


# ── SwarmLauncher ─────────────────────────────────────────────────────────────

class SwarmLauncher:
    """Manages multiple SITL instances for swarm simulation.

    Args:
        configs:      List of SITLConfig, one per drone.
        mode:         Launch mode (Docker or Native).
        stagger_s:    Seconds between successive instance launches.

    Usage::

        from drone_sdk.sitl import FleetConfig, SwarmLauncher, LaunchMode

        fleet   = FleetConfig(n_vehicles=3)
        swarm   = SwarmLauncher(fleet.configs, mode=LaunchMode.DOCKER)
        results = swarm.start()
        print(swarm.status())

        # Get sources for TelemetryEngine
        for launcher in swarm.launchers:
            engine.add_source(launcher.get_mavlink_source(vid))

        swarm.stop()
    """

    def __init__(
        self,
        configs:    List[SITLConfig],
        mode:       LaunchMode = LaunchMode.DOCKER,
        stagger_s:  float      = 2.0,
    ) -> None:
        self._launchers = [
            SITLLauncher(cfg, mode=mode) for cfg in configs
        ]
        self._stagger = stagger_s

    @property
    def launchers(self) -> List[SITLLauncher]:
        return self._launchers

    def start(self, timeout_per_instance: float = 60.0) -> List[LaunchResult]:
        """Start all instances with staggered launch.

        Returns:
            List of LaunchResult, one per instance.
        """
        results = []
        for launcher in self._launchers:
            result = launcher.start(timeout_s=timeout_per_instance)
            results.append(result)
            if not result.success:
                logger.error(
                    "SwarmLauncher: instance %d failed to start — stopping swarm",
                    launcher._config.instance,
                )
                break
            if self._stagger > 0:
                time.sleep(self._stagger)
        return results

    def stop(self) -> None:
        """Stop all instances."""
        for launcher in reversed(self._launchers):
            try:
                launcher.stop()
            except Exception:  # noqa: BLE001
                pass

    def status(self) -> List[Dict]:
        """Return status dict for each instance."""
        return [
            {
                "instance":    l._config.instance,
                "state":       l.state.name,
                "connection":  l.connection_string,
                "restarts":    l._n_restarts,
            }
            for l in self._launchers
        ]

    def all_running(self) -> bool:
        return all(l.is_running for l in self._launchers)
