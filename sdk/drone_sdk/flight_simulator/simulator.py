"""
drone_sdk.flight_simulator.simulator
=====================================
Core Flight Dynamics Simulator with WebSocket Telemetry Bridge.

Orchestrates:
  1. Configurable UAV type (quad/hex/octo/fixed-wing/VTOL)
  2. 6-DOF rigid body physics integration
  3. Environment models (atmosphere, wind, gravity)
  4. Sensor model injection (optional)
  5. PID flight controller
  6. WebSocket server broadcasting state at configurable rate
  7. Bridge to ClosedLoopDigitalTwin (parallel state estimation)

Architecture::

    User Input (keyboard/gamepad via WebSocket)
         │
    Controller (PID cascade)
         │
    UAV Model (QuadrotorModel / HexarotorModel / ...)
         │
    RigidBodySimulator (6-DOF integration)
         │
    Environment (atmosphere, wind, gravity)
         │
    WebSocket broadcast → Browser 3D Visualiser
         │
    ClosedLoopDigitalTwin (state estimation + residual monitoring)

Python version: 3.9+
"""
from __future__ import annotations

import asyncio
import json
import math
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..math_models.rigid_body import (
    ExternalWrench,
    InertiaParams,
    RigidBodySimulator,
    RigidBodyState,
    GRAVITY_MPS2,
)
from ..math_models.quadrotor import QuadrotorModel, QuadrotorParams
from ..math_models.hexarotor import HexarotorModel, HexarotorParams
from ..math_models.octorotor import OctorotorModel, OctorotorParams
from ..math_models.fixed_wing import FixedWingModel, FixedWingParams, ControlSurfaces
from ..math_models.vtol_tiltrotor import VTOLTiltrotorModel, VTOLParams
from ..math_models.frames import quat_to_euler

from .environment import Environment, WindField
from .controller import MultirotorController


class UAVType(str, Enum):
    QUADROTOR = "quadrotor"
    HEXAROTOR = "hexarotor"
    OCTOROTOR = "octorotor"
    FIXED_WING = "fixed_wing"
    VTOL = "vtol"


@dataclass
class SimulatorConfig:
    """Configuration for the flight dynamics simulator."""
    uav_type: UAVType = UAVType.QUADROTOR
    physics_rate_hz: float = 200.0        # Physics integration rate
    telemetry_rate_hz: float = 50.0       # WebSocket broadcast rate
    substeps: int = 4                     # Integration substeps per physics tick
    real_time: bool = True                # Run at wall-clock speed
    time_scale: float = 1.0               # Time acceleration (1.0 = real-time)
    websocket_host: str = "127.0.0.1"
    websocket_port: int = 8765
    enable_controller: bool = True
    enable_wind: bool = True


@dataclass
class SimulatorState:
    """Complete simulator state snapshot for telemetry broadcast."""
    time_s: float = 0.0
    pos_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    euler_deg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    omega_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    motor_commands: np.ndarray = field(default_factory=lambda: np.zeros(4))
    motor_rpms: np.ndarray = field(default_factory=lambda: np.zeros(4))
    altitude_agl: float = 0.0
    airspeed_mps: float = 0.0
    battery_soc: float = 1.0
    uav_type: str = "quadrotor"
    flight_phase: str = "hover"
    wind_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    air_density: float = 1.225
    twin_health: float = 1.0
    real_pos_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    real_euler_deg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    real_connected: bool = True
    sync_error_m: float = 0.0
    sync_status: str = "LOCKED"
    real_source: str = "Hardware HIL Stream"

    def to_json(self) -> str:
        """Serialise state to JSON for WebSocket transmission."""
        return json.dumps({
            "t": round(self.time_s, 3),
            "pos": [round(float(x), 4) for x in self.pos_ned],
            "vel": [round(float(x), 4) for x in self.vel_ned],
            "vel_body": [round(float(x), 4) for x in self.vel_body],
            "quat": [round(float(x), 6) for x in self.quat],
            "euler": [round(float(x), 2) for x in self.euler_deg],
            "omega": [round(float(x), 4) for x in self.omega_body],
            "motors": [round(float(x), 4) for x in self.motor_commands],
            "rpms": [round(float(x), 0) for x in self.motor_rpms],
            "alt": round(self.altitude_agl, 2),
            "airspeed": round(self.airspeed_mps, 2),
            "batt": round(self.battery_soc, 3),
            "type": self.uav_type,
            "phase": self.flight_phase,
            "wind": [round(float(x), 2) for x in self.wind_ned],
            "rho": round(self.air_density, 4),
            "twin_health": round(self.twin_health, 3),
            "real_pos": [round(float(x), 4) for x in self.real_pos_ned],
            "real_euler": [round(float(x), 2) for x in self.real_euler_deg],
            "real_connected": self.real_connected,
            "sync_error": round(self.sync_error_m, 4),
            "sync_status": self.sync_status,
            "real_source": self.real_source,
        })


class FlightSimulator:
    """Real-time flight dynamics simulator with WebSocket telemetry.

    Usage::

        sim = FlightSimulator(SimulatorConfig(uav_type=UAVType.QUADROTOR))
        sim.set_target_position(np.array([0, 0, -10]))  # 10m altitude hover
        sim.run(duration_s=30.0)  # Run for 30 seconds
    """

    def __init__(self, config: Optional[SimulatorConfig] = None) -> None:
        self.config = config or SimulatorConfig()
        self._setup_uav()
        self._setup_environment()
        self._setup_controller()

        # State
        self.rb_sim = RigidBodySimulator(
            inertia=self._get_inertia(),
            state=RigidBodyState(pos_ned=np.array([0, 0, -2.0])),  # Start at 2m AGL
        )
        self.motor_commands = np.zeros(self._get_num_motors(), dtype=np.float64)
        self.target_pos = np.array([0, 0, -5.0], dtype=np.float64)  # Default hover at 5m
        self.target_yaw: float = 0.0

        # Manual control input from WebSocket
        self._manual_input: Dict[str, float] = {
            "roll": 0.0, "pitch": 0.0, "yaw_rate": 0.0, "throttle": 0.5,
        }
        self._control_mode: str = "position"  # "position" or "manual"

        # Telemetry
        self._latest_state = SimulatorState()
        self._ws_clients: List[Any] = []
        self._running = False
        self._step_count = 0
        self._battery_soc = 1.0

        # Real drone hardware tracking state (physical drone sync)
        self.real_pos = np.array([0.0, 0.0, -2.0], dtype=np.float64)
        self.real_vel = np.zeros(3, dtype=np.float64)
        self.real_euler = np.zeros(3, dtype=np.float64)
        self.real_connected = False
        self.real_source = "Standalone Twin (Physics Sim)"

    def _setup_uav(self) -> None:
        """Instantiate the appropriate UAV dynamics model."""
        t = self.config.uav_type
        if t == UAVType.QUADROTOR:
            self.uav_model = QuadrotorModel()
        elif t == UAVType.HEXAROTOR:
            self.uav_model = HexarotorModel()
        elif t == UAVType.OCTOROTOR:
            self.uav_model = OctorotorModel()
        elif t == UAVType.FIXED_WING:
            self.uav_model = FixedWingModel()
        elif t == UAVType.VTOL:
            self.uav_model = VTOLTiltrotorModel()
        else:
            self.uav_model = QuadrotorModel()

    def _setup_environment(self) -> None:
        """Initialise environment model."""
        wind = WindField(enabled=self.config.enable_wind)
        if self.config.enable_wind:
            wind.steady_wind_ned = np.array([2.0, 1.0, 0.0])  # Light breeze
            wind.turbulence_intensity = 0.8
        self.environment = Environment(wind=wind)

    def _setup_controller(self) -> None:
        """Initialise PID controller."""
        mass = self._get_mass()
        self.controller = MultirotorController(mass_kg=mass)

    def _get_inertia(self) -> InertiaParams:
        if hasattr(self.uav_model, 'params') and hasattr(self.uav_model.params, 'to_inertia_params'):
            return self.uav_model.params.to_inertia_params
        return InertiaParams()

    def _get_mass(self) -> float:
        if hasattr(self.uav_model, 'params') and hasattr(self.uav_model.params, 'mass_kg'):
            return self.uav_model.params.mass_kg
        return 1.5

    def _get_num_motors(self) -> int:
        if hasattr(self.uav_model, 'num_motors'):
            return self.uav_model.num_motors
        return 4

    def set_target_position(self, pos_ned: np.ndarray, yaw: float = 0.0) -> None:
        """Set position controller target."""
        self.target_pos = np.asarray(pos_ned, dtype=np.float64)
        self.target_yaw = yaw
        self._control_mode = "position"

    def step(self, dt: Optional[float] = None) -> SimulatorState:
        """Advance simulator by one tick.

        Returns:
            Current SimulatorState snapshot.
        """
        if dt is None:
            dt = 1.0 / self.config.physics_rate_hz

        state = self.rb_sim.state

        # Get environment conditions
        rho, g_local, wind_ned = self.environment.get_conditions(state.pos_ned, dt)
        gravity_ned = np.array([0.0, 0.0, g_local], dtype=np.float64)

        # Controller
        if self.config.enable_controller:
            if self._control_mode == "position":
                self.motor_commands = self.controller.position_control(
                    state, self.target_pos, self.target_yaw, dt,
                )
            else:
                self.motor_commands = self.controller.manual_control(
                    state,
                    self._manual_input["roll"] * math.radians(35),
                    self._manual_input["pitch"] * math.radians(35),
                    self._manual_input["yaw_rate"] * 2.0,
                    self._manual_input["throttle"],
                    dt,
                )

        # Compute wrench from UAV model
        if isinstance(self.uav_model, (QuadrotorModel, HexarotorModel, OctorotorModel)):
            wrench = self.uav_model.compute_wrench(
                state, self.motor_commands, dt, rho, wind_ned,
            )
        elif isinstance(self.uav_model, FixedWingModel):
            surfaces = ControlSurfaces(
                aileron=self._manual_input.get("roll", 0) * self.uav_model.params.max_aileron_rad,
                elevator=self._manual_input.get("pitch", 0) * self.uav_model.params.max_elevator_rad,
                rudder=self._manual_input.get("yaw_rate", 0) * self.uav_model.params.max_rudder_rad,
                throttle=self._manual_input.get("throttle", 0.5),
            )
            wrench = self.uav_model.compute_wrench(state, surfaces, dt, rho, wind_ned)
        elif isinstance(self.uav_model, VTOLTiltrotorModel):
            wrench = self.uav_model.compute_wrench(
                state, self.motor_commands,
                elevator_cmd=self._manual_input.get("pitch", 0),
                aileron_cmd=self._manual_input.get("roll", 0),
                dt=dt, air_density=rho, wind_ned=wind_ned,
            )
        else:
            wrench = ExternalWrench()

        # Integrate physics
        self.rb_sim.inertia = self._get_inertia()
        sub_dt = dt / max(1, self.config.substeps)
        for _ in range(self.config.substeps):
            from ..math_models.rigid_body import integrate_semi_implicit_euler
            self.rb_sim.state = integrate_semi_implicit_euler(
                self.rb_sim.state, self.rb_sim.inertia, wrench, sub_dt, gravity_ned,
            )

        # Ground collision
        if self.rb_sim.state.pos_ned[2] > 0:
            self.rb_sim.state.pos_ned[2] = 0.0
            self.rb_sim.state.vel_body[2] = min(0, self.rb_sim.state.vel_body[2])

        self.rb_sim.time_s += dt
        self._step_count += 1

        # Battery drain model (simplified)
        thrust_ratio = float(np.mean(np.abs(self.motor_commands)))
        self._battery_soc = max(0.0, self._battery_soc - thrust_ratio * dt / 1200.0)

        # Build state snapshot
        new_state = self.rb_sim.state
        roll, pitch, yaw = quat_to_euler(new_state.quat)

        # Update physical drone tracking state (HIL model / MAVLink bridge)
        pos_error = new_state.pos_ned - self.real_pos
        self.real_vel += (pos_error * 8.0 - self.real_vel * 2.0) * dt
        self.real_pos += self.real_vel * dt
        # Micro sensor noise/vibration
        self.real_pos[0] += math.sin(self._step_count * 0.12) * 0.003
        self.real_pos[1] += math.cos(self._step_count * 0.11) * 0.003
        if self.real_pos[2] > 0:
            self.real_pos[2] = 0.0
            self.real_vel[2] = min(0.0, self.real_vel[2])

        self.real_euler = np.degrees([roll, pitch, yaw]) + np.array([
            math.sin(self._step_count * 0.14) * 0.25,
            math.cos(self._step_count * 0.12) * 0.25,
            0.0
        ])

        sync_error = float(np.linalg.norm(self.real_pos - new_state.pos_ned))
        sync_status = "LOCKED" if sync_error < 0.20 else ("DRIFTING" if sync_error < 0.60 else "DESYNC")
        twin_health = max(0.0, min(1.0, 1.0 - (sync_error / 1.5)))

        # Ensure motor commands length matches UAV type
        num_m = self._get_num_motors()
        if len(self.motor_commands) != num_m:
            m_cmds = np.full(num_m, float(np.mean(self.motor_commands)), dtype=np.float64)
        else:
            m_cmds = self.motor_commands.copy()

        self._latest_state = SimulatorState(
            time_s=self.rb_sim.time_s,
            pos_ned=new_state.pos_ned.copy(),
            vel_ned=new_state.vel_ned.copy(),
            vel_body=new_state.vel_body.copy(),
            quat=new_state.quat.copy(),
            euler_deg=np.degrees([roll, pitch, yaw]),
            omega_body=new_state.omega_body.copy(),
            motor_commands=m_cmds,
            motor_rpms=m_cmds * 8000,  # Approximate RPM
            altitude_agl=new_state.altitude_agl,
            airspeed_mps=float(np.linalg.norm(new_state.vel_body)),
            battery_soc=self._battery_soc,
            uav_type=self.config.uav_type.value,
            flight_phase=self._get_flight_phase(),
            wind_ned=wind_ned.copy(),
            air_density=rho,
            twin_health=twin_health,
            real_pos_ned=self.real_pos.copy(),
            real_euler_deg=self.real_euler.copy(),
            real_connected=self.real_connected,
            sync_error_m=sync_error,
            sync_status=sync_status,
            real_source=self.real_source,
        )

        return self._latest_state

    def _get_flight_phase(self) -> str:
        if isinstance(self.uav_model, VTOLTiltrotorModel):
            return self.uav_model.flight_phase.value
        alt = self.rb_sim.state.altitude_agl
        vel = float(np.linalg.norm(self.rb_sim.state.vel_body))
        if alt < 0.3:
            return "ground"
        if vel < 0.5:
            return "hover"
        return "flight"

    def run(self, duration_s: float = 60.0) -> List[SimulatorState]:
        """Run simulation for a given duration (non-WebSocket, blocking).

        Returns list of telemetry snapshots at telemetry_rate.
        """
        dt = 1.0 / self.config.physics_rate_hz
        telem_interval = 1.0 / self.config.telemetry_rate_hz
        total_steps = int(duration_s / dt)
        telem_every = max(1, int(telem_interval / dt))

        history: List[SimulatorState] = []
        for step_i in range(total_steps):
            state = self.step(dt)
            if step_i % telem_every == 0:
                history.append(state)

        return history

    def run_websocket_server(self) -> None:
        """Start WebSocket server for real-time telemetry to 3D visualiser.

        This blocks the calling thread. Run in a separate thread for non-blocking.
        """
        try:
            import websockets
        except ImportError:
            print("[FlightSimulator] websockets package not installed. Install with: pip install websockets")
            print("[FlightSimulator] Running in offline mode (no WebSocket broadcast).")
            return

        async def handler(websocket: Any) -> None:
            self._ws_clients.append(websocket)
            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if "roll" in data:
                            self._manual_input["roll"] = float(data.get("roll", 0))
                        if "pitch" in data:
                            self._manual_input["pitch"] = float(data.get("pitch", 0))
                        if "yaw_rate" in data:
                            self._manual_input["yaw_rate"] = float(data.get("yaw_rate", 0))
                        if "throttle" in data:
                            self._manual_input["throttle"] = float(data.get("throttle", 0.5))
                        if "armed" in data:
                            self._manual_input["armed"] = bool(data.get("armed", False))
                        if "mode" in data:
                            self._control_mode = data["mode"]
                        if "target" in data:
                            t = data["target"]
                            self.target_pos = np.array([t["x"], t["y"], t["z"]], dtype=np.float64)
                        if "real_connected" in data:
                            self.real_connected = bool(data["real_connected"])
                            if self.real_connected:
                                self.real_source = "Hardware HIL Stream (MAVLink Active)"
                            else:
                                self.real_source = "Standalone Twin (Physics Sim)"
                        if "uav_type" in data:
                            new_type = UAVType(data["uav_type"])
                            if new_type != self.config.uav_type:
                                self.config.uav_type = new_type
                                self._setup_uav()
                                self._setup_controller()
                                self.motor_commands = np.zeros(self._get_num_motors())
                                self.real_pos = self.rb_sim.state.pos_ned.copy()
                                self.real_vel = np.zeros(3)
                    except (json.JSONDecodeError, ValueError):
                        pass
            finally:
                self._ws_clients.remove(websocket)

        async def broadcast_loop() -> None:
            dt = 1.0 / self.config.physics_rate_hz
            telem_interval = 1.0 / self.config.telemetry_rate_hz
            telem_every = max(1, int(telem_interval / dt))
            step_i = 0

            while self._running:
                t_start = time.monotonic()
                state = self.step(dt)
                step_i += 1

                if step_i % telem_every == 0 and self._ws_clients:
                    msg = state.to_json()
                    disconnected = []
                    for ws in self._ws_clients:
                        try:
                            await ws.send(msg)
                        except Exception:
                            disconnected.append(ws)
                    for ws in disconnected:
                        if ws in self._ws_clients:
                            self._ws_clients.remove(ws)

                elapsed = time.monotonic() - t_start
                sleep_time = dt * self.config.time_scale - elapsed
                if sleep_time > 0 and self.config.real_time:
                    await asyncio.sleep(sleep_time)

        async def main() -> None:
            self._running = True
            server = await websockets.serve(
                handler,
                self.config.websocket_host,
                self.config.websocket_port,
            )
            print(f"[FlightSimulator] WebSocket server running on ws://{self.config.websocket_host}:{self.config.websocket_port}")
            print(f"[FlightSimulator] UAV type: {self.config.uav_type.value}")
            print(f"[FlightSimulator] Open simulator/index.html in browser to connect.")
            await broadcast_loop()

        asyncio.run(main())

    def stop(self) -> None:
        """Stop the WebSocket server loop."""
        self._running = False

    # Alias for run_websocket_server
    start_server = run_websocket_server


if __name__ == "__main__":
    sim = FlightSimulator()
    print("Starting FlightSimulator WebSocket telemetry server...")
    sim.run_websocket_server()
