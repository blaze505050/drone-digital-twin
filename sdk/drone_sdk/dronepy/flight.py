"""
dronepy.flight
==============
Core 6-DOF Flight Simulation Engine and Event Dispatcher for DronePy.
Integrates aerodynamics, actuators, discrete events, and flight controls.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from drone_sdk.contracts.coordinates import STANDARD_GRAVITY_MPS2
from drone_sdk.math_models.frames import quat_to_euler
from drone_sdk.math_models.rigid_body import (
    InertiaParams,
    RigidBodyState,
    compute_derivatives,
    integrate_semi_implicit_euler,
)

from .controllers import Controller, PIDPositionController
from .environment import Environment
from .events import FlightEvent
from .failures import Scenario
from .mission import Mission, Waypoint
from .results import FlightResult


class FlightSimulatorEngine:
    """Executes full 6-DOF multirotor flight simulations with discrete events."""

    def __init__(
        self,
        drone: Any,
        environment: Optional[Environment] = None,
        controller: Optional[Controller] = None,
        mission: Optional[Mission] = None,
        events: Optional[Union[List[FlightEvent], Scenario]] = None,
    ) -> None:
        self.drone = drone
        self.environment = environment or Environment.standard_atmosphere()
        self.controller = controller or PIDPositionController(mass_kg=self.drone.mass, num_motors=self.drone.num_motors)
        self.mission = mission

        # Events list
        if events is None:
            self.events = []
        elif isinstance(events, Scenario):
            self.events = events.to_events()
        else:
            self.events = list(events)

        self.events_log: List[Dict[str, Any]] = []
        self.current_time = 0.0

        # Waypoint state tracking
        self.current_wp_index = 0
        self.target_pos = np.array([0.0, 0.0, -5.0], dtype=np.float64)
        self.target_yaw = 0.0
        self.loiter_timer = 0.0

        if self.mission and len(self.mission) > 0:
            self.target_pos = self.mission[0].pos_ned.copy()
            if self.mission[0].yaw_deg is not None:
                self.target_yaw = math.radians(self.mission[0].yaw_deg)

    def log_event(self, description: str, timestamp: Optional[float] = None) -> None:
        """Record an event message into the flight telemetry event log."""
        self.events_log.append({
            "time": self.current_time if timestamp is None else timestamp,
            "description": description,
        })

    def run(
        self,
        duration: float = 60.0,
        dt: float = 0.002,
        initial_pos: Optional[np.ndarray] = None,
        initial_vel: Optional[np.ndarray] = None,
        initial_euler_rad: Optional[Tuple[float, float, float]] = None,
    ) -> FlightResult:
        """Run the flight simulation forward in time.

        Parameters
        ----------
        duration : float
            Simulation run time in seconds.
        dt : float
            Integration step size in seconds.
        initial_pos : np.ndarray, optional
            Initial NED position [x, y, z]. Defaults to [0, 0, 0] (ground).
        initial_vel : np.ndarray, optional
            Initial body velocity [u, v, w]. Defaults to zeros.
        initial_euler_rad : tuple, optional
            Initial Euler angles (roll, pitch, yaw) in radians.

        Returns
        -------
        FlightResult
        """
        # Reset controller and battery
        self.controller.reset()
        if hasattr(self.drone.battery, "reset"):
            self.drone.battery.reset()

        if self.mission and len(self.mission) > 0:
            self.log_event(f"Mission started with {len(self.mission)} waypoints.")

        total_steps = int(math.ceil(duration / dt))
        # Telemetry decimation: save at ~100 Hz to optimize memory
        save_every = max(1, int(0.01 / dt))
        history_len = int(math.ceil(total_steps / save_every))

        # Pre-allocate time series arrays
        t_arr = np.zeros(history_len, dtype=np.float64)
        pos_arr = np.zeros((history_len, 3), dtype=np.float64)
        alt_arr = np.zeros(history_len, dtype=np.float64)
        vel_ned_arr = np.zeros((history_len, 3), dtype=np.float64)
        vel_body_arr = np.zeros((history_len, 3), dtype=np.float64)
        airspeed_arr = np.zeros(history_len, dtype=np.float64)
        accel_body_arr = np.zeros((history_len, 3), dtype=np.float64)
        quat_arr = np.zeros((history_len, 4), dtype=np.float64)
        euler_rad_arr = np.zeros((history_len, 3), dtype=np.float64)
        euler_deg_arr = np.zeros((history_len, 3), dtype=np.float64)
        omega_arr = np.zeros((history_len, 3), dtype=np.float64)

        num_m = self.drone.num_motors
        motor_rpms_arr = np.zeros((history_len, num_m), dtype=np.float64)
        motor_thrusts_arr = np.zeros((history_len, num_m), dtype=np.float64)
        total_thrust_arr = np.zeros(history_len, dtype=np.float64)
        forces_body_arr = np.zeros((history_len, 3), dtype=np.float64)
        moments_body_arr = np.zeros((history_len, 3), dtype=np.float64)
        drag_body_arr = np.zeros((history_len, 3), dtype=np.float64)

        batt_v_arr = np.zeros(history_len, dtype=np.float64)
        batt_i_arr = np.zeros(history_len, dtype=np.float64)
        batt_p_arr = np.zeros(history_len, dtype=np.float64)
        batt_wh_arr = np.zeros(history_len, dtype=np.float64)
        batt_soc_arr = np.zeros(history_len, dtype=np.float64)
        motor_cmds_arr = np.zeros((history_len, num_m), dtype=np.float64)

        # Initial state setup
        p0 = np.asarray(initial_pos if initial_pos is not None else np.array([0.0, 0.0, 0.0]), dtype=np.float64)
        v0 = np.asarray(initial_vel if initial_vel is not None else np.zeros(3), dtype=np.float64)

        # Initial quaternion from euler
        if initial_euler_rad is not None:
            r, p, y = initial_euler_rad
            cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
            cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
            cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
            q0 = np.array([
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ], dtype=np.float64)
        else:
            q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        state = RigidBodyState(pos_ned=p0, vel_body=v0, quat=q0)
        grav_ned = np.array([0.0, 0.0, self.environment.gravity_mps2], dtype=np.float64)

        save_idx = 0
        self.current_time = 0.0

        for step_i in range(total_steps):
            t = step_i * dt
            self.current_time = t

            # 1. Trigger Scheduled Discrete Events
            for ev in self.events:
                if not ev.executed and t >= ev.time:
                    ev.trigger(self)

            # 2. Mission Waypoint Progression
            self._update_mission(state, dt)

            # 3. Flight Controller Computation
            motor_cmds = self.controller.compute(
                state=state,
                dt=dt,
                target_pos_ned=self.target_pos,
                target_yaw_rad=self.target_yaw,
            )

            # 4. Wind & Aerodynamics Evaluation
            v_wind = self.environment.wind_at(state.pos_ned, t=t, dt=dt)
            rho = self.environment.air_density_at(-state.pos_ned[2])

            # 5. Vehicle Wrench Computation
            wrench, thrusts, torques, current_a, power_w = self.drone.compute_wrench(
                state=state,
                motor_commands=motor_cmds,
                v_wind_ned=v_wind,
                dt=dt,
                air_density=rho,
            )

            # 6. Record State Snapshot (Decimated)
            if step_i % save_every == 0 and save_idx < history_len:
                t_arr[save_idx] = t
                pos_arr[save_idx] = state.pos_ned
                alt_arr[save_idx] = max(0.0, -state.pos_ned[2])
                vel_ned_arr[save_idx] = state.vel_ned
                vel_body_arr[save_idx] = state.vel_body
                airspeed_arr[save_idx] = float(np.linalg.norm(state.vel_body))
                quat_arr[save_idx] = state.quat

                roll, pitch, yaw = quat_to_euler(state.quat)
                euler_rad_arr[save_idx] = [roll, pitch, yaw]
                euler_deg_arr[save_idx] = [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]
                omega_arr[save_idx] = state.omega_body

                motor_rpms_arr[save_idx] = self.drone.motors.rpms
                motor_thrusts_arr[save_idx] = thrusts
                total_thrust_arr[save_idx] = float(np.sum(thrusts))
                forces_body_arr[save_idx] = wrench.force_body
                moments_body_arr[save_idx] = wrench.torque_body
                motor_cmds_arr[save_idx] = motor_cmds

                # Battery state
                b_st = self.drone.battery.state
                batt_v_arr[save_idx] = b_st.v_terminal
                batt_i_arr[save_idx] = current_a
                batt_p_arr[save_idx] = power_w
                batt_wh_arr[save_idx] = b_st.energy_consumed_wh
                batt_soc_arr[save_idx] = b_st.soc

                save_idx += 1

            # 7. Rigid Body Integration Step
            inertia_params = self.drone.to_inertia_params
            state = integrate_semi_implicit_euler(
                state=state,
                inertia=inertia_params,
                wrench=wrench,
                dt=dt,
                gravity_ned=grav_ned,
            )

            # 8. Ground Surface Contact / Rest Interaction
            if state.pos_ned[2] > 0.0:
                state.pos_ned[2] = 0.0
                state.vel_body[2] = min(0.0, state.vel_body[2])
                # Ground contact damping
                state.vel_body[0] *= 0.85
                state.vel_body[1] *= 0.85
                state.omega_body *= 0.5

        # Trim arrays to actual saved size
        return FlightResult(
            time=t_arr[:save_idx],
            pos_ned=pos_arr[:save_idx],
            altitude_agl=alt_arr[:save_idx],
            vel_ned=vel_ned_arr[:save_idx],
            vel_body=vel_body_arr[:save_idx],
            airspeed=airspeed_arr[:save_idx],
            accel_body=accel_body_arr[:save_idx],
            quaternion=quat_arr[:save_idx],
            euler_rad=euler_rad_arr[:save_idx],
            euler_deg=euler_deg_arr[:save_idx],
            omega_body=omega_arr[:save_idx],
            motor_rpms=motor_rpms_arr[:save_idx],
            motor_thrusts=motor_thrusts_arr[:save_idx],
            total_thrust=total_thrust_arr[:save_idx],
            forces_body=forces_body_arr[:save_idx],
            moments_body=moments_body_arr[:save_idx],
            drag_body=drag_body_arr[:save_idx],
            battery_voltage=batt_v_arr[:save_idx],
            battery_current=batt_i_arr[:save_idx],
            battery_power=batt_p_arr[:save_idx],
            battery_energy_wh=batt_wh_arr[:save_idx],
            battery_soc=batt_soc_arr[:save_idx],
            motor_commands=motor_cmds_arr[:save_idx],
            events_log=self.events_log,
            metadata={
                "vehicle": self.drone.name,
                "mass_kg": self.drone.mass,
                "duration_s": duration,
                "dt_s": dt,
            },
        )

    def _update_mission(self, state: RigidBodyState, dt: float) -> None:
        """Progress active mission waypoints based on position proximity."""
        if not self.mission or self.current_wp_index >= len(self.mission):
            return

        wp = self.mission[self.current_wp_index]
        dist = float(np.linalg.norm(state.pos_ned - wp.pos_ned))

        if dist < wp.acceptance_radius_m:
            if wp.loiter_time_s > 0:
                self.loiter_timer += dt
                if self.loiter_timer >= wp.loiter_time_s:
                    self.loiter_timer = 0.0
                    self._advance_waypoint()
            else:
                self._advance_waypoint()

    def _advance_waypoint(self) -> None:
        self.current_wp_index += 1
        if self.current_wp_index < len(self.mission):
            wp = self.mission[self.current_wp_index]
            self.target_pos = wp.pos_ned.copy()
            if wp.yaw_deg is not None:
                self.target_yaw = math.radians(wp.yaw_deg)
            self.log_event(f"Waypoint {self.current_wp_index} reached. Navigating to NED {self.target_pos}.")
        else:
            self.log_event("Mission completed.")

    simulate = run


class Flight(FlightSimulatorEngine):
    """
    RocketPy-style Flight execution abstraction.
    Allows declarative syntax:
        flight = Flight(drone=drone, environment=env, mission=mission)
        result = flight.simulate(duration=30.0)
    """

    def __init__(
        self,
        drone: Any,
        environment: Optional[Environment] = None,
        controller: Optional[Controller] = None,
        mission: Optional[Mission] = None,
        events: Optional[Union[List[FlightEvent], Scenario]] = None,
        duration: Optional[float] = None,
        dt: float = 0.002,
    ) -> None:
        super().__init__(
            drone=drone,
            environment=environment,
            controller=controller,
            mission=mission,
            events=events,
        )
        self.result: Optional[FlightResult] = None
        if duration is not None:
            self.result = self.simulate(duration=duration, dt=dt)
