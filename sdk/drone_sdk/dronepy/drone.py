"""
dronepy.drone
=============
Main Drone Vehicle Definition and Factory Abstractions for DronePy.
Represents multirotor geometry, mass properties, actuator groups, and dynamics coupling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from drone_sdk.battery_twin.pack_model import PackECMModel
from drone_sdk.contracts.coordinates import STANDARD_GRAVITY_MPS2
from drone_sdk.math_models.frames import quat_rotate_vector, quat_to_dcm
from drone_sdk.math_models.rigid_body import ExternalWrench, InertiaParams, RigidBodyState

from .aerodynamics import AerodynamicsModel, AnalyticalDrag
from .motors import Motor, MotorGroup
from .propellers import Propeller, UIUCPropeller


class Drone:
    """Multirotor UAV physical definition and flight simulation entry point.

    Example
    -------
    >>> from drone_sdk.dronepy import Drone, Environment
    >>> drone = Drone.quadcopter(mass=1.5, arm_length=0.25)
    >>> env = Environment.standard_atmosphere()
    >>> flight = drone.simulate(duration=30.0, environment=env)
    >>> flight.plot_trajectory()
    """

    def __init__(
        self,
        name: str = "Multirotor",
        mass: float = 1.50,
        arm_length: float = 0.25,
        inertia: Optional[Union[np.ndarray, List[float]]] = None,
        motors: Optional[Union[MotorGroup, List[Motor]]] = None,
        propellers: Optional[List[Propeller]] = None,
        aerodynamics: Optional[AerodynamicsModel] = None,
        battery: Optional[PackECMModel] = None,
        num_motors: int = 4,
        motor_angles_rad: Optional[List[float]] = None,
        motor_spin_dirs: Optional[List[int]] = None,
    ) -> None:
        self.name = name
        self.base_mass = float(mass)
        self.payload_mass = 0.0
        self.arm_length = float(arm_length)
        self.center_of_gravity = np.zeros(3, dtype=np.float64)

        # 3x3 Inertia tensor setup
        if inertia is None:
            # Approximate thin-rod / cylinder multirotor inertia
            ixx = 0.035 * (mass / 1.5) * ((arm_length / 0.25) ** 2)
            iyy = 0.046 * (mass / 1.5) * ((arm_length / 0.25) ** 2)
            izz = 0.098 * (mass / 1.5) * ((arm_length / 0.25) ** 2)
            self._base_inertia = np.diag([ixx, iyy, izz]).astype(np.float64)
        else:
            arr = np.asarray(inertia, dtype=np.float64)
            self._base_inertia = np.diag(arr) if arr.shape == (3,) else arr.copy()

        self._current_inertia = self._base_inertia.copy()

        # Actuator geometry
        self.num_motors = num_motors
        if motor_angles_rad is not None:
            self.motor_angles = list(motor_angles_rad)
        else:
            # Default symmetric distribution
            offset = math.pi / num_motors
            self.motor_angles = [offset + i * (2.0 * math.pi / num_motors) for i in range(num_motors)]

        if motor_spin_dirs is not None:
            self.motor_spin_dirs = list(motor_spin_dirs)
        else:
            # Alternating CW / CCW
            self.motor_spin_dirs = [1 if i % 2 == 0 else -1 for i in range(num_motors)]

        # Calculate 2D motor positions relative to CG [x_b, y_b, z_b]
        self.motor_positions = np.zeros((num_motors, 3), dtype=np.float64)
        for i in range(num_motors):
            angle = self.motor_angles[i]
            self.motor_positions[i, 0] = self.arm_length * math.cos(angle)
            self.motor_positions[i, 1] = self.arm_length * math.sin(angle)
            self.motor_positions[i, 2] = 0.0  # Motors on body horizontal plane

        # Motors
        if motors is None:
            default_motors = [
                Motor(
                    max_rpm=9500.0,
                    time_constant=0.035,
                    thrust_coefficient=0.082 * (mass / 1.5),
                    direction=self.motor_spin_dirs[i],
                    name=f"M{i+1}",
                )
                for i in range(num_motors)
            ]
            self.motors = MotorGroup(default_motors)
        elif isinstance(motors, MotorGroup):
            self.motors = motors
        else:
            self.motors = MotorGroup(motors)

        # Propellers
        self.propellers = propellers or [Propeller(diameter_in=10.0, pitch_in=4.5) for _ in range(num_motors)]

        # Aerodynamics
        self.aerodynamics = aerodynamics or AnalyticalDrag()

        # Battery Pack
        self.battery = battery or PackECMModel()

    @property
    def mass(self) -> float:
        """Total current mass including payload (kg)."""
        return self.base_mass + self.payload_mass

    @mass.setter
    def mass(self, value: float) -> None:
        self.base_mass = float(value) - self.payload_mass

    @property
    def mass_kg(self) -> float:
        """Alias for mass in kg."""
        return self.mass

    @mass_kg.setter
    def mass_kg(self, value: float) -> None:
        self.base_mass = float(value) - self.payload_mass

    @property
    def inertia_tensor(self) -> np.ndarray:
        """Current 3x3 inertia tensor about CG."""
        return self._current_inertia

    @property
    def to_inertia_params(self) -> InertiaParams:
        return InertiaParams(mass_kg=self.mass, inertia_tensor=self.inertia_tensor)

    # ── Factory Constructors ──────────────────────────────────────────────────

    @classmethod
    def quadcopter(
        cls,
        mass: float = 1.50,
        arm_length: float = 0.25,
        inertia: Optional[Union[np.ndarray, List[float]]] = None,
        motors: Optional[Union[MotorGroup, List[Motor]]] = None,
        propellers: Optional[List[Propeller]] = None,
        aerodynamics: Optional[AerodynamicsModel] = None,
        battery: Optional[PackECMModel] = None,
    ) -> "Drone":
        """Factory constructor for standard X-configuration Quadcopter."""
        # Motor angles for X-config: 45° (FR, CW), 135° (RR, CCW), 225° (RL, CW), 315° (FL, CCW)
        angles = [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4]
        spins = [1, -1, 1, -1]
        return cls(
            name="Quadcopter_X",
            mass=mass,
            arm_length=arm_length,
            inertia=inertia,
            motors=motors,
            propellers=propellers,
            aerodynamics=aerodynamics,
            battery=battery,
            num_motors=4,
            motor_angles_rad=angles,
            motor_spin_dirs=spins,
        )

    @classmethod
    def hexacopter(
        cls,
        mass: float = 2.60,
        arm_length: float = 0.35,
        inertia: Optional[Union[np.ndarray, List[float]]] = None,
        motors: Optional[Union[MotorGroup, List[Motor]]] = None,
        propellers: Optional[List[Propeller]] = None,
        aerodynamics: Optional[AerodynamicsModel] = None,
        battery: Optional[PackECMModel] = None,
    ) -> "Drone":
        """Factory constructor for standard 6-rotor Hexacopter."""
        angles = [i * (math.pi / 3.0) for i in range(6)]
        spins = [1, -1, 1, -1, 1, -1]
        return cls(
            name="Hexacopter",
            mass=mass,
            arm_length=arm_length,
            inertia=inertia,
            motors=motors,
            propellers=propellers,
            aerodynamics=aerodynamics,
            battery=battery,
            num_motors=6,
            motor_angles_rad=angles,
            motor_spin_dirs=spins,
        )

    @classmethod
    def octocopter(
        cls,
        mass: float = 4.50,
        arm_length: float = 0.45,
        inertia: Optional[Union[np.ndarray, List[float]]] = None,
        motors: Optional[Union[MotorGroup, List[Motor]]] = None,
        propellers: Optional[List[Propeller]] = None,
        aerodynamics: Optional[AerodynamicsModel] = None,
        battery: Optional[PackECMModel] = None,
    ) -> "Drone":
        """Factory constructor for standard 8-rotor Octocopter."""
        angles = [i * (math.pi / 4.0) for i in range(8)]
        spins = [1, -1, 1, -1, 1, -1, 1, -1]
        return cls(
            name="Octocopter",
            mass=mass,
            arm_length=arm_length,
            inertia=inertia,
            motors=motors,
            propellers=propellers,
            aerodynamics=aerodynamics,
            battery=battery,
            num_motors=8,
            motor_angles_rad=angles,
            motor_spin_dirs=spins,
        )

    # ── Dynamic Payload / Event Operations ───────────────────────────────────

    def add_payload(
        self,
        mass_added: Optional[float] = None,
        location_body: Optional[np.ndarray] = None,
        mass_kg: Optional[float] = None,
        offset_m: Optional[np.ndarray] = None,
    ) -> None:
        """Add payload mass at a specific offset from the vehicle origin."""
        added = float(mass_kg if mass_kg is not None else (mass_added if mass_added is not None else 0.0))
        r_loc = offset_m if offset_m is not None else (location_body if location_body is not None else np.array([0.0, 0.0, 0.05]))
        r_pay = np.asarray(r_loc, dtype=np.float64)

        old_mass = self.mass
        self.payload_mass += added
        new_mass = self.mass

        # Update center of gravity
        self.center_of_gravity = (old_mass * self.center_of_gravity + added * r_pay) / max(1e-6, new_mass)

        # Parallel axis theorem inertia shift: I_new = I_old + m * (r^2 * E - r x r)
        r_sq = np.dot(r_pay, r_pay)
        i_added = added * (r_sq * np.eye(3) - np.outer(r_pay, r_pay))
        self._current_inertia += i_added

    def release_payload(
        self,
        mass_released: Optional[float] = None,
    ) -> None:
        """Drop or release payload mass, dynamically reducing mass and restoring nominal inertia."""
        drop = self.payload_mass if mass_released is None else min(self.payload_mass, mass_released)
        frac = drop / max(1e-6, self.payload_mass) if self.payload_mass > 1e-6 else 1.0
        self.payload_mass -= drop
        if self.payload_mass <= 1e-6:
            self.center_of_gravity = np.zeros(3, dtype=np.float64)
            self._current_inertia = self._base_inertia.copy()
        else:
            self.center_of_gravity *= (1.0 - frac)
            self._current_inertia = self._base_inertia + (self._current_inertia - self._base_inertia) * (1.0 - frac)

        # Linearly relax inertia back towards base
        self._current_inertia = self._base_inertia + (self._current_inertia - self._base_inertia) * (1.0 - frac)

    def copy(self) -> "Drone":
        """Return an independent deep copy of this Drone."""
        import copy
        return copy.deepcopy(self)

    # ── Dynamics & Wrench Computation ────────────────────────────────────────

    def compute_wrench(
        self,
        state: RigidBodyState,
        motor_commands: np.ndarray,
        v_wind_ned: Optional[np.ndarray] = None,
        dt: float = 0.002,
        air_density: float = 1.225,
    ) -> Tuple[ExternalWrench, np.ndarray, np.ndarray, float, float]:
        """Compute total forces and torques acting on the drone body.

        Returns
        -------
        Tuple[ExternalWrench, thrusts_n, torques_nm, current_a, power_w]
        """
        # Step motor dynamics
        bus_v = self.battery.state.v_terminal if hasattr(self.battery, "state") else 16.0
        rho_ratio = air_density / 1.225

        thrusts, torques, total_curr, total_pwr = self.motors.step(
            motor_commands,
            bus_voltage=bus_v,
            dt=dt,
            air_density_ratio=rho_ratio,
        )

        # 1. Total Multirotor Thrust: Acts along -z_b (upward in FRD)
        total_thrust_n = float(np.sum(thrusts))
        f_thrust_body = np.array([0.0, 0.0, -total_thrust_n], dtype=np.float64)

        # 2. Control Torques from Individual Motor Placements & Reaction Torques
        # Roll torque (tau_x) = sum(y_i * T_i)
        # Pitch torque (tau_y) = -sum(x_i * T_i)
        # Yaw torque (tau_z) = sum(Q_i)
        tau_x = float(np.sum(self.motor_positions[:, 1] * thrusts))
        tau_y = float(-np.sum(self.motor_positions[:, 0] * thrusts))
        tau_z = float(np.sum(torques))
        tau_motors_body = np.array([tau_x, tau_y, tau_z], dtype=np.float64)

        # 3. Aerodynamics (Parasitic Drag & Rotational Damping)
        v_body = state.vel_body
        if v_wind_ned is not None:
            R_bn = quat_to_dcm(state.quat).T
            v_wind_body = R_bn @ v_wind_ned
            v_air_body = v_body - v_wind_body
        else:
            v_air_body = v_body

        f_aero, tau_aero = self.aerodynamics.compute_forces_and_moments(
            v_air_body=v_air_body,
            omega_body=state.omega_body,
            air_density=air_density,
        )

        total_force = f_thrust_body + f_aero
        total_torque = tau_motors_body + tau_aero

        # Advance battery ECM state with electrical load
        if hasattr(self.battery, "step_discharge"):
            self.battery.step_discharge(current_a=total_curr, dt_sec=dt)

        return (
            ExternalWrench(force_body=total_force, torque_body=total_torque),
            thrusts,
            torques,
            total_curr,
            total_pwr,
        )

    # ── Simulation Runner ─────────────────────────────────────────────────────

    def simulate(
        self,
        duration: float = 60.0,
        dt: float = 0.002,
        environment: Optional[Any] = None,
        controller: Optional[Any] = None,
        mission: Optional[Any] = None,
        events: Optional[List[Any]] = None,
        initial_position_ned: Optional[np.ndarray] = None,
        initial_velocity_body: Optional[np.ndarray] = None,
    ) -> Any:
        """Run a 6-DOF trajectory simulation and return a rich FlightResult.

        Parameters
        ----------
        duration : float
            Total flight simulation time in seconds.
        dt : float
            Simulation integration timestep (e.g. 0.002s = 500 Hz).
        environment : Environment, optional
            Atmosphere and wind field. Defaults to ISA standard atmosphere.
        controller : Controller, optional
            Flight controller. Defaults to cascaded PID position/attitude hold.
        mission : Mission, optional
            High-level waypoint/flight phase mission.
        events : list, optional
            Scheduled discrete events (payload release, motor failure, wind shear).

        Returns
        -------
        FlightResult
            Time-series telemetry object with plotting and export tools.
        """
        # Lazy import to avoid circular dependency
        from .flight import FlightSimulatorEngine
        sim = FlightSimulatorEngine(
            drone=self,
            environment=environment,
            controller=controller,
            mission=mission,
            events=events,
        )
        return sim.run(
            duration=duration,
            dt=dt,
            initial_pos=initial_position_ned,
            initial_vel=initial_velocity_body,
        )
