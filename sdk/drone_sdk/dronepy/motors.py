"""
dronepy.motors
==============
Electric Motor & Actuator Models for Multirotor Drone Flight Analysis.
Supports BLDC dynamics, spin-up response lag, voltage dependence, and failure injection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class Motor:
    """Brushless DC Motor and electronic speed controller (ESC) model.

    Parameters
    ----------
    max_rpm : float
        Maximum rotor speed in revolutions per minute (RPM).
    min_rpm : float
        Minimum idle RPM when armed.
    time_constant : float
        First-order spin-up/spin-down response time constant tau_m (seconds).
    thrust_coefficient : float
        Static thrust factor k_t such that Thrust = k_t * (RPM/1000)^2 (Newtons).
    torque_coefficient : float
        Aerodynamic reaction torque factor k_q such that Torque = k_q * (RPM/1000)^2 (N·m).
    voltage : float
        Nominal operating bus voltage (V).
    kv : float
        Motor velocity constant (RPM per Volt).
    resistance_ohm : float
        Armature winding internal electrical resistance (Ohms).
    no_load_current_a : float
        Idle no-load motor current draw (Amps).
    direction : int
        Spin direction: +1 for Clockwise (CW), -1 for Counter-Clockwise (CCW).
    name : str
        Descriptive motor identifier (e.g., 'Motor_1_FrontRight').
    """
    max_rpm: float = 9500.0
    min_rpm: float = 0.0
    time_constant: float = 0.035
    thrust_coefficient: float = 0.082   # N / (kRPM)^2, yields ~7.4 N at 9500 RPM
    torque_coefficient: float = 0.0018  # Nm / (kRPM)^2
    voltage: float = 16.0
    kv: float = 920.0
    resistance_ohm: float = 0.095
    no_load_current_a: float = 0.65
    direction: int = 1
    name: str = "Motor"

    # Dynamic state
    current_rpm: float = 0.0
    commanded_throttle: float = 0.0
    efficiency: float = 1.0
    is_failed: bool = False

    def failure(self, rpm_limit: Optional[float] = None) -> None:
        """Inject an immediate motor failure or RPM cap."""
        if rpm_limit is not None and rpm_limit > 0:
            self.max_rpm = float(rpm_limit)
        else:
            self.is_failed = True
            self.current_rpm = 0.0

    fail = failure

    def recover(self) -> None:
        """Recover motor from failure state."""
        self.is_failed = False

    def set_efficiency(self, efficiency: float) -> None:
        """Set motor mechanical efficiency multiplier [0.0, 1.0]."""
        self.efficiency = max(0.0, min(1.0, float(efficiency)))

    def set_rpm(self, target_rpm: float) -> None:
        """Directly override the instantaneous motor RPM."""
        if not self.is_failed:
            self.current_rpm = float(np.clip(target_rpm, self.min_rpm, self.max_rpm))

    def step(
        self,
        command: Optional[float] = None,
        bus_voltage: Optional[float] = None,
        dt: float = 0.002,
        air_density_ratio: float = 1.0,
        cmd_rpm: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        """Advance motor dynamics by timestep dt. Supports normalized command or cmd_rpm."""
        if self.is_failed:
            self.current_rpm = 0.0
            self.commanded_throttle = 0.0
            return 0.0, 0.0, 0.0, 0.0

        if cmd_rpm is not None:
            cmd = float(np.clip(cmd_rpm / max(1.0, self.max_rpm), 0.0, 1.0))
        elif command is not None:
            cmd = float(np.clip(command, 0.0, 1.0))
        else:
            cmd = 0.0
        self.commanded_throttle = cmd

        v_bus = bus_voltage if bus_voltage is not None else self.voltage

        # Voltage scaling on achievable max RPM
        voltage_scale = min(1.2, max(0.4, v_bus / max(1.0, self.voltage)))
        achievable_max_rpm = self.max_rpm * voltage_scale

        target_rpm = self.min_rpm + cmd * (achievable_max_rpm - self.min_rpm)

        # First-order response lag: dOmega/dt = (Omega_target - Omega) / tau
        if self.time_constant > 1e-5:
            alpha = min(1.0, dt / self.time_constant)
            self.current_rpm += (target_rpm - self.current_rpm) * alpha
        else:
            self.current_rpm = target_rpm

        self.current_rpm = float(np.clip(self.current_rpm, 0.0, self.max_rpm))

        # Aerodynamic Thrust and Torque
        krpm = self.current_rpm / 1000.0
        thrust_n = self.thrust_coefficient * (krpm ** 2) * air_density_ratio * self.efficiency
        torque_mag_nm = self.torque_coefficient * (krpm ** 2) * air_density_ratio * self.efficiency
        # Reaction torque on drone body is opposite to rotor spin direction
        torque_nm = -self.direction * torque_mag_nm

        # Electrical Current: I = (Torque * Omega_rad / V) + I_loss + I_ohmic
        omega_rad = (self.current_rpm / 60.0) * (2.0 * math.pi)
        mech_power_w = torque_mag_nm * omega_rad

        if cmd > 0.01:
            i_load = mech_power_w / max(1.0, v_bus)
            i_ohmic = (cmd * v_bus / max(0.01, self.resistance_ohm)) * 0.05
            current_a = (self.no_load_current_a * cmd + i_load + i_ohmic)
        else:
            current_a = 0.05

        electrical_power_w = current_a * v_bus

        return max(0.0, thrust_n), torque_nm, max(0.0, current_a), max(0.0, electrical_power_w)


class MotorGroup:
    """Collection of motors mounted on the drone airframe."""

    def __init__(self, motors: List[Motor]) -> None:
        self.motors = list(motors)

    def __len__(self) -> int:
        return len(self.motors)

    def __getitem__(self, idx: int) -> Motor:
        return self.motors[idx]

    def failure(self, motor_index: int) -> None:
        """Fail a specific motor by index (0-indexed)."""
        if 0 <= motor_index < len(self.motors):
            self.motors[motor_index].failure()

    def recover_all(self) -> None:
        """Restore all motors to nominal condition."""
        for m in self.motors:
            m.recover()

    def step(
        self,
        commands: Union[np.ndarray, List[float]],
        bus_voltage: Optional[float] = None,
        dt: float = 0.002,
        air_density_ratio: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Step all motors in the group.

        Returns
        -------
        thrusts_n : np.ndarray (N,)
        torques_nm : np.ndarray (N,)
        total_current_a : float
        total_power_w : float
        """
        cmds = np.asarray(commands, dtype=np.float64)
        if len(cmds) != len(self.motors):
            # Broadcast or pad
            padded = np.zeros(len(self.motors), dtype=np.float64)
            n_copy = min(len(cmds), len(self.motors))
            padded[:n_copy] = cmds[:n_copy]
            cmds = padded

        n = len(self.motors)
        thrusts = np.zeros(n, dtype=np.float64)
        torques = np.zeros(n, dtype=np.float64)
        total_current = 0.0
        total_power = 0.0

        for i, motor in enumerate(self.motors):
            t_n, q_nm, i_a, p_w = motor.step(
                float(cmds[i]),
                bus_voltage=bus_voltage,
                dt=dt,
                air_density_ratio=air_density_ratio,
            )
            thrusts[i] = t_n
            torques[i] = q_nm
            total_current += i_a
            total_power += p_w

        return thrusts, torques, total_current, total_power

    @property
    def rpms(self) -> np.ndarray:
        """Current RPM vector of all motors."""
        return np.array([m.current_rpm for m in self.motors], dtype=np.float64)
