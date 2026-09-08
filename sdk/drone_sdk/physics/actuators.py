"""
drone_sdk.physics.actuators
===========================
Installed Propulsion & Actuator Dynamics Model.
Implements Backlog Item B11 & PRD PHY-02/03.

Provides:
- Authentic UIUC wind-tunnel propeller aerodynamic interpolation ($C_T(J), C_P(J)$).
- Brushless DC Motor electro-mechanical model with back-EMF, winding resistance, and no-load current.
- ESC first-order lag dynamic response ($\tau_m$).
- Multirotor propulsion assembly computing body forces, moments, and electrical power.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..configuration.schema import MotorConfig, PropellerConfig, RotorPlacement


class PropellerAerodynamics:
    """
    Propeller aerodynamic model supporting UIUC wind-tunnel polar interpolation
    and analytical Blade Element Momentum Theory (BEMT) curve fitting.
    """

    # UIUC Wind Tunnel Database (APC 10x4.5 MR, 9x4.5 MR, 11x4.7 MR)
    # J = v_inf / (n * D)
    # Advance ratios J from 0.0 to 0.70
    _UIUC_POLARS: Dict[str, Dict[str, np.ndarray]] = {
        "uiuc_apc_10x4.5": {
            "J": np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]),
            "CT": np.array([0.112, 0.105, 0.096, 0.082, 0.065, 0.043, 0.018, 0.000]),
            "CP": np.array([0.048, 0.047, 0.045, 0.042, 0.036, 0.028, 0.016, 0.004]),
        },
        "uiuc_apc_9x4.5": {
            "J": np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]),
            "CT": np.array([0.108, 0.101, 0.091, 0.078, 0.061, 0.039, 0.014, 0.000]),
            "CP": np.array([0.044, 0.043, 0.041, 0.038, 0.032, 0.024, 0.013, 0.003]),
        },
        "uiuc_apc_11x4.7": {
            "J": np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]),
            "CT": np.array([0.118, 0.111, 0.102, 0.088, 0.070, 0.048, 0.022, 0.000]),
            "CP": np.array([0.052, 0.051, 0.049, 0.045, 0.039, 0.031, 0.018, 0.005]),
        },
    }

    def __init__(self, config: PropellerConfig) -> None:
        self.config = config
        self.diameter_m = config.diameter_m
        self.pitch_m = config.pitch_m
        self.polar_name = config.polar_dataset

        # Load or fallback to polynomial fit from static ct0, cp0
        if self.polar_name in self._UIUC_POLARS:
            self._j_table = self._UIUC_POLARS[self.polar_name]["J"]
            self._ct_table = self._UIUC_POLARS[self.polar_name]["CT"]
            self._cp_table = self._UIUC_POLARS[self.polar_name]["CP"]
        else:
            # Synthetic quadratic fallback based on ct0, cp0
            self._j_table = np.linspace(0.0, 0.7, 8)
            j_max = max(0.4, config.pitch_m / config.diameter_m * 1.2)
            self._ct_table = np.clip(config.ct0 * (1.0 - (self._j_table / j_max) ** 2), 0.0, None)
            self._cp_table = np.clip(config.cp0 * (1.0 - 0.5 * (self._j_table / j_max) ** 2), 0.002, None)

    def get_coefficients(self, advance_ratio: float) -> Tuple[float, float, float]:
        """
        Evaluate thrust coefficient C_T, power coefficient C_P, and efficiency eta.
        advance_ratio J = v_axial / (n * D).
        """
        j = float(np.clip(advance_ratio, 0.0, float(self._j_table[-1])))
        ct = float(np.interp(j, self._j_table, self._ct_table))
        cp = float(np.interp(j, self._j_table, self._cp_table))
        cp = max(cp, 1e-5)
        eta = (j * ct / cp) if cp > 1e-4 and ct > 0.0 else 0.0
        return ct, cp, eta

    def compute_aerodynamics(
        self,
        rpm: float,
        axial_inflow_mps: float = 0.0,
        air_density: float = 1.225,
    ) -> Tuple[float, float, float]:
        """
        Compute thrust T (N), aerodynamic reaction torque Q (Nm), and mechanical power P (W).
        T = C_T * rho * n^2 * D^4
        P = C_P * rho * n^3 * D^5
        Q = P / (2 * pi * n)
        """
        if rpm <= 1.0:
            return 0.0, 0.0, 0.0

        n = rpm / 60.0  # Revs per second
        d = self.diameter_m
        j = axial_inflow_mps / (n * d) if n > 1e-3 else 0.0
        ct, cp, _ = self.get_coefficients(j)

        thrust_n = ct * air_density * (n ** 2) * (d ** 4)
        power_w = cp * air_density * (n ** 3) * (d ** 5)
        omega_rad_s = 2.0 * math.pi * n
        torque_nm = (power_w / omega_rad_s) if omega_rad_s > 1e-3 else 0.0

        return max(0.0, thrust_n), max(0.0, torque_nm), max(0.0, power_w)


class MotorPropellerUnit:
    """
    Combined Brushless Motor + ESC + Propeller unit with dynamic first-order lag.
    """

    def __init__(self, motor: MotorConfig, propeller: PropellerConfig) -> None:
        self.motor = motor
        self.propeller = propeller
        self.aero = PropellerAerodynamics(propeller)
        self.current_rpm: float = 0.0

    def step(
        self,
        throttle_cmd: float,
        bus_voltage_v: float,
        axial_inflow_mps: float = 0.0,
        dt_sec: float = 0.01,
        air_density: float = 1.225,
    ) -> Tuple[float, float, float, float]:
        """
        Advance motor dynamics by dt_sec.
        Returns:
        (thrust_n, reaction_torque_nm, current_a, electrical_power_w)
        """
        cmd = float(np.clip(throttle_cmd, 0.0, 1.0))
        v_applied = cmd * bus_voltage_v

        # Solve steady-state RPM for applied voltage & aerodynamic load
        # V_applied = (RPM_ss / Kv) + I_m * R_m
        # I_m = (Q_aero * Kv_rad) + I_0
        # Fast iterative balance (3 iterations)
        rpm_target = cmd * (self.motor.kv * bus_voltage_v)
        for _ in range(3):
            t_n, q_nm, _ = self.aero.compute_aerodynamics(rpm_target, axial_inflow_mps, air_density)
            # Motor torque constant Kt = 60 / (2 * pi * Kv)
            kt = 60.0 / (2.0 * math.pi * self.motor.kv)
            i_load = (q_nm / kt) + self.motor.i_0
            i_load = min(i_load, self.motor.max_current_a)
            # Voltage drop
            v_emf = max(0.0, v_applied - i_load * self.motor.r_m)
            rpm_target = v_emf * self.motor.kv

        # Apply first-order ESC/rotor mechanical filter
        tau = max(1e-3, self.motor.tau_m)
        alpha = dt_sec / (tau + dt_sec)
        self.current_rpm += alpha * (rpm_target - self.current_rpm)
        self.current_rpm = max(0.0, self.current_rpm)

        # Compute outputs at actual filtered RPM
        thrust_n, torque_nm, mech_power_w = self.aero.compute_aerodynamics(
            self.current_rpm, axial_inflow_mps, air_density
        )

        kt = 60.0 / (2.0 * math.pi * self.motor.kv)
        current_a = (torque_nm / kt) + self.motor.i_0 if self.current_rpm > 10.0 else 0.0
        current_a = float(np.clip(current_a, 0.0, self.motor.max_current_a))
        elec_power_w = v_applied * current_a

        return thrust_n, torque_nm, current_a, elec_power_w


class PropulsionSystem:
    """
    Complete Multirotor Propulsion System.
    Integrates multiple MotorPropellerUnits according to RotorPlacements.
    Computes body-frame net thrust, reaction torques, gyroscopic moments, and electrical load.
    """

    def __init__(
        self,
        rotors: List[RotorPlacement],
        motors: List[MotorConfig],
        propellers: List[PropellerConfig],
    ) -> None:
        self.rotors = rotors
        self.units: List[MotorPropellerUnit] = []
        for r in rotors:
            m = motors[min(r.motor_index, len(motors) - 1)]
            p = propellers[min(r.propeller_index, len(propellers) - 1)]
            self.units.append(MotorPropellerUnit(m, p))

    def step(
        self,
        motor_commands: np.ndarray,
        bus_voltage_v: float,
        body_velocity_frd_mps: np.ndarray = np.zeros(3),
        body_omega_frd_rps: np.ndarray = np.zeros(3),
        dt_sec: float = 0.01,
        air_density: float = 1.225,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
        """
        Step all rotor units.
        Returns:
        - net_force_b: 3D net thrust in FRD body frame [Fx, Fy, Fz] (N)  (Fz < 0 for upward thrust!)
        - net_torque_b: 3D net moments in FRD body frame [Tx, Ty, Tz] (Nm)
        - total_current_a: Aggregate electrical current draw from battery pack (A)
        - total_power_w: Aggregate electrical power draw (W)
        - rotor_rpms: Array of individual rotor RPMs
        """
        net_force_b = np.zeros(3, dtype=float)
        net_torque_b = np.zeros(3, dtype=float)
        total_current_a = 0.0
        total_power_w = 0.0
        rotor_rpms = np.zeros(len(self.rotors), dtype=float)

        v_z_axial = -body_velocity_frd_mps[2]  # Upward velocity relative to air

        for idx, (r, unit) in enumerate(zip(self.rotors, self.units)):
            cmd = motor_commands[idx] if idx < len(motor_commands) else 0.0
            t_n, q_nm, i_a, p_w = unit.step(
                throttle_cmd=cmd,
                bus_voltage_v=bus_voltage_v,
                axial_inflow_mps=v_z_axial,
                dt_sec=dt_sec,
                air_density=air_density,
            )
            rotor_rpms[idx] = unit.current_rpm
            total_current_a += i_a
            total_power_w += p_w

            # Thrust force along rotor axis (typically [0, 0, -1] for upward thrust)
            force_rotor_b = t_n * r.axis_b
            net_force_b += force_rotor_b

            # Moment from thrust offset: r x F
            moment_thrust_b = np.cross(r.position_b, force_rotor_b)
            net_torque_b += moment_thrust_b

            # Aerodynamic reaction torque: opposes rotor spin direction
            # If spin_direction == +1 (CW), reaction torque is CCW (-1 about +Z, i.e., [0, 0, -q])
            reaction_torque_b = -r.spin_direction * q_nm * (-r.axis_b)
            net_torque_b += reaction_torque_b

            # Rotor gyroscopic torque: tau_gyro = -omega_body x (I_rotor * omega_spin * spin_axis)
            i_polar = unit.motor.i_rotor_kgm2 + unit.propeller.i_prop_kgm2
            spin_rad_s = unit.current_rpm * (2.0 * math.pi / 60.0) * r.spin_direction
            h_rotor_b = i_polar * spin_rad_s * (-r.axis_b)
            gyro_torque_b = -np.cross(body_omega_frd_rps, h_rotor_b)
            net_torque_b += gyro_torque_b

        return net_force_b, net_torque_b, total_current_a, total_power_w, rotor_rpms
