"""
state_manager.validator
=======================
PhysicsValidator — enforces physical plausibility bounds on DroneStateVector.

Purpose
-------
Catch software bugs, sensor glitches, or corrupted telemetry *before* invalid
data propagates into control algorithms, AI models, or the dashboard.

The validator is intentionally lenient about values that are physically
possible but operationally unusual (e.g. 85° bank angle), and strict about
values that are physically impossible (e.g. battery SoC > 1.0, quaternion
norm far from 1.0, negative rotor speed).

Usage
-----
    validator = PhysicsValidator(config)
    result = validator.validate(state)
    if not result.is_valid:
        for issue in result.issues:
            print(issue)

All validation methods return bool so they can be composed or tested
individually in unit tests.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

from .schema import DroneStateVector, VehicleConfig

logger = logging.getLogger(__name__)


# ── Validation result ─────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of a single validation pass.

    Attributes:
        is_valid:  True when no issues were found.
        issues:    Human-readable list of constraint violations.
        warnings:  Non-fatal concerns (out-of-envelope but physically possible).
    """
    is_valid: bool               = True
    issues:   list[str]          = field(default_factory=list)
    warnings: list[str]          = field(default_factory=list)

    def add_issue(self, msg: str) -> None:
        """Record a fatal issue and mark the result invalid."""
        self.is_valid = False
        self.issues.append(msg)

    def add_warning(self, msg: str) -> None:
        """Record a non-fatal warning (does not affect is_valid)."""
        self.warnings.append(msg)

    def __bool__(self) -> bool:  # noqa: D105
        return self.is_valid

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ValidationResult(valid={self.is_valid}, "
            f"issues={len(self.issues)}, warnings={len(self.warnings)})"
        )


# ── PhysicsValidator ──────────────────────────────────────────────────────────

class PhysicsValidator:
    """Validates DroneStateVector against configurable physical bounds.

    All methods are pure (no side effects) so they can be tested in isolation.

    Args:
        config: VehicleConfig containing bounds appropriate for the airframe.

    Example::

        cfg = VehicleConfig(vehicle_id="q1", max_speed_ms=20.0)
        val = PhysicsValidator(cfg)
        result = val.validate(state)
        assert result.is_valid
    """

    def __init__(self, config: VehicleConfig) -> None:
        self._cfg = config

    # ── Master validation entry point ─────────────────────────────────────────

    def validate(self, state: DroneStateVector) -> ValidationResult:
        """Run all validation checks on a DroneStateVector.

        Returns a ValidationResult that aggregates all issues found.
        """
        result = ValidationResult()

        # Run each subsystem validator; they write directly into result.
        self._check_quaternion(state, result)
        self._check_attitude(state, result)
        self._check_position(state, result)
        self._check_velocity(state, result)
        self._check_angular_rates(state, result)
        self._check_rotors(state, result)
        self._check_battery(state, result)
        self._check_gps(state, result)
        self._check_derived(state, result)

        if not result.is_valid:
            logger.debug(
                "Validator[%s]: failed — %d issues: %s",
                state.vehicle_id,
                len(result.issues),
                result.issues,
            )

        return result

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_quaternion(self, s: DroneStateVector, r: ValidationResult) -> None:
        """Quaternion unit-norm constraint: |q| ≈ 1."""
        norm = s.quaternion_norm
        tol  = self._cfg.quaternion_norm_tol
        if abs(norm - 1.0) > tol:
            r.add_issue(
                f"Quaternion norm {norm:.6f} deviates from 1.0 by "
                f"{abs(norm - 1.0):.2e} (tolerance {tol:.2e})"
            )

    def _check_attitude(self, s: DroneStateVector, r: ValidationResult) -> None:
        """Euler angle range checks."""
        max_roll  = self._cfg.max_roll_rad
        max_pitch = self._cfg.max_pitch_rad

        if abs(s.roll) > max_roll:
            r.add_issue(
                f"Roll {math.degrees(s.roll):.1f}° exceeds limit "
                f"±{math.degrees(max_roll):.1f}°"
            )
        if abs(s.pitch) > max_pitch:
            r.add_issue(
                f"Pitch {math.degrees(s.pitch):.1f}° exceeds limit "
                f"±{math.degrees(max_pitch):.1f}°"
            )
        # Yaw is unrestricted [-π, π], but must be finite
        if not math.isfinite(s.yaw):
            r.add_issue(f"Yaw is not finite: {s.yaw}")

    def _check_position(self, s: DroneStateVector, r: ValidationResult) -> None:
        """Position bounds: altitude and lateral range."""
        alt_agl = s.altitude_agl
        max_alt = self._cfg.max_altitude_m

        if alt_agl > max_alt:
            r.add_issue(
                f"Altitude AGL {alt_agl:.1f} m exceeds ceiling {max_alt:.1f} m"
            )
        # Lateral range from origin (proxy for home position)
        lateral = math.sqrt(s.x**2 + s.y**2)
        max_lat = self._cfg.max_lateral_range_m
        if lateral > max_lat:
            r.add_warning(
                f"Lateral range {lateral:.0f} m exceeds soft limit {max_lat:.0f} m"
            )
        # Finiteness
        for name, val in (("x", s.x), ("y", s.y), ("z", s.z)):
            if not math.isfinite(val):
                r.add_issue(f"Position component '{name}' is not finite: {val}")

    def _check_velocity(self, s: DroneStateVector, r: ValidationResult) -> None:
        """Velocity magnitude and per-axis finiteness."""
        speed = math.sqrt(s.vx**2 + s.vy**2 + s.vz**2)
        max_speed = self._cfg.max_speed_ms
        if speed > max_speed:
            r.add_issue(
                f"Groundspeed {speed:.1f} m/s exceeds limit {max_speed:.1f} m/s"
            )

        max_vs = self._cfg.max_vertical_speed_ms
        if abs(s.vz) > max_vs:
            r.add_issue(
                f"Vertical speed {abs(s.vz):.1f} m/s exceeds limit {max_vs:.1f} m/s"
            )

        for name, val in (("vx", s.vx), ("vy", s.vy), ("vz", s.vz)):
            if not math.isfinite(val):
                r.add_issue(f"Velocity component '{name}' is not finite: {val}")

    def _check_angular_rates(self, s: DroneStateVector, r: ValidationResult) -> None:
        """Body-frame angular rate limits."""
        max_rate = self._cfg.max_angular_rate_rads
        for name, val in (
            ("roll_rate", s.roll_rate),
            ("pitch_rate", s.pitch_rate),
            ("yaw_rate", s.yaw_rate),
        ):
            if not math.isfinite(val):
                r.add_issue(f"Angular rate '{name}' is not finite: {val}")
                continue
            if abs(val) > max_rate:
                r.add_issue(
                    f"Angular rate '{name}' {abs(val):.2f} rad/s exceeds "
                    f"limit {max_rate:.2f} rad/s"
                )

    def _check_rotors(self, s: DroneStateVector, r: ValidationResult) -> None:
        """Rotor speed limits: must be in [min_omega, max_omega]."""
        min_w = self._cfg.min_rotor_omega_rads
        max_w = self._cfg.max_rotor_omega_rads
        for idx, omega in enumerate(s.rotor_speeds(), start=1):
            if not math.isfinite(omega):
                r.add_issue(f"Rotor {idx} omega is not finite: {omega}")
                continue
            if omega < min_w:
                r.add_issue(
                    f"Rotor {idx} omega {omega:.1f} rad/s below minimum {min_w:.1f}"
                )
            if omega > max_w:
                r.add_issue(
                    f"Rotor {idx} omega {omega:.1f} rad/s above maximum {max_w:.1f}"
                )

    def _check_battery(self, s: DroneStateVector, r: ValidationResult) -> None:
        """Battery electrical checks: voltage, SoC, SoH."""
        v_min = self._cfg.battery_voltage_min
        v_max = self._cfg.battery_voltage_max

        # Voltage: zero = no telemetry yet (skip); negative = sensor fault (flag)
        if s.battery_voltage < 0:
            r.add_issue(f"Battery voltage {s.battery_voltage:.3f} V is negative")
        elif s.battery_voltage > 0:
            if s.battery_voltage < v_min:
                r.add_issue(
                    f"Battery voltage {s.battery_voltage:.2f} V below critical "
                    f"minimum {v_min:.2f} V"
                )
            if s.battery_voltage > v_max:
                r.add_issue(
                    f"Battery voltage {s.battery_voltage:.2f} V exceeds maximum "
                    f"{v_max:.2f} V (over-charge)"
                )

        # State of Charge
        if not (0.0 <= s.battery_soc <= 1.0):
            r.add_issue(f"Battery SoC {s.battery_soc:.3f} not in [0, 1]")

        # State of Health
        if not (0.0 <= s.battery_soh <= 1.0):
            r.add_issue(f"Battery SoH {s.battery_soh:.3f} not in [0, 1]")

        # Current must be non-negative (convention: positive = discharging)
        if s.battery_current < 0:
            r.add_warning(
                f"Battery current {s.battery_current:.2f} A is negative — "
                "charging detected or sensor error"
            )

    def _check_gps(self, s: DroneStateVector, r: ValidationResult) -> None:
        """GPS coordinate range checks (only when fix is present)."""
        if s.gps_fix_type < 2:
            return  # No fix — skip coordinate validation

        if not (-90.0 <= s.gps_lat <= 90.0):
            r.add_issue(f"GPS latitude {s.gps_lat:.6f}° out of range [-90, 90]")

        if not (-180.0 <= s.gps_lon <= 180.0):
            r.add_issue(f"GPS longitude {s.gps_lon:.6f}° out of range [-180, 180]")

        if s.gps_satellites < 0:
            r.add_issue(f"GPS satellite count {s.gps_satellites} is negative")

        if s.gps_hdop < 0:
            r.add_issue(f"GPS HDOP {s.gps_hdop:.2f} is negative")

    def _check_derived(self, s: DroneStateVector, r: ValidationResult) -> None:
        """Derived scalar checks: groundspeed, airspeed, heading."""
        if s.groundspeed < 0:
            r.add_issue(f"groundspeed {s.groundspeed:.2f} m/s is negative")
        if s.airspeed < 0:
            r.add_issue(f"airspeed {s.airspeed:.2f} m/s is negative")
        if not (0.0 <= s.heading < 360.0):
            r.add_warning(
                f"heading {s.heading:.1f}° not in [0, 360) — may be un-initialised"
            )

    # ── Convenience single-field validators ───────────────────────────────────

    def is_valid_quaternion(self, q0: float, q1: float, q2: float, q3: float) -> bool:
        """Return True if the quaternion has unit norm within tolerance."""
        norm = math.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        return abs(norm - 1.0) <= self._cfg.quaternion_norm_tol

    def is_valid_battery_voltage(self, voltage: float) -> bool:
        """Return True if voltage is within configured pack voltage range."""
        if voltage <= 0:
            return True  # No telemetry yet
        return self._cfg.battery_voltage_min <= voltage <= self._cfg.battery_voltage_max

    def is_valid_speed(self, vx: float, vy: float, vz: float) -> bool:
        """Return True if the 3-D speed magnitude is within limits."""
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        return speed <= self._cfg.max_speed_ms

    def is_valid_rotor_omega(self, omega: float) -> bool:
        """Return True if a single rotor speed is within limits."""
        return self._cfg.min_rotor_omega_rads <= omega <= self._cfg.max_rotor_omega_rads
