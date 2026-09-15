"""
drone_sdk.math_models.frames
=============================
Coordinate Frame Transformations & Attitude Representations.

Provides a complete toolkit for converting between:
  - NED (North-East-Down) — standard aerospace inertial frame
  - ENU (East-North-Up) — ROS/robotics convention
  - FRD (Forward-Right-Down) — body frame

Attitude representations:
  - Euler angles (Tait-Bryan ZYX: yaw-pitch-roll)
  - Quaternion (Hamilton, scalar-first [q0, q1, q2, q3])
  - Direction Cosine Matrix (3×3 rotation matrix)

Wind triangle:
  - Airspeed, angle of attack α, sideslip β from body velocity & wind

References:
  - Stevens, Lewis & Johnson, "Aircraft Control and Simulation", 3rd Ed, Wiley 2016.
  - Diebel, "Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors", 2006.

Python version: 3.9+
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Euler ↔ Quaternion ↔ DCM conversions (ZYX Tait-Bryan)
# ─────────────────────────────────────────────────────────────────────────────

def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (roll φ, pitch θ, yaw ψ) to quaternion [q0, q1, q2, q3].

    Convention: ZYX intrinsic (aerospace standard).
    q0 is the scalar part.
    """
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

    return np.array([
        cr * cp * cy + sr * sp * sy,   # q0
        sr * cp * cy - cr * sp * sy,   # q1
        cr * sp * cy + sr * cp * sy,   # q2
        cr * cp * sy - sr * sp * cy,   # q3
    ], dtype=np.float64)


def quat_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """Convert quaternion [q0, q1, q2, q3] to Euler angles (roll, pitch, yaw).

    Returns (roll, pitch, yaw) in radians. Handles gimbal lock at ±90° pitch.
    """
    q0, q1, q2, q3 = q
    # Roll (φ)
    sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (θ) — clamped to avoid NaN at gimbal lock
    sinp = 2.0 * (q0 * q2 - q3 * q1)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    # Yaw (ψ)
    siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_dcm(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler angles → 3×3 Direction Cosine Matrix (body-to-NED rotation R_nb).

    R_nb rotates a vector from body frame to NED:  v_ned = R_nb @ v_body.
    Convention: ZYX intrinsic (R = Rz(ψ) @ Ry(θ) @ Rx(φ)).
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    return np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [   -sp,             cp * sr,                  cp * cr      ],
    ], dtype=np.float64)


def dcm_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract Euler angles (roll, pitch, yaw) from a rotation matrix R_nb."""
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: pitch ≈ ±90°
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Quaternion → 3×3 DCM (body-to-NED)."""
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*(q2*q2 + q3*q3),  2*(q1*q2 - q0*q3),      2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),      1 - 2*(q1*q1 + q3*q3),  2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),      2*(q2*q3 + q0*q1),      1 - 2*(q1*q1 + q2*q2)],
    ], dtype=np.float64)


def dcm_to_quat(R: np.ndarray) -> np.ndarray:
    """DCM → quaternion [q0, q1, q2, q3] via Shepperd's method (numerically stable)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q0 = 0.25 / s
        q1 = (R[2, 1] - R[1, 2]) * s
        q2 = (R[0, 2] - R[2, 0]) * s
        q3 = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q0 = (R[2, 1] - R[1, 2]) / s
        q1 = 0.25 * s
        q2 = (R[0, 1] + R[1, 0]) / s
        q3 = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q0 = (R[0, 2] - R[2, 0]) / s
        q1 = (R[0, 1] + R[1, 0]) / s
        q2 = 0.25 * s
        q3 = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q0 = (R[1, 0] - R[0, 1]) / s
        q1 = (R[0, 2] + R[2, 0]) / s
        q2 = (R[1, 2] + R[2, 1]) / s
        q3 = 0.25 * s

    qv = np.array([q0, q1, q2, q3], dtype=np.float64)
    if qv[0] < 0:
        qv = -qv  # Enforce positive scalar part convention
    return qv / np.linalg.norm(qv)


# ─────────────────────────────────────────────────────────────────────────────
#  Quaternion algebra
# ─────────────────────────────────────────────────────────────────────────────

def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product p ⊗ q."""
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    return np.array([
        p0*q0 - p1*q1 - p2*q2 - p3*q3,
        p0*q1 + p1*q0 + p2*q3 - p3*q2,
        p0*q2 - p1*q3 + p2*q0 + p3*q1,
        p0*q3 + p1*q2 - p2*q1 + p3*q0,
    ], dtype=np.float64)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate q* = [q0, -q1, -q2, -q3]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q: v' = q ⊗ [0,v] ⊗ q*."""
    v_quat = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    result = quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))
    return result[1:4]


def quat_normalise(q: np.ndarray) -> np.ndarray:
    """Normalise quaternion to unit length."""
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Quaternion kinematic equation: dq/dt = 0.5 * q ⊗ [0, ω].

    Args:
        q: Current quaternion [q0, q1, q2, q3].
        omega: Angular velocity in body frame [p, q, r] (rad/s).
    """
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=np.float64)
    return 0.5 * quat_multiply(q, omega_quat)


# ─────────────────────────────────────────────────────────────────────────────
#  Angular velocity ↔ Euler rate Jacobian
# ─────────────────────────────────────────────────────────────────────────────

def euler_rate_to_body_rate(roll: float, pitch: float, euler_rates: np.ndarray) -> np.ndarray:
    """Convert Euler rates [φ̇, θ̇, ψ̇] to body angular rates [p, q, r].

    Jacobian: ω_body = T(φ, θ) @ Euler_rates
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cp = max(abs(cp), 1e-8) * (1 if cp >= 0 else -1)

    T = np.array([
        [1,  0,    -sp],
        [0,  cr,   sr * cp],
        [0, -sr,   cr * cp],
    ], dtype=np.float64)
    return T @ euler_rates


def body_rate_to_euler_rate(roll: float, pitch: float, omega: np.ndarray) -> np.ndarray:
    """Convert body angular rates [p, q, r] to Euler rates [φ̇, θ̇, ψ̇].

    Inverse Jacobian: Euler_rates = T^{-1}(φ, θ) @ ω_body.
    Note: Singular at θ = ±90° — prefer quaternion representation.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    tp = sp / max(abs(cp), 1e-8) * (1 if cp >= 0 else -1)

    T_inv = np.array([
        [1,  sr * tp,   cr * tp],
        [0,  cr,       -sr],
        [0,  sr / max(abs(cp), 1e-8) * (1 if cp >= 0 else -1),
             cr / max(abs(cp), 1e-8) * (1 if cp >= 0 else -1)],
    ], dtype=np.float64)
    return T_inv @ omega


# ─────────────────────────────────────────────────────────────────────────────
#  Frame conversions
# ─────────────────────────────────────────────────────────────────────────────

def ned_to_enu(v_ned: np.ndarray) -> np.ndarray:
    """Convert NED vector to ENU: [N, E, D] → [E, N, -D]."""
    return np.array([v_ned[1], v_ned[0], -v_ned[2]], dtype=np.float64)


def enu_to_ned(v_enu: np.ndarray) -> np.ndarray:
    """Convert ENU vector to NED: [E, N, U] → [N, E, -U]."""
    return np.array([v_enu[1], v_enu[0], -v_enu[2]], dtype=np.float64)


def body_to_ned(v_body: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate body-frame vector to NED using quaternion."""
    return quat_rotate_vector(q, v_body)


def ned_to_body(v_ned: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate NED vector to body frame using quaternion conjugate."""
    return quat_rotate_vector(quat_conjugate(q), v_ned)


# ─────────────────────────────────────────────────────────────────────────────
#  Wind triangle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WindTriangle:
    """Aerodynamic angles computed from body velocity and wind."""
    airspeed_mps: float        # Total airspeed V_a (m/s)
    alpha_rad: float           # Angle of attack α (rad)
    beta_rad: float            # Sideslip angle β (rad)
    v_air_body: np.ndarray     # Airspeed vector in body frame [u_a, v_a, w_a]


def compute_wind_triangle(
    v_body: np.ndarray,
    wind_ned: np.ndarray,
    q: np.ndarray,
) -> WindTriangle:
    """Compute angle of attack α, sideslip β, and airspeed from body velocity and wind.

    Args:
        v_body: Vehicle velocity in body frame [u, v, w] (m/s).
        wind_ned: Wind velocity in NED frame [w_N, w_E, w_D] (m/s).
        q: Vehicle attitude quaternion [q0, q1, q2, q3].

    Returns:
        WindTriangle with airspeed, α, β in the body frame.
    """
    # Transform wind to body frame
    wind_body = ned_to_body(wind_ned, q)

    # Air-relative velocity in body frame
    v_air = v_body - wind_body
    u_a, v_a, w_a = v_air

    airspeed = float(np.linalg.norm(v_air))

    if airspeed < 1e-3:
        return WindTriangle(airspeed_mps=airspeed, alpha_rad=0.0, beta_rad=0.0, v_air_body=v_air)

    alpha = math.atan2(w_a, max(abs(u_a), 1e-6) * (1 if u_a >= 0 else -1))
    beta = math.asin(max(-1.0, min(1.0, v_a / airspeed)))

    return WindTriangle(
        airspeed_mps=airspeed,
        alpha_rad=alpha,
        beta_rad=beta,
        v_air_body=v_air,
    )
