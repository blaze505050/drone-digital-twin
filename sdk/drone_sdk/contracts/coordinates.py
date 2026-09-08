"""
contracts.coordinates
=====================
Aerospace Coordinate Frame and Kinematic Standards.

Defines the mathematical conventions and invariants for navigation and body frames
used across all 28 digital twin modules:

1. Navigation Frame: North-East-Down (NED)
   - +X: Geodetic True North
   - +Y: Geodetic East
   - +Z: Down towards Earth center
   - Gravity vector in NED: g_ned = [0.0, 0.0, +9.80665] m/s^2

2. Body Frame: Forward-Right-Down (FRD)
   - +X_b: Longitudinal nose axis
   - +Y_b: Starboard / right lateral axis
   - +Z_b: Ventral / downward normal axis

3. Attitude Representation:
   - Hamilton unit quaternion: q = [q_w, q_x, q_y, q_z]
   - Rotation maps body vectors to navigation frame: v_ned = R(q) @ v_body
   - Passive coordinate transformation: v_body = R(q)^T @ v_ned

4. Accelerometer Specific Force:
   - Accelerometers measure specific force: f_b = R(q)^T (a_ned - g_ned)
   - At static level rest: a_ned = 0 => f_b = [0, 0, -9.80665] m/s^2 (upward normal support)

Python version: 3.9+
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# Standard Earth gravitational acceleration (WGS 84 nominal)
STANDARD_GRAVITY_MPS2: float = 9.80665
GRAVITY_NED: np.ndarray = np.array([0.0, 0.0, STANDARD_GRAVITY_MPS2], dtype=np.float64)


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Convert Hamilton quaternion [qw, qx, qy, qz] to body-to-NED direction cosine matrix R.

    Parameters
    ----------
    q : np.ndarray
        Unit quaternion [qw, qx, qy, qz] where qw is scalar part.

    Returns
    -------
    np.ndarray
        3x3 orthogonal rotation matrix R such that v_ned = R @ v_body.
    """
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    q0, q1, q2, q3 = q / q_norm

    return np.array([
        [1.0 - 2.0 * (q2 * q2 + q3 * q3), 2.0 * (q1 * q2 - q0 * q3),       2.0 * (q1 * q3 + q0 * q2)],
        [2.0 * (q1 * q2 + q0 * q3),       1.0 - 2.0 * (q1 * q1 + q3 * q3), 2.0 * (q2 * q3 - q0 * q1)],
        [2.0 * (q1 * q3 - q0 * q2),       2.0 * (q2 * q3 + q0 * q1),       1.0 - 2.0 * (q1 * q1 + q2 * q2)],
    ], dtype=np.float64)


def rot_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to Hamilton quaternion [qw, qx, qy, qz]."""
    tr = np.trace(R)
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def specific_force_from_acceleration(acc_ned: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Calculate measured body specific force from inertial NED acceleration and attitude."""
    R = quat_to_rot_matrix(q)
    return R.T @ (acc_ned - GRAVITY_NED)


def acceleration_from_specific_force(f_body: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Calculate inertial NED acceleration from body specific force and attitude."""
    R = quat_to_rot_matrix(q)
    return R @ f_body + GRAVITY_NED
