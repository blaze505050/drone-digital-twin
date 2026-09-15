"""
drone_sdk.math_models.rigid_body
=================================
Generic 6-DOF Rigid Body Dynamics (Newton-Euler Formulation).

Implements the full nonlinear equations of motion for a rigid body
in 3D space using the Newton-Euler formulation in the body-fixed frame:

Translational dynamics (body frame):
    m(v̇_b + ω × v_b) = F_total

Rotational dynamics (body frame):
    I·ω̇ + ω × (I·ω) = τ_total

Kinematic propagation (quaternion):
    q̇ = ½ q ⊗ [0, ω]

Numerical integrators:
    - Semi-implicit Euler (symplectic, energy-preserving for oscillatory systems)
    - Classical RK4 (4th-order accuracy)

The state vector is 13-dimensional:
    x = [pos_ned(3), vel_body(3), quat(4), omega_body(3)]

References:
    - Stevens, Lewis & Johnson, "Aircraft Control and Simulation", 3rd Ed, 2016.
    - Zipfel, "Modeling and Simulation of Aerospace Vehicle Dynamics", 3rd Ed, 2014.

Python version: 3.9+
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

from .frames import (
    quat_conjugate,
    quat_derivative,
    quat_multiply,
    quat_normalise,
    quat_rotate_vector,
    quat_to_dcm,
    quat_to_euler,
)


# ─────────────────────────────────────────────────────────────────────────────
#  State representation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RigidBodyState:
    """Complete 13-DOF rigid body state vector.

    All quantities in NED inertial frame (position) or FRD body frame
    (velocity, angular rates).
    """
    pos_ned: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    vel_body: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    quat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0], dtype=np.float64))
    omega_body: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __post_init__(self) -> None:
        self.pos_ned = np.asarray(self.pos_ned, dtype=np.float64)
        self.vel_body = np.asarray(self.vel_body, dtype=np.float64)
        self.quat = np.asarray(self.quat, dtype=np.float64)
        self.omega_body = np.asarray(self.omega_body, dtype=np.float64)
        self.quat = quat_normalise(self.quat)

    @property
    def vel_ned(self) -> np.ndarray:
        """Velocity in NED frame."""
        return quat_rotate_vector(self.quat, self.vel_body)

    @property
    def euler(self) -> Tuple[float, float, float]:
        """Euler angles (roll, pitch, yaw) in radians."""
        return quat_to_euler(self.quat)

    @property
    def altitude_agl(self) -> float:
        """Altitude above ground level (positive up, from NED z)."""
        return max(0.0, -self.pos_ned[2])

    @property
    def dcm(self) -> np.ndarray:
        """Body-to-NED direction cosine matrix."""
        return quat_to_dcm(self.quat)

    def to_array(self) -> np.ndarray:
        """Flatten to 13-element array [pos(3), vel(3), quat(4), omega(3)]."""
        return np.concatenate([self.pos_ned, self.vel_body, self.quat, self.omega_body])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "RigidBodyState":
        """Reconstruct from 13-element array."""
        return cls(
            pos_ned=arr[0:3].copy(),
            vel_body=arr[3:6].copy(),
            quat=arr[6:10].copy(),
            omega_body=arr[10:13].copy(),
        )

    def copy(self) -> "RigidBodyState":
        return RigidBodyState(
            pos_ned=self.pos_ned.copy(),
            vel_body=self.vel_body.copy(),
            quat=self.quat.copy(),
            omega_body=self.omega_body.copy(),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Inertia parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InertiaParams:
    """Mass and inertia tensor for a rigid body."""
    mass_kg: float = 1.5
    inertia_tensor: np.ndarray = field(
        default_factory=lambda: np.diag([0.035, 0.046, 0.098]).astype(np.float64)
    )

    def __post_init__(self) -> None:
        self.inertia_tensor = np.asarray(self.inertia_tensor, dtype=np.float64)
        if self.inertia_tensor.shape == (3,):
            self.inertia_tensor = np.diag(self.inertia_tensor)

    @property
    def inertia_inv(self) -> np.ndarray:
        """Inverse of the inertia tensor."""
        return np.linalg.inv(self.inertia_tensor)


# ─────────────────────────────────────────────────────────────────────────────
#  External force/torque interface
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExternalWrench:
    """Combined forces and torques acting on the rigid body (body frame)."""
    force_body: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    torque_body: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __post_init__(self) -> None:
        self.force_body = np.asarray(self.force_body, dtype=np.float64)
        self.torque_body = np.asarray(self.torque_body, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
#  Core dynamics equations
# ─────────────────────────────────────────────────────────────────────────────

GRAVITY_MPS2 = 9.80665  # Standard gravity (m/s²)


def compute_derivatives(
    state: RigidBodyState,
    inertia: InertiaParams,
    wrench: ExternalWrench,
    gravity_ned: np.ndarray = np.array([0, 0, GRAVITY_MPS2]),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute state derivatives from Newton-Euler equations.

    Returns:
        (pos_dot, vel_dot, quat_dot, omega_dot) — all in appropriate frames.
    """
    q = state.quat
    v_b = state.vel_body
    w = state.omega_body
    m = inertia.mass_kg
    I = inertia.inertia_tensor

    # 1. Position derivative: ṗ_ned = R_nb · v_body
    R_nb = quat_to_dcm(q)
    pos_dot = R_nb @ v_b

    # 2. Translational dynamics in body frame:
    #    m(v̇_b + ω × v_b) = F_body + R_bn · F_grav
    #    ⟹ v̇_b = F_body/m + R_bn·g - ω × v_b
    R_bn = R_nb.T
    g_body = R_bn @ gravity_ned
    vel_dot = wrench.force_body / m + g_body - np.cross(w, v_b)

    # 3. Quaternion kinematic: q̇ = ½ q ⊗ [0, ω]
    quat_dot = quat_derivative(q, w)

    # 4. Rotational dynamics (Euler's equation):
    #    I·ω̇ = τ - ω × (I·ω)
    Iw = I @ w
    omega_dot = np.linalg.solve(I, wrench.torque_body - np.cross(w, Iw))

    return pos_dot, vel_dot, quat_dot, omega_dot


# ─────────────────────────────────────────────────────────────────────────────
#  Numerical integrators
# ─────────────────────────────────────────────────────────────────────────────

def integrate_semi_implicit_euler(
    state: RigidBodyState,
    inertia: InertiaParams,
    wrench: ExternalWrench,
    dt: float,
    gravity_ned: np.ndarray = np.array([0, 0, GRAVITY_MPS2]),
) -> RigidBodyState:
    """Semi-implicit (symplectic) Euler integration step.

    Updates velocities first, then uses the new velocities to update positions.
    Better energy conservation than explicit Euler for oscillatory dynamics.
    """
    pos_dot, vel_dot, quat_dot, omega_dot = compute_derivatives(
        state, inertia, wrench, gravity_ned
    )

    # Update velocities first (semi-implicit)
    new_omega = state.omega_body + omega_dot * dt
    new_vel = state.vel_body + vel_dot * dt

    # Update positions using NEW velocities
    new_state = state.copy()
    new_state.omega_body = new_omega
    new_state.vel_body = new_vel

    # Recompute position derivative with new velocity
    R_nb = quat_to_dcm(state.quat)
    new_state.pos_ned = state.pos_ned + R_nb @ new_vel * dt

    # Quaternion integration with new omega
    new_quat_dot = quat_derivative(state.quat, new_omega)
    new_state.quat = quat_normalise(state.quat + new_quat_dot * dt)

    return new_state


def integrate_rk4(
    state: RigidBodyState,
    inertia: InertiaParams,
    wrench_fn: Callable[[RigidBodyState], ExternalWrench],
    dt: float,
    gravity_ned: np.ndarray = np.array([0, 0, GRAVITY_MPS2]),
) -> RigidBodyState:
    """Classical 4th-order Runge-Kutta integration step.

    Args:
        wrench_fn: Callable that computes the external wrench for a given state.
                   This allows forces/torques that depend on state (e.g., drag).
    """
    def state_deriv(s: RigidBodyState) -> np.ndarray:
        w = wrench_fn(s)
        pd, vd, qd, od = compute_derivatives(s, inertia, w, gravity_ned)
        return np.concatenate([pd, vd, qd, od])

    y0 = state.to_array()

    k1 = state_deriv(state)
    k2 = state_deriv(RigidBodyState.from_array(y0 + 0.5 * dt * k1))
    k3 = state_deriv(RigidBodyState.from_array(y0 + 0.5 * dt * k2))
    k4 = state_deriv(RigidBodyState.from_array(y0 + dt * k3))

    y_new = y0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    new_state = RigidBodyState.from_array(y_new)
    new_state.quat = quat_normalise(new_state.quat)
    return new_state


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: RigidBodySimulator
# ─────────────────────────────────────────────────────────────────────────────

class RigidBodySimulator:
    """Self-contained 6-DOF rigid body simulator.

    Manages state, inertia, and provides step() for advancing dynamics.
    """

    def __init__(
        self,
        inertia: Optional[InertiaParams] = None,
        state: Optional[RigidBodyState] = None,
        integrator: str = "semi_implicit",
    ) -> None:
        self.inertia = inertia or InertiaParams()
        self.state = state or RigidBodyState()
        self.integrator = integrator
        self.time_s: float = 0.0
        self.step_count: int = 0

    def step(
        self,
        wrench: ExternalWrench,
        dt: float,
        n_substeps: int = 4,
    ) -> RigidBodyState:
        """Advance simulation by dt seconds using n_substeps sub-integration steps."""
        sub_dt = dt / max(1, n_substeps)

        for _ in range(n_substeps):
            if self.integrator == "rk4":
                self.state = integrate_rk4(
                    self.state, self.inertia,
                    wrench_fn=lambda s: wrench,
                    dt=sub_dt,
                )
            else:
                self.state = integrate_semi_implicit_euler(
                    self.state, self.inertia, wrench, sub_dt,
                )

        self.time_s += dt
        self.step_count += 1
        return self.state

    def reset(self, state: Optional[RigidBodyState] = None) -> None:
        """Reset simulator to initial state."""
        self.state = state or RigidBodyState()
        self.time_s = 0.0
        self.step_count = 0
