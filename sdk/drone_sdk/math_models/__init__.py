"""
drone_sdk.math_models
=====================
UAV Mathematical Models Library.

Complete Newton-Euler rigid body dynamics for all major UAV configurations:

  - **Quadrotor** (X and + configurations)
  - **Hexarotor** (flat X, flat star, coaxial Y6)
  - **Octorotor** (flat X8, coaxial X8)
  - **Fixed-Wing** (conventional, with stability derivatives and trim solver)
  - **VTOL Tilt-Rotor** (quadplane hybrid with transition corridor)

Plus supporting modules:

  - **Frames** — coordinate frame transformations, quaternion algebra, wind triangle
  - **Rigid Body** — generic 6-DOF Newton-Euler dynamics with RK4 and semi-implicit integrators

Python version: 3.9+
"""
from __future__ import annotations

from .frames import (
    euler_to_quat,
    quat_to_euler,
    euler_to_dcm,
    dcm_to_euler,
    quat_to_dcm,
    dcm_to_quat,
    quat_multiply,
    quat_conjugate,
    quat_rotate_vector,
    quat_normalise,
    quat_derivative,
    euler_rate_to_body_rate,
    body_rate_to_euler_rate,
    ned_to_enu,
    enu_to_ned,
    body_to_ned,
    ned_to_body,
    compute_wind_triangle,
    WindTriangle,
)

from .rigid_body import (
    RigidBodyState,
    InertiaParams,
    ExternalWrench,
    RigidBodySimulator,
    compute_derivatives,
    integrate_semi_implicit_euler,
    integrate_rk4,
    GRAVITY_MPS2,
)

from .quadrotor import (
    QuadrotorModel,
    QuadrotorParams,
    QuadConfig,
    mixing_matrix_x,
    mixing_matrix_plus,
    ground_effect_factor,
)

from .hexarotor import (
    HexarotorModel,
    HexarotorParams,
    HexConfig,
    mixing_matrix_hex_flat,
    mixing_matrix_coaxial_y6,
)

from .octorotor import (
    OctorotorModel,
    OctorotorParams,
    OctoConfig,
    mixing_matrix_flat_x8,
    mixing_matrix_coaxial_x8,
)

from .fixed_wing import (
    FixedWingModel,
    FixedWingParams,
    ControlSurfaces,
)

from .vtol_tiltrotor import (
    VTOLTiltrotorModel,
    VTOLParams,
    VTOLFlightPhase,
)


__all__ = [
    # Frames
    "euler_to_quat", "quat_to_euler", "euler_to_dcm", "dcm_to_euler",
    "quat_to_dcm", "dcm_to_quat", "quat_multiply", "quat_conjugate",
    "quat_rotate_vector", "quat_normalise", "quat_derivative",
    "euler_rate_to_body_rate", "body_rate_to_euler_rate",
    "ned_to_enu", "enu_to_ned", "body_to_ned", "ned_to_body",
    "compute_wind_triangle", "WindTriangle",
    # Rigid body
    "RigidBodyState", "InertiaParams", "ExternalWrench", "RigidBodySimulator",
    "compute_derivatives", "integrate_semi_implicit_euler", "integrate_rk4",
    "GRAVITY_MPS2",
    # Quadrotor
    "QuadrotorModel", "QuadrotorParams", "QuadConfig",
    "mixing_matrix_x", "mixing_matrix_plus", "ground_effect_factor",
    # Hexarotor
    "HexarotorModel", "HexarotorParams", "HexConfig",
    "mixing_matrix_hex_flat", "mixing_matrix_coaxial_y6",
    # Octorotor
    "OctorotorModel", "OctorotorParams", "OctoConfig",
    "mixing_matrix_flat_x8", "mixing_matrix_coaxial_x8",
    # Fixed-wing
    "FixedWingModel", "FixedWingParams", "ControlSurfaces",
    # VTOL
    "VTOLTiltrotorModel", "VTOLParams", "VTOLFlightPhase",
]
