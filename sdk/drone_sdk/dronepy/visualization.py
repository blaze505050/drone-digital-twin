"""
dronepy.visualization
=====================
High-Quality Engineering Plotting Suite for DronePy Flight Results.
Works seamlessly in Jupyter notebooks and interactive Python sessions.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def _get_plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib is required for DronePy visualization. Install with: pip install matplotlib")


def plot_trajectory(
    result: Any,
    title: str = "3D Flight Trajectory",
    show: bool = True,
) -> Any:
    """Plot 3D spatial flight trajectory and 2D ground track."""
    plt = _get_plt()
    fig = plt.figure(figsize=(12, 5))

    # 3D trajectory
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    # NED to visual: X=East(Y_ned), Y=North(X_ned), Z=Altitude(-Z_ned)
    x_north = result.pos_ned[:, 0]
    y_east = result.pos_ned[:, 1]
    z_alt = result.altitude_agl

    ax1.plot(y_east, x_north, z_alt, label="Trajectory", color="#00d4ff", linewidth=2.0)
    ax1.scatter([y_east[0]], [x_north[0]], [z_alt[0]], color="#00ff88", s=50, label="Start")
    ax1.scatter([y_east[-1]], [x_north[-1]], [z_alt[-1]], color="#ff4757", s=50, label="End")

    ax1.set_xlabel("East (m)")
    ax1.set_ylabel("North (m)")
    ax1.set_zlabel("Altitude AGL (m)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2D Top-Down Ground Track
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(y_east, x_north, color="#3b82f6", linewidth=2.0)
    ax2.scatter([y_east[0]], [x_north[0]], color="#00ff88", s=60, label="Takeoff")
    ax2.scatter([y_east[-1]], [x_north[-1]], color="#ff4757", s=60, label="Landing")
    ax2.set_xlabel("East (m)")
    ax2.set_ylabel("North (m)")
    ax2.set_title("Ground Track (Top-Down)")
    ax2.axis("equal")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_attitude(result: Any, show: bool = True) -> Any:
    """Plot Roll, Pitch, and Yaw attitude angles versus time."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(result.time, result.euler_deg[:, 0], label="Roll (deg)", color="#ff8a3d", linewidth=1.5)
    ax.plot(result.time, result.euler_deg[:, 1], label="Pitch (deg)", color="#22d97f", linewidth=1.5)
    ax.plot(result.time, result.euler_deg[:, 2], label="Yaw (deg)", color="#00d4ff", linewidth=1.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Vehicle Attitude (Euler Angles)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_velocity(result: Any, show: bool = True) -> Any:
    """Plot translational velocity components and total airspeed."""
    plt = _get_plt()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # NED components
    ax1.plot(result.time, result.vel_ned[:, 0], label="v_north", color="#3b82f6")
    ax1.plot(result.time, result.vel_ned[:, 1], label="v_east", color="#22d97f")
    ax1.plot(result.time, -result.vel_ned[:, 2], label="v_up", color="#a855f7")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.set_title("Inertial NED Velocity")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Airspeed and Altitude
    ax2.plot(result.time, result.airspeed, label="Airspeed", color="#00d4ff", linewidth=1.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Airspeed (m/s)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_motor_rpm(result: Any, show: bool = True) -> Any:
    """Plot individual motor RPM trajectories."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(10, 4.5))

    num_m = result.motor_rpms.shape[1]
    colors = ["#00d4ff", "#ff4757", "#22d97f", "#ff8a3d", "#a855f7", "#3b82f6", "#eab308", "#ec4899"]

    for i in range(num_m):
        c = colors[i % len(colors)]
        ax.plot(result.time, result.motor_rpms[:, i], label=f"Motor {i+1}", color=c, linewidth=1.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rotor Speed (RPM)")
    ax.set_title("Multirotor Actuator Speeds (RPM)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_power(result: Any, show: bool = True) -> Any:
    """Plot electrical power draw (W) and discharge current (A)."""
    plt = _get_plt()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(result.time, result.battery_power, color="#ff8a3d", linewidth=1.8, label="Electrical Power")
    ax1.set_ylabel("Power (Watts)")
    ax1.set_title("Propulsion Power & Current Consumption")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    ax2.plot(result.time, result.battery_current, color="#ff4757", linewidth=1.8, label="Pack Current")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Current (Amps)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_energy(result: Any, show: bool = True) -> Any:
    """Plot cumulative energy consumed (Wh) and battery State of Charge (%)."""
    plt = _get_plt()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)

    ax1.plot(result.time, result.battery_energy_wh, color="#00d4ff", linewidth=2.0, label="Energy Consumed")
    ax1.set_ylabel("Energy (Watt-hours)")
    ax1.set_title("Cumulative Battery Energy & State of Charge")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2.plot(result.time, result.battery_soc * 100.0, color="#22d97f", linewidth=2.0, label="SOC")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("State of Charge (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_forces(result: Any, show: bool = True) -> Any:
    """Plot total thrust and body aerodynamic forces."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(result.time, result.total_thrust, label="Total Thrust (N)", color="#00d4ff", linewidth=2.0)
    ax.plot(result.time, result.forces_body[:, 0], label="F_x (Forward)", color="#ff8a3d", linewidth=1.2)
    ax.plot(result.time, result.forces_body[:, 1], label="F_y (Lateral)", color="#22d97f", linewidth=1.2)
    ax.plot(result.time, result.forces_body[:, 2], label="F_z (Vertical)", color="#a855f7", linewidth=1.2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (Newtons)")
    ax.set_title("Aerodynamic and Propulsion Forces")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_moments(result: Any, show: bool = True) -> Any:
    """Plot roll, pitch, and yaw body torques."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(result.time, result.moments_body[:, 0], label="tau_roll (N·m)", color="#ff8a3d", linewidth=1.5)
    ax.plot(result.time, result.moments_body[:, 1], label="tau_pitch (N·m)", color="#22d97f", linewidth=1.5)
    ax.plot(result.time, result.moments_body[:, 2], label="tau_yaw (N·m)", color="#00d4ff", linewidth=1.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (N·m)")
    ax.set_title("Body Control Torques")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_dashboard(result: Any, show: bool = True) -> Any:
    """4-panel overview summary plot: Trajectory, Altitude, Motor RPM, Power."""
    plt = _get_plt()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 8))

    # 1. Altitude vs Time
    ax1.plot(result.time, result.altitude_agl, color="#00d4ff", linewidth=2.0)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Altitude AGL (m)")
    ax1.set_title("Altitude Profile")
    ax1.grid(True, alpha=0.3)

    # 2. Attitude Roll & Pitch
    ax2.plot(result.time, result.euler_deg[:, 0], label="Roll", color="#ff8a3d")
    ax2.plot(result.time, result.euler_deg[:, 1], label="Pitch", color="#22d97f")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Degrees")
    ax2.set_title("Attitude Angles")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Motor RPMs
    for i in range(result.motor_rpms.shape[1]):
        ax3.plot(result.time, result.motor_rpms[:, i], label=f"M{i+1}", alpha=0.85)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("RPM")
    ax3.set_title("Actuator RPMs")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="lower right")

    # 4. Energy Consumed
    ax4.plot(result.time, result.battery_energy_wh, color="#a855f7", linewidth=2.0)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Energy (Wh)")
    ax4.set_title(f"Energy: {result.battery_energy_wh[-1]:.2f} Wh consumed")
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    if show:
        plt.show()
    return fig
