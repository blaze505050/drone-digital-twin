"""
dronepy.dispersion
==================
Dispersion Analysis, Covariance Ellipses, and Heatmap Visualizations for DronePy.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np


def _get_plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib is required for DronePy dispersion analysis.")


def plot_landing_dispersion(mc_result: Any, target_pos: Optional[np.ndarray] = None, show: bool = True) -> Any:
    """Plot 2D landing dispersion scatter with CEP50 and CEP95 confidence circles."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 8))

    x_north = mc_result.landing_positions_ned[:, 0]
    y_east = mc_result.landing_positions_ned[:, 1]

    # Target
    t_east = target_pos[1] if target_pos is not None else 0.0
    t_north = target_pos[0] if target_pos is not None else 0.0

    # Scatter points
    ax.scatter(y_east, x_north, color="#00d4ff", alpha=0.55, edgecolors="none", s=25, label=f"Landings (N={mc_result.runs})")
    ax.scatter([t_east], [t_north], color="#ff4757", s=120, marker="X", label="Target Landing Pad", zorder=5)

    # Mean point
    mean_e = float(np.mean(y_east))
    mean_n = float(np.mean(x_north))
    ax.scatter([mean_e], [mean_n], color="#00ff88", s=80, marker="o", label="Centroid (Mean)", zorder=4)

    # Compute distances from centroid
    distances = np.sqrt((y_east - mean_e) ** 2 + (x_north - mean_n) ** 2)
    cep50 = float(np.percentile(distances, 50.0))
    cep95 = float(np.percentile(distances, 95.0))

    # Circles
    circle_50 = plt.Circle((mean_e, mean_n), cep50, color="#22d97f", fill=False, linestyle="--", linewidth=1.5, label=f"CEP 50% ({cep50:.2f} m)")
    circle_95 = plt.Circle((mean_e, mean_n), cep95, color="#ff8a3d", fill=False, linestyle=":", linewidth=2.0, label=f"CEP 95% ({cep95:.2f} m)")
    ax.add_patch(circle_50)
    ax.add_patch(circle_95)

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"Landing Position Dispersion (Monte Carlo, N={mc_result.runs})")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_trajectory_envelopes(mc_result: Any, show: bool = True) -> Any:
    """Plot bundled 3D trajectory traces across stochastic Monte Carlo realizations."""
    plt = _get_plt()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    for traj in mc_result.trajectories_sample:
        # X=East, Y=North, Z=Altitude
        ax.plot(traj[:, 1], traj[:, 0], -traj[:, 2], color="#00d4ff", alpha=0.35, linewidth=1.0)

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Altitude AGL (m)")
    ax.set_title(f"Stochastic Trajectory Envelopes (Sample of {len(mc_result.trajectories_sample)} runs)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if show:
        plt.show()
    return fig
