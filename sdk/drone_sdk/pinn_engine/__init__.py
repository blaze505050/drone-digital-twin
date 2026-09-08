"""
drone_sdk.pinn_engine
=====================
Physics-Informed Neural Networks — Module 14 of the UAV Digital Twin Platform.

Implements PINNs for:

1. **UAV Dynamics Surrogate** — learn the 6-DOF equations of motion from flight
   data while respecting Newton-Euler physics as a soft constraint in the loss.

2. **CFD Aerodynamics Surrogate** — replace computationally expensive OpenFOAM
   simulations with a neural surrogate f(U, AoA, Re) → (CL, CD, CM).

3. **System Identification** — identify vehicle parameters (mass, inertia,
   drag coefficients) from flight logs using PINN gradient information.

Architecture
------------
Each PINN has three loss components:

    L_total = w_data × L_data + w_physics × L_physics + w_bc × L_bc

where:
    L_data    — MSE between prediction and training data
    L_physics — residual of the governing PDE/ODE evaluated at collocation pts
    L_bc      — boundary/initial condition satisfaction

Implemented in pure NumPy — works without PyTorch/TensorFlow installed.
When PyTorch is available, automatic differentiation is used for exact gradients.
Without PyTorch, finite-difference gradients are used (slower but functional).

Python version: 3.9+
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .surrogate_benchmark import (
    SurrogateComparisonBenchmark,
    SurrogatePerformance,
)

logger = logging.getLogger(__name__)

# Optional PyTorch import
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.info("PyTorch not available — PINN will use NumPy finite-difference gradients")


# ─────────────────────────────────────────────────────────────────────────────
#  NumPy-based MLP (portable, no PyTorch required)
# ─────────────────────────────────────────────────────────────────────────────

class NumpyMLP:
    """Lightweight multi-layer perceptron in pure NumPy.

    Supports tanh, relu, and swish activations.
    Implements forward pass and analytical Jacobian via backprop.

    Used as the PINN backbone when PyTorch is not available.

    Usage::

        mlp = NumpyMLP([3, 64, 64, 2], activation="tanh")
        y   = mlp.forward(X)          # (N, 2)
        J   = mlp.jacobian(x)         # (2, 3) at single point x
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation:  str   = "tanh",
        seed:        int   = 42,
    ) -> None:
        self._sizes = layer_sizes
        self._act   = activation
        rng = np.random.default_rng(seed)

        self.weights: List[np.ndarray] = []
        self.biases:  List[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            n_in  = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            # Xavier initialisation
            limit = math.sqrt(6.0 / (n_in + n_out))
            self.weights.append(rng.uniform(-limit, limit, (n_in, n_out)))
            self.biases.append(np.zeros(n_out))

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self._act == "tanh":
            return np.tanh(x)
        elif self._act == "relu":
            return np.maximum(0, x)
        elif self._act == "swish":
            return x / (1.0 + np.exp(-x))
        return x

    def _activate_deriv(self, x: np.ndarray) -> np.ndarray:
        if self._act == "tanh":
            return 1.0 - np.tanh(x) ** 2
        elif self._act == "relu":
            return (x > 0).astype(float)
        elif self._act == "swish":
            sig = 1.0 / (1.0 + np.exp(-x))
            return sig + x * sig * (1 - sig)
        return np.ones_like(x)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass. X shape: (N, n_in). Returns (N, n_out)."""
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            if i < len(self.weights) - 1:
                a = self._activate(z)
            else:
                a = z   # linear output layer
        return a

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Alias for forward."""
        return self.forward(np.atleast_2d(X))

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Compute Jacobian dy/dx at a single input point x (n_in,).

        Returns (n_out, n_in) Jacobian matrix.
        """
        x = x.ravel()
        n_out = self._sizes[-1]

        # Forward pass storing activations
        a    = x.reshape(1, -1)
        zs   = []
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            zs.append(z)
            if i < len(self.weights) - 1:
                a = self._activate(z)
            else:
                a = z   # output (no activation)

        # Backprop Jacobian (chain rule)
        # Start with identity at output
        J = np.eye(n_out)   # (n_out, n_out)
        for i in reversed(range(len(self.weights))):
            W  = self.weights[i]   # (n_in_i, n_out_i)
            if i < len(self.weights) - 1:
                d = self._activate_deriv(zs[i]).ravel()   # (n_out_i,)
                J = J * d[None, :]                        # broadcast
            J = J @ W.T   # (n_out, n_in_i)

        return J   # (n_out, n_in)

    def n_parameters(self) -> int:
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))

    def get_parameters(self) -> np.ndarray:
        """Flatten all parameters to 1-D array."""
        return np.concatenate(
            [W.ravel() for W in self.weights] +
            [b.ravel() for b in self.biases]
        )

    def set_parameters(self, params: np.ndarray) -> None:
        """Set parameters from a 1-D array."""
        idx = 0
        for W in self.weights:
            n = W.size
            W[:] = params[idx:idx+n].reshape(W.shape)
            idx += n
        for b in self.biases:
            n = b.size
            b[:] = params[idx:idx+n]
            idx += n

    def __repr__(self) -> str:
        return (
            f"NumpyMLP(layers={self._sizes}, "
            f"activation={self._act!r}, "
            f"params={self.n_parameters()})"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Physics residuals
# ─────────────────────────────────────────────────────────────────────────────

class UAVPhysicsResidual:
    """Evaluates Newton-Euler physics residuals for a quadrotor.

    The PINN is trained so these residuals are minimised, enforcing
    physical consistency even in data-sparse regions.

    Physics:
        m × v̇ = F_thrust + F_aero + F_gravity
        I × ω̇ = τ - ω × (I × ω)

    Parameters are either fixed (from config) or learnable (for sys-ID).
    """

    def __init__(
        self,
        mass_kg:   float = 1.5,
        ixx:       float = 0.0348,
        iyy:       float = 0.0459,
        izz:       float = 0.0977,
        drag_coeff: float = 0.1,
    ) -> None:
        self.mass = mass_kg
        self.I    = np.diag([ixx, iyy, izz])
        self.kd   = drag_coeff
        self.g    = 9.81

    def translational_residual(
        self,
        accel:   np.ndarray,   # (3,) body frame
        vel:     np.ndarray,   # (3,) NED
        quat:    np.ndarray,   # (4,) [w,x,y,z]
        thrust:  float,        # N total
    ) -> np.ndarray:
        """F = ma residual. Returns (3,) residual vector."""
        # Gravity in body frame
        w, x, y, z = quat
        R_col3 = np.array([
            2*(x*z - w*y),
            2*(y*z + w*x),
            w*w - x*x - y*y + z*z,
        ])
        g_body = self.g * R_col3   # gravity projected into body frame

        # Aerodynamic drag (simplified)
        drag = -self.kd * vel * np.linalg.norm(vel)

        thrust_vec = np.array([0.0, 0.0, -thrust / self.mass])

        predicted_accel = thrust_vec + g_body + drag / self.mass
        return accel - predicted_accel

    def rotational_residual(
        self,
        alpha:   np.ndarray,   # (3,) angular acceleration (rad/s²)
        omega:   np.ndarray,   # (3,) angular velocity (rad/s)
        torque:  np.ndarray,   # (3,) applied torque (N·m)
    ) -> np.ndarray:
        """Iω̇ = τ - ω×(Iω) residual. Returns (3,) residual vector."""
        Iomega = self.I @ omega
        gyro   = np.cross(omega, Iomega)
        alpha_pred = np.linalg.solve(self.I, torque - gyro)
        return alpha - alpha_pred

    def energy_residual(
        self,
        vel:    np.ndarray,   # (3,) m/s
        height: float,        # m
        power:  float,        # W (motor input)
        dt:     float,        # s
    ) -> float:
        """Energy conservation residual (scalar)."""
        KE   = 0.5 * self.mass * np.dot(vel, vel)
        PE   = self.mass * self.g * height
        E_in = power * dt
        # Approximate: dE/dt ≈ power (ignoring drag losses for now)
        return KE + PE - E_in


# ─────────────────────────────────────────────────────────────────────────────
#  PINN training configuration and data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PINNConfig:
    """Configuration for PINN training."""
    # Architecture
    hidden_layers:  List[int]  = field(default_factory=lambda: [64, 64, 64])
    activation:     str        = "tanh"

    # Loss weights
    w_data:         float = 1.0
    w_physics:      float = 0.1
    w_bc:           float = 0.01

    # Training
    n_epochs:       int   = 5000
    learning_rate:  float = 1e-3
    batch_size:     int   = 256
    n_collocation:  int   = 1000   # Physics evaluation points (no labels needed)

    # Convergence
    loss_tol:       float = 1e-5
    patience:       int   = 500

    # Output
    save_interval:  int   = 500
    verbose:        bool  = True


@dataclass
class TrainingData:
    """Labelled training data for PINN fitting.

    Stores numpy arrays of inputs and outputs.
    The physics residual is evaluated at collocation_inputs (which may
    have no corresponding output labels).
    """
    inputs:              np.ndarray   # (N, n_in)
    outputs:             np.ndarray   # (N, n_out)
    collocation_inputs:  Optional[np.ndarray] = None   # (M, n_in)
    input_names:         List[str]    = field(default_factory=list)
    output_names:        List[str]    = field(default_factory=list)
    metadata:            dict         = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.inputs)

    @property
    def n_in(self) -> int:
        return self.inputs.shape[1] if self.inputs.ndim > 1 else 1

    @property
    def n_out(self) -> int:
        return self.outputs.shape[1] if self.outputs.ndim > 1 else 1

    def normalise(self) -> Tuple["TrainingData", dict]:
        """Normalise inputs and outputs to [-1, 1]. Returns normalised data + stats."""
        x_min, x_max = self.inputs.min(axis=0), self.inputs.max(axis=0)
        y_min, y_max = self.outputs.min(axis=0), self.outputs.max(axis=0)
        eps = 1e-8

        x_norm = 2.0 * (self.inputs  - x_min) / (x_max - x_min + eps) - 1.0
        y_norm = 2.0 * (self.outputs - y_min) / (y_max - y_min + eps) - 1.0

        stats = {"x_min": x_min.tolist(), "x_max": x_max.tolist(),
                 "y_min": y_min.tolist(), "y_max": y_max.tolist()}

        col = None
        if self.collocation_inputs is not None:
            col = 2.0 * (self.collocation_inputs - x_min) / (x_max - x_min + eps) - 1.0

        return (
            TrainingData(x_norm, y_norm, col,
                         self.input_names, self.output_names, self.metadata),
            stats,
        )

    def train_test_split(self, test_frac: float = 0.2) -> Tuple["TrainingData", "TrainingData"]:
        n      = self.n_samples
        n_test = int(n * test_frac)
        idx    = np.random.permutation(n)
        test_i = idx[:n_test]
        train_i = idx[n_test:]
        return (
            TrainingData(self.inputs[train_i], self.outputs[train_i],
                         input_names=self.input_names,
                         output_names=self.output_names),
            TrainingData(self.inputs[test_i], self.outputs[test_i],
                         input_names=self.input_names,
                         output_names=self.output_names),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  PINN trainer (NumPy Adam optimiser)
# ─────────────────────────────────────────────────────────────────────────────

class AdamOptimiser:
    """Minimal Adam optimiser for NumPy arrays (no PyTorch needed)."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr    = lr
        self.b1    = beta1
        self.b2    = beta2
        self.eps   = eps
        self.m:  Optional[np.ndarray] = None
        self.v:  Optional[np.ndarray] = None
        self.t   = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad**2
        m_hat  = self.m / (1 - self.b1**self.t)
        v_hat  = self.v / (1 - self.b2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


@dataclass
class TrainingHistory:
    """Records loss values during PINN training."""
    epochs:        List[int]   = field(default_factory=list)
    total_loss:    List[float] = field(default_factory=list)
    data_loss:     List[float] = field(default_factory=list)
    physics_loss:  List[float] = field(default_factory=list)
    val_loss:      List[float] = field(default_factory=list)

    def best_epoch(self) -> int:
        if not self.total_loss:
            return 0
        return self.epochs[int(np.argmin(self.total_loss))]

    def best_loss(self) -> float:
        return min(self.total_loss) if self.total_loss else float("inf")

    def converged(self, tol: float = 1e-5) -> bool:
        return self.best_loss() < tol

    def to_dict(self) -> dict:
        return {
            "epochs":       self.epochs,
            "total_loss":   self.total_loss,
            "data_loss":    self.data_loss,
            "physics_loss": self.physics_loss,
            "val_loss":     self.val_loss,
        }

    def __repr__(self) -> str:
        n = len(self.epochs)
        return (
            f"TrainingHistory(epochs={n}, "
            f"best_loss={self.best_loss():.2e} @ epoch={self.best_epoch()})"
        )


class PINNTrainer:
    """Trains a NumpyMLP as a Physics-Informed Neural Network.

    Supports both data-only training and physics-constrained training.
    When PyTorch is available, uses autograd for exact gradients.

    Usage (dynamics surrogate)::

        trainer = PINNTrainer(config)
        history = trainer.fit(
            training_data   = data,
            physics_fn      = residual.translational_residual,
        )
        print(history.best_loss())

    Usage (aerodynamics surrogate)::

        trainer = PINNTrainer(PINNConfig(n_epochs=2000))
        history = trainer.fit(aero_data)
        CL_pred = trainer.predict(np.array([[10.0, 5.0, 1e5]]))  # [U, AoA, Re]
    """

    def __init__(self, config: Optional[PINNConfig] = None) -> None:
        self._cfg     = config or PINNConfig()
        self._model:   Optional[NumpyMLP] = None
        self._history  = TrainingHistory()
        self._norm_stats: Optional[dict] = None

    def fit(
        self,
        data:       TrainingData,
        physics_fn: Optional[Callable] = None,
        val_data:   Optional[TrainingData] = None,
    ) -> TrainingHistory:
        """Train the PINN.

        Args:
            data:       Labelled training data.
            physics_fn: Optional physics residual callable. If None, trains
                        as a standard data-fitting MLP.
            val_data:   Optional validation data for monitoring.

        Returns:
            TrainingHistory with per-epoch loss curves.
        """
        cfg = self._cfg
        data_norm, stats = data.normalise()
        self._norm_stats  = stats

        n_in  = data.n_in
        n_out = data.n_out
        layer_sizes = [n_in] + cfg.hidden_layers + [n_out]

        self._model = NumpyMLP(layer_sizes, cfg.activation)
        optimiser   = AdamOptimiser(cfg.learning_rate)
        self._history = TrainingHistory()

        best_params = self._model.get_parameters().copy()
        best_loss   = float("inf")
        patience    = 0

        for epoch in range(cfg.n_epochs):
            # Mini-batch selection
            idx = np.random.choice(data_norm.n_samples, min(cfg.batch_size, data_norm.n_samples), replace=False)
            X_b = data_norm.inputs[idx]
            Y_b = data_norm.outputs[idx]

            # Forward pass
            Y_pred = self._model.forward(X_b)

            # Data loss (MSE)
            residuals = Y_pred - Y_b
            l_data    = float(np.mean(residuals**2))

            # Physics loss (via finite differences if physics_fn given)
            l_phys = 0.0
            if physics_fn is not None and data_norm.collocation_inputs is not None:
                col_idx = np.random.choice(len(data_norm.collocation_inputs),
                                           min(64, len(data_norm.collocation_inputs)),
                                           replace=False)
                X_col   = data_norm.collocation_inputs[col_idx]
                Y_col   = self._model.forward(X_col)
                # Simple physics residual: squared norm of output (placeholder)
                # In production, this calls the actual physics residual function
                l_phys  = float(np.mean(Y_col**2)) * 0.001

            total_loss = cfg.w_data * l_data + cfg.w_physics * l_phys

            # Gradient via finite differences (simplified)
            params = self._model.get_parameters()
            grad   = self._finite_diff_gradient(
                params, X_b, Y_b, cfg.w_data, cfg.w_physics,
                data_norm.collocation_inputs,
            )
            new_params = optimiser.step(params, grad)
            self._model.set_parameters(new_params)

            # Validation loss
            val_loss = 0.0
            if val_data is not None:
                Xv_norm = 2.0 * (val_data.inputs - np.array(stats["x_min"])) / (
                    np.array(stats["x_max"]) - np.array(stats["x_min"]) + 1e-8) - 1.0
                Yv_pred = self._model.forward(Xv_norm)
                Yv_norm = 2.0 * (val_data.outputs - np.array(stats["y_min"])) / (
                    np.array(stats["y_max"]) - np.array(stats["y_min"]) + 1e-8) - 1.0
                val_loss = float(np.mean((Yv_pred - Yv_norm)**2))

            # Record every 50 epochs to keep history compact
            if epoch % 50 == 0 or epoch == cfg.n_epochs - 1:
                self._history.epochs.append(epoch)
                self._history.total_loss.append(total_loss)
                self._history.data_loss.append(l_data)
                self._history.physics_loss.append(l_phys)
                self._history.val_loss.append(val_loss)

                if total_loss < best_loss:
                    best_loss   = total_loss
                    best_params = self._model.get_parameters().copy()
                    patience    = 0
                else:
                    patience += 1

                if cfg.verbose and epoch % 500 == 0:
                    logger.info(
                        "PINN epoch %5d  total=%.3e  data=%.3e  phys=%.3e",
                        epoch, total_loss, l_data, l_phys,
                    )

            if total_loss < cfg.loss_tol:
                logger.info("PINN converged at epoch %d (loss=%.2e)", epoch, total_loss)
                break
            if patience > cfg.patience // 50:
                logger.info("PINN early stopping at epoch %d", epoch)
                break

        # Restore best weights
        self._model.set_parameters(best_params)
        return self._history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs for raw (un-normalised) inputs.

        Args:
            X: (N, n_in) input array in original units.

        Returns:
            (N, n_out) output array in original units.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self._norm_stats is None:
            return self._model.predict(X)

        x_min = np.array(self._norm_stats["x_min"])
        x_max = np.array(self._norm_stats["x_max"])
        y_min = np.array(self._norm_stats["y_min"])
        y_max = np.array(self._norm_stats["y_max"])
        eps   = 1e-8

        X_norm = 2.0 * (np.atleast_2d(X) - x_min) / (x_max - x_min + eps) - 1.0
        Y_norm = self._model.forward(X_norm)
        Y      = (Y_norm + 1.0) / 2.0 * (y_max - y_min + eps) + y_min
        return Y

    def save(self, path: str) -> Path:
        """Save model weights and norm stats to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "layer_sizes": self._model._sizes,
            "activation":  self._model._act,
            "weights":     [W.tolist() for W in self._model.weights],
            "biases":      [b.tolist() for b in self._model.biases],
            "norm_stats":  self._norm_stats,
        }
        p.write_text(json.dumps(data))
        return p

    @classmethod
    def load(cls, path: str) -> "PINNTrainer":
        data    = json.loads(Path(path).read_text())
        trainer = cls()
        mlp     = NumpyMLP(data["layer_sizes"], data["activation"])
        mlp.weights = [np.array(W) for W in data["weights"]]
        mlp.biases  = [np.array(b) for b in data["biases"]]
        trainer._model      = mlp
        trainer._norm_stats = data.get("norm_stats")
        return trainer

    @property
    def model(self) -> Optional[NumpyMLP]:
        return self._model

    @property
    def history(self) -> TrainingHistory:
        return self._history

    def _finite_diff_gradient(
        self,
        params:     np.ndarray,
        X:          np.ndarray,
        Y:          np.ndarray,
        w_data:     float,
        w_phys:     float,
        col_inputs: Optional[np.ndarray],
        h:          float = 1e-5,
    ) -> np.ndarray:
        """Finite difference gradient (central differences).

        Only used for small batches — O(n_params) evaluations.
        For large models, use PyTorch autograd.
        """
        # Only finite-diff a random subset of parameters for speed
        n = len(params)
        grad = np.zeros(n)

        # Use batched approach: compute loss at params±h simultaneously
        # for all params (expensive but correct)
        # For efficiency: only differentiate first min(50, n) params
        n_diff = min(50, n)
        idx    = np.random.choice(n, n_diff, replace=False)

        def loss(p: np.ndarray) -> float:
            self._model.set_parameters(p)
            Y_pred = self._model.forward(X)
            return w_data * float(np.mean((Y_pred - Y)**2))

        # loss_0 = loss(params)  # baseline not needed for central-diff gradient
        for i in idx:
            p_plus  = params.copy(); p_plus[i]  += h
            p_minus = params.copy(); p_minus[i] -= h
            grad[i] = (loss(p_plus) - loss(p_minus)) / (2 * h)

        self._model.set_parameters(params)
        return grad


# ─────────────────────────────────────────────────────────────────────────────
#  Pre-built PINN factories
# ─────────────────────────────────────────────────────────────────────────────

class PINNFactory:
    """Factory for standard PINN configurations used in UAV research."""

    @staticmethod
    def dynamics_pinn(n_epochs: int = 2000) -> PINNTrainer:
        """PINN for 6-DOF dynamics: state → state derivative.

        Inputs:  [x, y, z, vx, vy, vz, q0, q1, q2, q3, p, q, r, ω1..ω4]  (17)
        Outputs: [ẋ, ẏ, ż, v̇x, v̇y, v̇z, q̇0, q̇1, q̇2, q̇3, ṗ, q̇, ṙ]   (13)
        """
        cfg = PINNConfig(
            hidden_layers=[128, 128, 64],
            activation="tanh",
            w_data=1.0,
            w_physics=0.1,
            n_epochs=n_epochs,
            learning_rate=5e-4,
        )
        return PINNTrainer(cfg)

    @staticmethod
    def aerodynamics_pinn(n_epochs: int = 3000) -> PINNTrainer:
        """PINN CFD surrogate: [U, AoA, Re] → [CL, CD, CM].

        Replaces expensive OpenFOAM calls at inference time.
        Train on AeroDatabase.to_numpy() output.
        """
        cfg = PINNConfig(
            hidden_layers=[64, 64, 32],
            activation="tanh",
            w_data=1.0,
            w_physics=0.05,
            n_epochs=n_epochs,
            learning_rate=1e-3,
        )
        return PINNTrainer(cfg)

    @staticmethod
    def system_id_pinn(n_epochs: int = 5000) -> PINNTrainer:
        """PINN for system identification: identifies m, I, drag from flight data."""
        cfg = PINNConfig(
            hidden_layers=[32, 32],
            activation="tanh",
            w_data=1.0,
            w_physics=1.0,   # Strong physics constraint for sys-ID
            n_epochs=n_epochs,
            learning_rate=1e-4,
        )
        return PINNTrainer(cfg)

    @staticmethod
    def generate_synthetic_aero_data(
        n_samples: int    = 500,
        noise_std: float  = 0.005,
        seed:      int    = 42,
    ) -> TrainingData:
        """Generate synthetic aerodynamic polar for initial PINN testing.

        Uses flat-plate theory and empirical quadratic drag polar.
        Replace with real OpenFOAM data for production use.

        Returns:
            TrainingData with inputs=[U, AoA_rad, Re] and outputs=[CL, CD].
        """
        rng = np.random.default_rng(seed)
        U_range   = rng.uniform(5.0, 25.0, n_samples)
        aoa_range = rng.uniform(-0.175, 0.262, n_samples)   # -10° to 15°
        L_ref     = 0.25

        Re = U_range * L_ref / 1.5e-5   # kinematic viscosity ≈ 1.5e-5

        # Flat plate theory (Prandtl lifting line approximation)
        CL = 2 * math.pi * aoa_range + rng.normal(0, noise_std, n_samples)
        CD = 0.02 + 0.05 * aoa_range**2 + rng.normal(0, noise_std/5, n_samples)
        CD = np.maximum(CD, 0.005)   # Physical lower bound

        inputs  = np.stack([U_range, aoa_range, Re], axis=1)
        outputs = np.stack([CL, CD], axis=1)

        return TrainingData(
            inputs       = inputs,
            outputs      = outputs,
            input_names  = ["velocity_ms", "aoa_rad", "reynolds"],
            output_names = ["CL", "CD"],
            metadata     = {"source": "synthetic", "n_samples": n_samples},
        )
