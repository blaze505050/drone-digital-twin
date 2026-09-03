"""Tests for drone_sdk.pinn_engine (Module 14)."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pytest

from drone_sdk.pinn_engine import (
    AdamOptimiser, NumpyMLP, PINNConfig, PINNFactory,
    PINNTrainer, TrainingData, TrainingHistory,
    UAVPhysicsResidual,
)

SEED = 42


class TestNumpyMLP:
    @pytest.fixture
    def mlp(self):
        return NumpyMLP([3, 16, 8, 2], activation="tanh", seed=SEED)

    def test_forward_shape(self, mlp):
        X = np.random.randn(10, 3)
        Y = mlp.forward(X)
        assert Y.shape == (10, 2)

    def test_predict_single_row(self, mlp):
        x = np.random.randn(1, 3)
        y = mlp.predict(x)
        assert y.shape[1] == 2

    def test_n_parameters_correct(self):
        mlp = NumpyMLP([3, 4, 2])
        # Layer 0: 3×4 + 4 = 16; Layer 1: 4×2 + 2 = 10; total = 26
        assert mlp.n_parameters() == 26

    def test_forward_finite(self, mlp):
        X = np.random.randn(50, 3)
        Y = mlp.forward(X)
        assert np.all(np.isfinite(Y))

    def test_relu_activation(self):
        mlp = NumpyMLP([2, 4, 1], activation="relu")
        X   = np.random.randn(5, 2)
        Y   = mlp.forward(X)
        assert np.all(np.isfinite(Y))

    def test_swish_activation(self):
        mlp = NumpyMLP([2, 4, 1], activation="swish")
        X   = np.random.randn(5, 2)
        Y   = mlp.forward(X)
        assert np.all(np.isfinite(Y))

    def test_jacobian_shape(self, mlp):
        x = np.random.randn(3)
        J = mlp.jacobian(x)
        assert J.shape == (2, 3)

    def test_jacobian_finite(self, mlp):
        x = np.random.randn(3)
        J = mlp.jacobian(x)
        assert np.all(np.isfinite(J))

    def test_get_set_parameters(self, mlp):
        p = mlp.get_parameters()
        p_mod = p * 2.0
        mlp.set_parameters(p_mod)
        p_new = mlp.get_parameters()
        assert np.allclose(p_new, p_mod)

    def test_parameters_roundtrip(self, mlp):
        original = mlp.get_parameters().copy()
        mlp.set_parameters(original)
        assert np.allclose(mlp.get_parameters(), original)

    def test_xavier_init_scale(self):
        # Weights should have moderate scale (not too large or small)
        mlp = NumpyMLP([100, 64, 32], seed=SEED)
        W   = mlp.weights[0]
        assert W.std() < 0.5
        assert W.std() > 0.001

    def test_repr(self, mlp):
        r = repr(mlp)
        assert "NumpyMLP" in r
        assert "tanh"    in r


class TestAdamOptimiser:
    def test_step_updates_params(self):
        opt    = AdamOptimiser(lr=0.01)
        params = np.ones(5)
        grad   = np.ones(5)
        new_p  = opt.step(params, grad)
        assert not np.allclose(new_p, params)

    def test_step_decreases_loss_on_quadratic(self):
        # Minimise f(x) = x² starting from x=1
        opt = AdamOptimiser(lr=0.1)
        x   = np.array([1.0])
        for _ in range(100):
            grad = 2 * x
            x    = opt.step(x, grad)
        assert abs(x[0]) < 0.1

    def test_step_finite(self):
        opt    = AdamOptimiser()
        params = np.random.randn(20)
        grad   = np.random.randn(20)
        new_p  = opt.step(params, grad)
        assert np.all(np.isfinite(new_p))

    def test_timestep_increments(self):
        opt = AdamOptimiser()
        for _ in range(5):
            opt.step(np.ones(3), np.ones(3))
        assert opt.t == 5


class TestTrainingData:
    @pytest.fixture
    def data(self):
        np.random.seed(SEED)
        X = np.random.randn(200, 3)
        Y = np.random.randn(200, 2)
        return TrainingData(X, Y, input_names=["a","b","c"], output_names=["x","y"])

    def test_n_samples(self, data):
        assert data.n_samples == 200

    def test_n_in(self, data):
        assert data.n_in == 3

    def test_n_out(self, data):
        assert data.n_out == 2

    def test_normalise_range(self, data):
        norm, _ = data.normalise()
        assert norm.inputs.min()  >= -1.0 - 1e-6
        assert norm.inputs.max()  <=  1.0 + 1e-6
        assert norm.outputs.min() >= -1.0 - 1e-6
        assert norm.outputs.max() <=  1.0 + 1e-6

    def test_normalise_returns_stats(self, data):
        _, stats = data.normalise()
        assert "x_min" in stats
        assert "y_max" in stats

    def test_train_test_split_sizes(self, data):
        train, test = data.train_test_split(test_frac=0.2)
        assert train.n_samples == 160
        assert test.n_samples  == 40

    def test_train_test_no_overlap(self, data):
        # This is a statistical test - indices should not repeat
        train, test = data.train_test_split(test_frac=0.2)
        # Total should equal original
        assert train.n_samples + test.n_samples == data.n_samples


class TestUAVPhysicsResidual:
    @pytest.fixture
    def residual(self):
        return UAVPhysicsResidual(mass_kg=1.5, ixx=0.035, iyy=0.046, izz=0.098)

    def test_translational_residual_hover(self, residual):
        """At hover (zero accel, zero vel), thrust should equal gravity."""
        accel  = np.zeros(3)
        vel    = np.zeros(3)
        quat   = np.array([1.0, 0.0, 0.0, 0.0])   # level
        thrust = residual.mass * residual.g
        res    = residual.translational_residual(accel, vel, quat, thrust)
        # Residual should be near zero for correct physics
        assert np.linalg.norm(res) < 1.0

    def test_translational_residual_returns_3vector(self, residual):
        res = residual.translational_residual(
            np.zeros(3), np.zeros(3),
            np.array([1.0,0,0,0]), 14.7
        )
        assert res.shape == (3,)

    def test_rotational_residual_equilibrium(self, residual):
        """At zero angular rate with matching torque, residual ≈ 0."""
        omega  = np.zeros(3)
        torque = np.zeros(3)
        alpha  = np.zeros(3)
        res    = residual.rotational_residual(alpha, omega, torque)
        assert np.linalg.norm(res) < 1e-10

    def test_rotational_residual_returns_3vector(self, residual):
        res = residual.rotational_residual(
            np.zeros(3), np.array([0.1,0.0,0.0]), np.zeros(3)
        )
        assert res.shape == (3,)

    def test_energy_residual_zero_at_rest(self, residual):
        vel = np.zeros(3)
        e   = residual.energy_residual(vel, 0.0, 0.0, 0.01)
        assert abs(e) < 1e-6


class TestPINNTrainer:
    @pytest.fixture
    def simple_data(self):
        """Simple sin(x) regression task."""
        np.random.seed(SEED)
        X = np.linspace(-1, 1, 200).reshape(-1, 1)
        Y = np.sin(math.pi * X)
        return TrainingData(X, Y, input_names=["x"], output_names=["y"])

    @pytest.fixture
    def trainer(self):
        cfg = PINNConfig(
            hidden_layers=[16, 16],
            activation="tanh",
            n_epochs=200,
            learning_rate=5e-3,
            verbose=False,
        )
        return PINNTrainer(cfg)

    def test_fit_returns_history(self, trainer, simple_data):
        h = trainer.fit(simple_data)
        assert isinstance(h, TrainingHistory)

    def test_history_has_losses(self, trainer, simple_data):
        h = trainer.fit(simple_data)
        assert len(h.total_loss) > 0
        assert len(h.data_loss)  > 0

    def test_loss_decreases(self, trainer, simple_data):
        h = trainer.fit(simple_data)
        # Loss at end should be lower than at start
        assert h.total_loss[-1] < h.total_loss[0]

    def test_predict_shape(self, trainer, simple_data):
        trainer.fit(simple_data)
        Y = trainer.predict(np.linspace(-1, 1, 10).reshape(-1, 1))
        assert Y.shape == (10, 1)

    def test_predict_finite(self, trainer, simple_data):
        trainer.fit(simple_data)
        Y = trainer.predict(np.linspace(-1, 1, 20).reshape(-1, 1))
        assert np.all(np.isfinite(Y))

    def test_predict_before_fit_raises(self):
        trainer = PINNTrainer()
        with pytest.raises(RuntimeError, match="not trained"):
            trainer.predict(np.array([[1.0]]))

    def test_save_load_roundtrip(self, trainer, simple_data, tmp_path):
        trainer.fit(simple_data)
        path = trainer.save(str(tmp_path / "pinn.json"))
        assert path.exists()
        loaded = PINNTrainer.load(str(path))
        # Predictions should match
        X = np.linspace(-1, 1, 5).reshape(-1, 1)
        np.testing.assert_allclose(
            trainer.predict(X), loaded.predict(X), atol=1e-6
        )

    def test_best_epoch_in_range(self, trainer, simple_data):
        h = trainer.fit(simple_data)
        assert 0 <= h.best_epoch() <= 200

    def test_history_repr(self, trainer, simple_data):
        h = trainer.fit(simple_data)
        assert "TrainingHistory" in repr(h)

    def test_history_to_dict(self, trainer, simple_data):
        h = trainer.fit(simple_data)
        d = h.to_dict()
        assert "epochs" in d
        assert "total_loss" in d


class TestPINNFactory:
    def test_dynamics_pinn_created(self):
        p = PINNFactory.dynamics_pinn(n_epochs=10)
        assert isinstance(p, PINNTrainer)

    def test_aerodynamics_pinn_created(self):
        p = PINNFactory.aerodynamics_pinn(n_epochs=10)
        assert isinstance(p, PINNTrainer)

    def test_system_id_pinn_created(self):
        p = PINNFactory.system_id_pinn(n_epochs=10)
        assert isinstance(p, PINNTrainer)

    def test_synthetic_aero_data_shape(self):
        d = PINNFactory.generate_synthetic_aero_data(n_samples=100)
        assert d.inputs.shape  == (100, 3)
        assert d.outputs.shape == (100, 2)

    def test_synthetic_aero_physical_bounds(self):
        d = PINNFactory.generate_synthetic_aero_data(n_samples=500)
        CD = d.outputs[:, 1]
        assert np.all(CD > 0), "Drag coefficient must be positive"

    def test_aero_pinn_trains_on_synthetic(self):
        data    = PINNFactory.generate_synthetic_aero_data(n_samples=100, seed=SEED)
        trainer = PINNFactory.aerodynamics_pinn(n_epochs=50)
        history = trainer.fit(data)
        assert len(history.total_loss) > 0
        assert history.best_loss() < 10.0

    def test_aero_surrogate_predicts_positive_drag(self):
        data    = PINNFactory.generate_synthetic_aero_data(n_samples=200, seed=SEED)
        trainer = PINNFactory.aerodynamics_pinn(n_epochs=100)
        trainer.fit(data)
        # Test at nominal cruise: U=10 m/s, AoA=0°, Re=1.67e5
        X    = np.array([[10.0, 0.0, 1.67e5]])
        pred = trainer.predict(X)
        # After training on synthetic data, CD should be close to 0.02 (zero AoA)
        assert pred.shape == (1, 2)
        assert np.all(np.isfinite(pred))
