"""
Unit tests for Bounded System Identification & Rollback Engine (B18).
Verifies:
- Regressor condition number and dynamic excitation checking.
- Bounded parameter estimation (+/- 15% limits).
- Holdout evaluation, promotion to ACTIVE, and automatic rollback.
"""
import numpy as np
import pytest

from drone_sdk.identification import (
    BoundedParameterIdentifier,
    IdentificationDataset,
    ParameterSet,
)


def test_excitation_check_passes_on_dynamic_flight():
    t = np.linspace(0.0, 5.0, 100)
    # Dynamic sinusoidal throttle command with substantial variance
    cmds = np.zeros((100, 4))
    for m in range(4):
        cmds[:, m] = 0.55 + 0.15 * np.sin(2.0 * np.pi * 0.5 * t + m * 0.5)

    accel_ned = np.zeros((100, 3))
    accel_ned[:, 2] = -9.81 - 5.0 * np.sin(2.0 * np.pi * 0.5 * t)
    omega_body = np.zeros((100, 3))

    dataset = IdentificationDataset(
        time_sec=t,
        motor_commands=cmds,
        measured_accel_ned=accel_ned,
        measured_omega_body=omega_body,
    )

    identifier = BoundedParameterIdentifier()
    is_ok, cond_num, msg = identifier.check_excitation(dataset)
    assert is_ok is True
    assert cond_num <= 100.0


def test_excitation_check_fails_on_flat_hover():
    t = np.linspace(0.0, 5.0, 100)
    # Flat constant hover command (zero excitation variance)
    cmds = np.full((100, 4), 0.55)
    accel_ned = np.full((100, 3), [0.0, 0.0, -9.81])
    omega_body = np.zeros((100, 3))

    dataset = IdentificationDataset(
        time_sec=t,
        motor_commands=cmds,
        measured_accel_ned=accel_ned,
        measured_omega_body=omega_body,
    )

    identifier = BoundedParameterIdentifier()
    is_ok, _, msg = identifier.check_excitation(dataset)
    assert is_ok is False
    assert "Insufficient excitation" in msg


def test_bounded_parameter_fit_and_promotion():
    t = np.linspace(0.0, 5.0, 100)
    cmds = np.zeros((100, 4))
    for m in range(4):
        cmds[:, m] = 0.55 + 0.15 * np.sin(2.0 * np.pi * 0.5 * t + m * 0.5)

    accel_ned = np.zeros((100, 3))
    # True vehicle has +8% thrust scale
    accel_ned[:, 2] = 9.81 - 26.0 * 1.08 * (np.sum(cmds, axis=1) / 4.0)
    omega_body = np.zeros((100, 3))

    train_data = IdentificationDataset(t[:50], cmds[:50], accel_ned[:50], omega_body[:50])
    val_data = IdentificationDataset(t[50:], cmds[50:], accel_ned[50:], omega_body[50:])

    identifier = BoundedParameterIdentifier(max_parameter_shift_pct=15.0)
    success, candidate, msg = identifier.fit_candidate_parameters(train_data)

    assert success is True
    assert candidate.status == "CANDIDATE"
    # Parameter must stay within +/- 15% of 1.0
    assert 0.85 <= candidate.motor_thrust_scale <= 1.15
    assert 0.85 <= candidate.mass_scale <= 1.15

    # Evaluate on held-out validation segment
    promoted, active_res, metrics = identifier.evaluate_and_promote(candidate, val_data, min_improvement_pct=3.0)
    assert promoted is True
    assert active_res.status == "ACTIVE"
    assert metrics["improvement_pct"] > 3.0


def test_rollback_on_degraded_candidate():
    t = np.linspace(0.0, 5.0, 100)
    cmds = np.zeros((100, 4))
    cmds[:, :] = 0.55 + 0.10 * np.sin(2.0 * np.pi * 0.5 * t)[:, None]
    accel_ned = np.zeros((100, 3))
    # True model is standard 1.0 scale
    accel_ned[:, 2] = 9.81 - 26.0 * (np.sum(cmds, axis=1) / 4.0)
    val_data = IdentificationDataset(t, cmds, accel_ned, np.zeros((100, 3)))

    identifier = BoundedParameterIdentifier()
    # Artificially created bad candidate
    bad_cand = ParameterSet(version=2, mass_scale=1.15, motor_thrust_scale=0.85, status="CANDIDATE")

    promoted, active_res, metrics = identifier.evaluate_and_promote(bad_cand, val_data, min_improvement_pct=5.0)
    assert promoted is False
    assert bad_cand.status == "ROLLED_BACK"
    assert active_res.version == 1  # Retained original active model
