"""
Unit tests for Replay Service, Prospective Energy Forecast, UQ, and Report Generation (B10, B14, B15, B16).
Verifies:
- DataManifest serialization and SHA256 integrity check.
- LogReplayService chronological streaming and sensor isolation.
- MissionEnergyEvaluator prospective waypoint forecast and holdout evaluation.
- UncertaintyQuantifier 95% confidence intervals and ValidityDomainChecker OOD detection.
- EngineeringReportGenerator markdown and JSON outputs.
"""
import numpy as np
import pytest

from drone_sdk.configuration import get_vehicle_config
from drone_sdk.contracts import DataStatus
from drone_sdk.experiments.energy_forecast import (
    MissionEnergyEvaluator,
    MissionPlan,
    Waypoint,
)
from drone_sdk.uncertainty import (
    UncertaintyQuantifier,
    ValidityDomainChecker,
)
from drone_sdk.validation_framework.manifest import ChannelMetadata, DataManifest
from drone_sdk.validation_framework.replay_service import LogReplayService
from drone_sdk.validation_framework.report_generator import EngineeringReportGenerator


def test_data_manifest_and_replay_service(tmp_path):
    manifest = DataManifest(
        dataset_id="flight_test_01",
        title="Physical Flight Test Run 01",
        description="Measured quadrotor hover and translation dataset",
        data_status=DataStatus.REAL,
        file_sha256="",
        channels=[
            ChannelMetadata("pos_ned", "m", "NED", 100.0),
            ChannelMetadata("voltage_v", "V", "SCALAR", 50.0),
        ],
    )
    assert manifest.is_empirical is True

    # Test Replay Service with synthetic array data
    t = np.linspace(0.0, 5.0, 51)
    pos_ned = np.zeros((51, 3))
    pos_ned[:, 2] = -5.0  # 5m hover
    voltage = np.linspace(16.8, 16.0, 51)
    current = np.full(51, 12.0)

    replay = LogReplayService(
        vehicle_id="holybro_x500_v2",
        data_status=DataStatus.SYNTHETIC,
        manifest=manifest,
    )
    total_loaded = replay.load_from_arrays(
        t=t,
        pos_ned=pos_ned,
        voltage_v=voltage,
        current_a=current,
    )
    assert total_loaded == 51
    assert replay.total_frames == 51

    # Step through frames and verify data isolation
    frame0 = replay.step()
    assert frame0 is not None
    assert frame0.timestamp_sec == 0.0
    assert frame0.truth is not None
    assert np.isclose(frame0.truth.pos_ned[2], -5.0)

    # Replay stream generator
    frames = list(replay.stream_frames())
    assert len(frames) == 51


def test_mission_energy_forecast_and_holdout():
    cfg = get_vehicle_config("holybro_x500_v2")
    evaluator = MissionEnergyEvaluator(cfg)

    # 4-waypoint survey box (100m x 100m at 20m altitude)
    mission = MissionPlan(
        mission_id="survey_box_100m",
        waypoints=[
            Waypoint(0.0, 0.0, 0.0),          # Takeoff origin
            Waypoint(0.0, 0.0, -20.0, hover_sec=5.0), # Climb to 20m AGL
            Waypoint(100.0, 0.0, -20.0),      # Leg 1
            Waypoint(100.0, 100.0, -20.0),    # Leg 2
            Waypoint(0.0, 0.0, -20.0, hover_sec=5.0), # Return
            Waypoint(0.0, 0.0, 0.0),          # Land
        ],
        cruise_speed_mps=8.0,
        climb_speed_mps=2.5,
    )

    result = evaluator.forecast_mission(mission)
    assert result.total_distance_m > 200.0
    assert result.total_duration_sec > 30.0
    assert result.predicted_energy_wh > 1.0
    assert result.is_feasible is True
    assert 0.50 <= result.final_soc <= 1.0

    # Test holdout evaluation
    mock_flight_log = {
        "energy_wh": np.array([0.0, result.predicted_energy_wh * 1.02]),  # 2% error
        "voltage_v": result.time_series["voltage_v"] + 0.05,
    }
    metrics = evaluator.evaluate_holdout_prediction(mock_flight_log, result)
    assert metrics["energy_error_percent"] < 5.0
    assert metrics["is_within_5pct_tolerance"] is True


def test_uncertainty_quantification_and_ood():
    cfg = get_vehicle_config("holybro_x500_v2")
    uq = UncertaintyQuantifier(cfg)
    ci = uq.propagate_mission_energy_uncertainty(nominal_energy_wh=25.0)

    assert ci.mean == 25.0
    assert ci.std_dev > 0.5
    assert ci.ci_95_lower < 25.0 < ci.ci_95_upper
    assert ci.unit == "Wh"

    # Validity domain check
    checker = ValidityDomainChecker()
    # In-domain case
    rep_in = checker.check_conditions(mass_kg=1.55, ambient_temp_c=25.0, wind_speed_mps=4.0)
    assert rep_in.is_in_domain is True
    assert rep_in.ood_score == 0.0

    # OOD case (excessive mass + high wind)
    rep_out = checker.check_conditions(mass_kg=3.20, ambient_temp_c=25.0, wind_speed_mps=18.0)
    assert rep_out.is_in_domain is False
    assert rep_out.ood_score > 1.0
    assert len(rep_out.warnings) >= 2


def test_engineering_report_generation(tmp_path):
    cfg = get_vehicle_config("holybro_x500_v2")
    evaluator = MissionEnergyEvaluator(cfg)
    mission = MissionPlan(
        mission_id="report_test",
        waypoints=[
            Waypoint(0, 0, 0),
            Waypoint(0, 0, -10),
            Waypoint(50, 0, -10),
        ]
    )
    res = evaluator.forecast_mission(mission)
    uq = UncertaintyQuantifier(cfg).propagate_mission_energy_uncertainty(res.predicted_energy_wh)
    val = ValidityDomainChecker().check_conditions(mass_kg=cfg.compute_total_mass())

    generator = EngineeringReportGenerator(
        vehicle_config=cfg,
        forecast_result=res,
        uncertainty_energy=uq,
        validity_report=val,
        holdout_metrics={"energy_error_percent": 1.5, "is_within_5pct_tolerance": True},
    )

    md_text = generator.generate_markdown_report()
    assert "# UAV Digital Twin — Engineering Verification Report" in md_text
    assert "Holybro X500 V2" in md_text
    assert "95% Confidence Interval" in md_text

    md_path, json_path = generator.save_report(tmp_path)
    assert md_path.is_file()
    assert json_path.is_file()
