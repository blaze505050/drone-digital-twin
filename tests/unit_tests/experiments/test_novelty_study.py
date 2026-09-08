"""
Unit tests for Preregistered Novelty Study & Ablation Matrix (B26).
Verifies:
- 4-model comparative study across payload shifts and wind speeds.
- Identification-gated adaptive twin superiority over uncalibrated nominal baseline.
- Systematic architectural ablations (w/o excitation, w/o safety clamps, w/o holdout gate).
"""
import numpy as np
import pytest

from drone_sdk.experiments.novelty_study import (
    NoveltyStudyRunner,
)


def test_novelty_study_comparative_benchmark():
    runner = NoveltyStudyRunner()
    results = runner.run_comparative_study(
        payload_shifts_kg=[0.0, 0.200],
        wind_speeds_mps=[0.0, 5.0],
    )

    # 2 payloads * 2 winds * 4 methods = 16 points
    assert len(results) == 16

    # For shifted payload (+200g), adaptive twin error should be significantly lower than nominal baseline
    nom_shifted = [r for r in results if r.method_name == "Nominal Baseline" and r.payload_added_kg == 0.200 and r.wind_speed_mps == 0.0][0]
    adapt_shifted = [r for r in results if r.method_name == "Adaptive Twin (Ours)" and r.payload_added_kg == 0.200 and r.wind_speed_mps == 0.0][0]

    assert adapt_shifted.energy_error_pct < nom_shifted.energy_error_pct
    assert adapt_shifted.energy_error_pct <= 5.0
    assert adapt_shifted.is_safe is True


def test_ablation_matrix():
    runner = NoveltyStudyRunner()
    ablations = runner.run_ablation_matrix()

    assert len(ablations) == 4
    ref = ablations[0]
    assert ref.ablation_name == "Full Architecture (Reference)"
    assert ref.mean_energy_error_pct < 3.0
    assert ref.divergence_count == 0

    # Ablations without safety bounds or excitation should exhibit higher errors/divergences
    no_clamp = [a for a in ablations if "Safety" in a.ablation_name][0]
    assert no_clamp.mean_energy_error_pct > ref.mean_energy_error_pct
    assert no_clamp.divergence_count > 0

    # Format Markdown
    points = runner.run_comparative_study(payload_shifts_kg=[0.0], wind_speeds_mps=[0.0])
    md_str = runner.format_study_markdown(points, ablations)
    assert "# Preregistered Novelty Study" in md_str
    assert "Architectural Ablation Matrix" in md_str
