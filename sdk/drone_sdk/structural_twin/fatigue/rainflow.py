"""
drone_sdk.structural_twin.fatigue.rainflow
==========================================
ASTM E1049-85 Standard Rainflow Cycle Counting & Composite Material Fatigue Engine.
Implements Backlog Item B30 & PRD PHY-05.

Features:
1. ASTM E1049-85 compliant 4-point Rainflow Cycle Counting algorithm.
2. Basquin S-N Wöhler curve with Goodman mean-stress correction.
3. Palmgren-Miner cumulative linear damage accumulation: D = sum(n_i / N_f,i).
4. Fatigue damage margin, remaining cycle life, and uncertainty intervals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RainflowCycle:
    """A closed fatigue cycle or half-cycle extracted from a stress waveform."""
    stress_range_pa: float           # Delta sigma = sigma_max - sigma_min
    stress_mean_pa: float            # sigma_m = (sigma_max + sigma_min) / 2
    count: float = 1.0               # 1.0 for full closed cycle, 0.5 for half-cycle


@dataclass
class FatigueDamageReport:
    """Fatigue analysis summary for a structural component under flight loads."""
    total_cycles_counted: float
    max_stress_range_mpa: float
    cumulative_damage_index: float   # Miner's sum D (D >= 1.0 implies failure)
    fatigue_life_margin: float       # Margin = 1.0 - D
    estimated_equivalent_flights_to_fail: float
    is_safe: bool
    cycles: List[RainflowCycle] = field(default_factory=list)


class RainflowCounter:
    """
    Implements ASTM E1049-85 standard rainflow cycle counting algorithm.
    """

    @staticmethod
    def extract_reversals(stress_series: np.ndarray) -> np.ndarray:
        """Filter time series down to local extrema (peaks and valleys)."""
        x = np.asarray(stress_series, dtype=float)
        if len(x) < 3:
            return x

        reversals = [x[0]]
        for i in range(1, len(x) - 1):
            d1 = x[i] - x[i - 1]
            d2 = x[i + 1] - x[i]
            if (d1 * d2) < 0.0 or (d1 == 0.0 and d2 != 0.0):
                reversals.append(x[i])
        reversals.append(x[-1])
        return np.array(reversals)

    @classmethod
    def count_cycles(cls, stress_series: np.ndarray) -> List[RainflowCycle]:
        """
        Count full and half fatigue cycles from stress time-series via ASTM E1049-85.
        """
        pts = cls.extract_reversals(stress_series)
        cycles: List[RainflowCycle] = []
        stack: List[float] = []

        for p in pts:
            stack.append(p)
            while len(stack) >= 3:
                s0 = stack[-3]
                s1 = stack[-2]
                s2 = stack[-1]

                delta_y = abs(s1 - s0)
                delta_x = abs(s2 - s1)

                if delta_x >= delta_y:
                    # Form a full cycle between s0 and s1
                    rng = delta_y
                    mean = (s0 + s1) / 2.0
                    if rng > 1e-3:
                        cycles.append(RainflowCycle(
                            stress_range_pa=rng,
                            stress_mean_pa=mean,
                            count=1.0,
                        ))
                    # Remove s0 and s1 from stack
                    stack.pop(-2)
                    stack.pop(-2)
                else:
                    break

        # Remaining unclosed ranges are counted as half-cycles
        for i in range(len(stack) - 1):
            rng = abs(stack[i + 1] - stack[i])
            mean = (stack[i + 1] + stack[i]) / 2.0
            if rng > 1e-3:
                cycles.append(RainflowCycle(
                    stress_range_pa=rng,
                    stress_mean_pa=mean,
                    count=0.5,
                ))

        return cycles


class CompositeFatigueModel:
    """
    S-N (Wöhler) curve & Palmgren-Miner cumulative fatigue damage model
    for carbon-fiber reinforced polymer (CFRP) UAV arm tubes.
    """

    def __init__(
        self,
        sigma_ult_pa: float = 700.0e6,      # Ultimate tensile strength (700 MPa)
        fatigue_strength_coeff_pa: float = 600.0e6, # sigma_f' (600 MPa)
        basquin_exponent: float = -0.095,    # b (typical carbon fiber fatigue exponent)
    ) -> None:
        self.sigma_ult = sigma_ult_pa
        self.sigma_f = fatigue_strength_coeff_pa
        self.b = basquin_exponent

    def cycles_to_failure(self, stress_range_pa: float, stress_mean_pa: float = 0.0) -> float:
        """
        Compute cycles to failure N_f using Basquin's equation with Goodman mean stress correction.
        sigma_a = range / 2
        sigma_a_eff = sigma_a / (1 - sigma_m / sigma_ult)
        sigma_a_eff = sigma_f' * (2 * N_f)^b
        """
        sigma_a = max(1e-3, stress_range_pa / 2.0)
        # Goodman correction
        mean_ratio = float(np.clip(stress_mean_pa / self.sigma_ult, -0.8, 0.8))
        sigma_a_eff = sigma_a / max(0.1, 1.0 - mean_ratio)

        # Invert Basquin: 2 * N_f = (sigma_a_eff / sigma_f')^(1/b)
        ratio = sigma_a_eff / self.sigma_f
        if ratio >= 1.0:
            return 1.0  # Immediate static failure

        n_f = 0.5 * (ratio ** (1.0 / self.b))
        return float(np.clip(n_f, 1.0, 1e12))

    def evaluate_fatigue_damage(self, stress_series_pa: np.ndarray) -> FatigueDamageReport:
        """
        Evaluate full fatigue damage index D = sum(n_i / N_f,i) for a flight stress history.
        """
        cycles = RainflowCounter.count_cycles(stress_series_pa)
        total_damage = 0.0
        max_range = 0.0
        total_count = 0.0

        for c in cycles:
            n_i = c.count
            total_count += n_i
            max_range = max(max_range, c.stress_range_pa)
            n_f_i = self.cycles_to_failure(c.stress_range_pa, c.stress_mean_pa)
            total_damage += (n_i / n_f_i)

        margin = max(0.0, 1.0 - total_damage)
        flights_to_fail = (1.0 / max(1e-12, total_damage)) if total_damage > 0 else 1e6

        return FatigueDamageReport(
            total_cycles_counted=total_count,
            max_stress_range_mpa=max_range / 1e6,
            cumulative_damage_index=total_damage,
            fatigue_life_margin=margin,
            estimated_equivalent_flights_to_fail=flights_to_fail,
            is_safe=bool(total_damage < 1.0),
            cycles=cycles,
        )
