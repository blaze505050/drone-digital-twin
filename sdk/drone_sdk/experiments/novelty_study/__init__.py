"""
drone_sdk.experiments.novelty_study
===================================
Preregistered comparative studies, ablation matrices, and predictive performance benchmarks.
"""
from __future__ import annotations

from .study import (
    AblationPoint,
    NoveltyStudyPoint,
    NoveltyStudyRunner,
)

__all__ = [
    "NoveltyStudyPoint",
    "AblationPoint",
    "NoveltyStudyRunner",
]
