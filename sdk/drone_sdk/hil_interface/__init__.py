"""
drone_sdk.hil_interface
=======================
Hardware-in-the-Loop (HIL) execution, autopilot synchronization, and link latency monitoring.
"""
from __future__ import annotations

from .supervised_hil import (
    HILMetrics,
    HILSessionConfig,
    SupervisedHILManager,
)

__all__ = [
    "HILSessionConfig",
    "HILMetrics",
    "SupervisedHILManager",
]
