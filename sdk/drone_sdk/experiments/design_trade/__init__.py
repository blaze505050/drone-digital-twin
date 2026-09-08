"""
drone_sdk.experiments.design_trade
==================================
Propeller sizing, payload trade studies, and multirotor configuration optimization.
"""
from __future__ import annotations

from .trade_study import DesignTradeStudy, TradePointResult

__all__ = [
    "TradePointResult",
    "DesignTradeStudy",
]
