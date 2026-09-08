"""
drone_sdk.configuration
=======================
Vehicle configuration, component BOM, and asset registry for UAV Digital Twin.
"""
from __future__ import annotations

from typing import Dict

from .schema import (
    AirframeConfig,
    AvionicsConfig,
    BatteryPackConfig,
    ComponentBOM,
    MotorConfig,
    PropellerConfig,
    RotorPlacement,
    VehicleConfiguration,
)
from .vehicles.holybro_x500 import create_holybro_x500_v2

_REGISTRY: Dict[str, VehicleConfiguration] = {
    "holybro_x500_v2": create_holybro_x500_v2(),
}


def get_vehicle_config(asset_id: str = "holybro_x500_v2") -> VehicleConfiguration:
    """Retrieve a registered vehicle configuration by asset ID."""
    if asset_id in _REGISTRY:
        return _REGISTRY[asset_id]
    # Default fallback
    return create_holybro_x500_v2()


def register_vehicle_config(config: VehicleConfiguration) -> None:
    """Register a custom vehicle configuration."""
    _REGISTRY[config.asset_id] = config


__all__ = [
    "ComponentBOM",
    "MotorConfig",
    "PropellerConfig",
    "BatteryPackConfig",
    "AirframeConfig",
    "AvionicsConfig",
    "RotorPlacement",
    "VehicleConfiguration",
    "create_holybro_x500_v2",
    "get_vehicle_config",
    "register_vehicle_config",
]
