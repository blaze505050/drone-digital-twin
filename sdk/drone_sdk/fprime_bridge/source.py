"""
drone_sdk.fprime_bridge.source
==============================
NASA F Prime (F') TelemetrySource for TelemetryEngine.

Integrates the NASA F Prime flight computer into the multi-source
TelemetryEngine and SourceArbiter pipeline, stamping incoming updates
with DataSource.HIL or DataSource.SITL.

Python version: 3.9+
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from drone_sdk.fprime_bridge.bridge import FPrimeSITLBridge
from drone_sdk.state_manager.schema import DataSource, DroneStateUpdate
from drone_sdk.telemetry_engine.source import (
    SourcePriority,
    SourceStatus,
    TelemetrySource,
)


class FPrimeTelemetrySource(TelemetrySource):
    """TelemetrySource adapter connecting NASA F' SITL into TelemetryEngine."""

    def __init__(
        self,
        vehicle_id: str = "fprime_drone",
        bridge: Optional[FPrimeSITLBridge] = None,
        priority: SourcePriority = SourcePriority.HARDWARE,
    ) -> None:
        super().__init__(
            source_id=f"fprime_{vehicle_id}",
            vehicle_id=vehicle_id,
            priority=priority,
            data_source=DataSource.HIL if priority == SourcePriority.HARDWARE else DataSource.SITL,
        )
        self.bridge = bridge or FPrimeSITLBridge(vehicle_id=vehicle_id)
        self._last_state_update: Optional[DroneStateUpdate] = None

    def connect(self) -> None:
        """Connect bridge and transition to CONNECTED status."""
        if self.bridge.start():
            self._status = SourceStatus.CONNECTED
        else:
            self._status = SourceStatus.ERROR

    def disconnect(self) -> None:
        """Stop bridge and set DISCONNECTED status."""
        self.bridge.stop()
        self._status = SourceStatus.DISCONNECTED

    def poll(self) -> List[DroneStateUpdate]:
        """Poll commands from F' and convert to state updates if available."""
        if self._status != SourceStatus.CONNECTED:
            return []

        commands = self.bridge.poll_commands()
        updates: List[DroneStateUpdate] = []

        for cmd in commands:
            upd = DroneStateUpdate(
                vehicle_id=self.vehicle_id,
                source=self.data_source,
            )
            upd.position = cmd.pos_target_ned.copy()
            upd.velocity = cmd.vel_target_ned.copy()
            upd.yaw = float(cmd.yaw_target_rad)
            updates.append(upd)
            self._last_state_update = upd

        return updates

    @property
    def latest_update(self) -> Optional[DroneStateUpdate]:
        return self._last_state_update
