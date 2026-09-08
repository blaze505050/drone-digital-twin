"""
Unit tests for Supervised HIL Manager & Autopilot Synchronization (B25).
Verifies:
- Bidirectional HIL session lifecycle.
- RTT latency statistics and packet counters.
- Link-loss detection and automatic emergency failsafe triggering.
"""
import time
import numpy as np
import pytest

from drone_sdk.contracts.data_status import DataStatus
from drone_sdk.contracts.streams import SensorMeasurement
from drone_sdk.hil_interface import (
    HILSessionConfig,
    SupervisedHILManager,
)


def test_supervised_hil_manager_sync_and_latency():
    cfg = HILSessionConfig(
        vehicle_id="holybro_x500_v2",
        sync_rate_hz=100.0,
        max_allowed_rtt_ms=50.0,
        link_loss_timeout_sec=1.0,
    )
    hil = SupervisedHILManager(cfg)
    started = hil.start_session()
    assert started is True

    # Step through 10 HIL cycles
    for i in range(10):
        meas = SensorMeasurement(
            vehicle_id="holybro_x500_v2",
            sensor_type="imu",
            timestamp_mono=100.0 + i * 0.01,
            status=DataStatus.SYNTHETIC,
            values=np.array([0.0, 0.0, -9.81]),
        )
        actuation, metrics = hil.sync_step(meas, current_mono=100.0 + i * 0.01)

        assert actuation is not None
        assert actuation.motor_signals.shape == (4,)
        assert metrics.is_link_alive is True
        assert metrics.packets_sent == i + 1
        assert metrics.avg_rtt_ms > 0.0

    metrics_final = hil.stop_session()
    assert metrics_final.packets_sent == 10
    assert metrics_final.packets_recv == 10
    assert metrics_final.avg_rtt_ms <= 50.0


def test_supervised_hil_link_loss_timeout():
    cfg = HILSessionConfig(link_loss_timeout_sec=1.0)
    hil = SupervisedHILManager(cfg)
    hil.start_session()

    # Initial sync at t=0
    meas = SensorMeasurement(vehicle_id="test", sensor_type="imu")
    act, m = hil.sync_step(meas, current_mono=0.0)
    assert m.is_link_alive is True

    # Jump forward by 2.0 seconds with no incoming packets
    act_lost, m_lost = hil.sync_step(meas, current_mono=2.0)
    assert act_lost is None
    assert m_lost.is_link_alive is False
    assert m_lost.failsafe_triggered is True
