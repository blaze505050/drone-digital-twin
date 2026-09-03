"""Tests for Sim-to-Real Command Sinks and MissionExecutor integration."""
import numpy as np
import pytest

from drone_sdk.mission_planner import (
    ControlCommand,
    CommandSink,
    GazeboCommandSink,
    MAVLinkCommandSink,
    Mission,
    MissionExecutor,
    MissionItem,
    GeoPoint,
)
from drone_sdk.state_manager import StateStore, StateFactory


def test_gazebo_command_sink():
    received = []
    sink = GazeboCommandSink(callback=lambda cmd: received.append(cmd))

    cmd = ControlCommand(
        pos_target_ned=np.array([10.0, 5.0, -15.0]),
        vel_target_ned=np.array([2.0, 1.0, 0.0]),
        yaw_target_rad=0.5,
        thrust_normalized=0.6,
    )

    assert sink.send(cmd)
    assert sink.total_dispatched == 1
    assert len(received) == 1
    assert received[0].pos_target_ned[0] == 10.0


def test_mavlink_command_sink():
    sink = MAVLinkCommandSink(target_system=1, target_component=1)

    cmd = ControlCommand(
        pos_target_ned=np.array([5.0, 0.0, -10.0]),
        vel_target_ned=np.array([1.0, 0.0, 0.0]),
        yaw_target_rad=0.0,
        thrust_normalized=0.55,
    )

    assert sink.send(cmd)
    assert sink.total_dispatched == 1
    raw_packet = sink.last_packet
    assert raw_packet is not None
    assert len(raw_packet) >= 40  # Encoded binary struct payload


def test_mission_executor_with_sink():
    home = GeoPoint(lat=47.3977, lon=8.5455, alt=500.0)
    mission = Mission("test_sink_mission", home=home)
    mission.add(MissionItem.waypoint(0, GeoPoint(lat=47.3980, lon=8.5460, alt=510.0), altitude_agl=10.0, speed_ms=5.0))

    store = StateStore.get_or_create("sink_uav")
    store.update(StateFactory.create_initial("sink_uav"))

    sink = GazeboCommandSink()
    executor = MissionExecutor(mission=mission, vehicle_id="sink_uav", sink=sink)

    assert executor.tick()
    assert sink.total_dispatched == 1
    cmd = sink.last_command
    assert cmd is not None
    assert cmd.pos_target_ned is not None
