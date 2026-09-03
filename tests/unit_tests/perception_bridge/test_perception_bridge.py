"""Tests for YOLO Perception Bridge and vision telemetry integration."""
import numpy as np
import pytest

from drone_sdk.perception_bridge import (
    CameraIntrinsics,
    VisualDetection,
    VisionTelemetrySource,
    YOLOPerceptionBridge,
)
from drone_sdk.state_manager import DataSource, StateFactory
from drone_sdk.telemetry_engine import SourcePriority, TelemetryEngine


def test_yolo_perception_bridge_detection():
    bridge = YOLOPerceptionBridge()
    detections = bridge.detect_targets()

    assert len(detections) >= 1
    det = detections[0]
    assert isinstance(det, VisualDetection)
    assert det.label == "landing_pad"
    assert det.confidence > 0.8
    assert det.estimated_range_m > 0.0
    assert len(det.relative_pos_cam) == 3


def test_yolo_simulated_target_projection():
    bridge = YOLOPerceptionBridge()
    drone_state = StateFactory.create_initial("uav_vision").copy_with(z=-10.0)
    # Drone at (0, 0, -10), target on ground at (20, 0, -5) -> 5m below drone, 20m ahead
    target_ground = np.array([20.0, 0.0, -5.0])

    detections = bridge.detect_targets(simulated_target_ned=target_ground, drone_state=drone_state)
    assert len(detections) == 1
    det = detections[0]
    assert abs(det.bearing_deg) < 5.0  # Ahead along x-axis
    assert det.elevation_deg > 0.0     # Target is below


def test_perception_to_state_update():
    bridge = YOLOPerceptionBridge()
    detections = bridge.detect_targets()
    landmark_ned = np.array([50.0, 20.0, 0.0])

    update = bridge.create_state_update(detections, vehicle_id="uav_vision", known_target_ned=landmark_ned)
    assert update is not None
    assert update.source == DataSource.VISION
    assert update.x is not None
    assert update.y is not None


def test_vision_telemetry_source():
    source = VisionTelemetrySource(vehicle_id="uav_vision")
    assert source.priority == SourcePriority.ROS2
    source.connect()

    updates = source.poll()
    assert len(updates) == 1
    assert updates[0].source == DataSource.VISION

    source.disconnect()
    assert len(source.poll()) == 0
