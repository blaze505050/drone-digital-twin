"""Tests for Glob3R 3D Vision Perception & Pseudo-LiDAR Engine."""
import os
import tempfile
import numpy as np
import pytest

from drone_sdk.perception_bridge import (
    Glob3RPerceptionEngine,
    PointCloud3D,
    VoxelGrid3D,
    ElevationMap2D,
    LandingZoneCandidate,
    PseudoLidarScan,
)
from drone_sdk.state_manager import StateFactory


def test_pointcloud3d_operations():
    # 100 random 3D points
    pts = np.random.uniform(-10.0, 10.0, (100, 3))
    colors = np.full((100, 3), 255, dtype=np.uint8)
    confidences = np.full(100, 0.95, dtype=np.float32)

    cloud = PointCloud3D(pts, colors, confidences)
    assert len(cloud) == 100
    assert not cloud.is_empty()

    # Voxel downsampling
    down = cloud.voxel_downsample(voxel_size_m=2.0)
    assert len(down) < len(cloud)
    assert len(down) > 0

    # Outlier filtering
    filtered = cloud.filter_outliers(nb_neighbors=5, std_ratio=2.0)
    assert len(filtered) <= len(cloud)

    # PLY export
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cloud.to_ply(tmp_path)
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 100
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_pointcloud_ground_plane_segmentation():
    # Synthetic flat ground at z = 0 with a few vertical obstacles
    ground_pts = np.random.uniform(-5.0, 5.0, (80, 3))
    ground_pts[:, 2] = 0.0 + np.random.normal(0.0, 0.02, 80)

    obs_pts = np.random.uniform(1.0, 3.0, (20, 3))
    obs_pts[:, 2] = np.random.uniform(-5.0, -1.0, 20)  # Obstacles above ground

    cloud = PointCloud3D(np.vstack([ground_pts, obs_pts]))
    ground, obstacles, plane = cloud.segment_ground_plane(distance_threshold_m=0.10)

    assert len(ground) > 40
    assert len(obstacles) > 5
    assert len(ground) + len(obstacles) == len(cloud)
    # Ground plane normal should be roughly vertical (NED z-axis)
    assert abs(plane[2]) > 0.60


def test_voxel_grid_occupancy_and_corridor():
    grid = VoxelGrid3D(bounds_x=(-10.0, 10.0), bounds_y=(-10.0, 10.0), bounds_z=(-5.0, 5.0), resolution_m=0.50)

    # Insert point obstacle at (5.0, 0.0, -2.0)
    obstacle = PointCloud3D(np.array([[5.0, 0.0, -2.0]]))
    sensor_origin = np.array([0.0, 0.0, 0.0])
    grid.insert_point_cloud(obstacle, sensor_origin)

    occupied = grid.get_occupied_centroids(threshold_prob=0.60)
    assert len(occupied) >= 1
    # Check that centroid is near (5, 0, -2)
    diff = np.linalg.norm(occupied[0] - np.array([5.0, 0.0, -2.0]))
    assert diff < 0.60

    # Corridor collision checking
    blocked = grid.is_corridor_free(start_ned=np.array([0.0, 0.0, -2.0]), end_ned=np.array([7.0, 0.0, -2.0]), safety_radius_m=0.6)
    assert not blocked  # Should detect obstacle!

    clear = grid.is_corridor_free(start_ned=np.array([0.0, 5.0, -2.0]), end_ned=np.array([7.0, 5.0, -2.0]), safety_radius_m=0.6)
    assert clear  # Parallel path is clear


def test_elevation_map_and_landing_zones():
    elev = ElevationMap2D(bounds_x=(-10.0, 10.0), bounds_y=(-10.0, 10.0), cell_size_m=0.50)

    # Generate flat terrain with small tilt
    x = np.random.uniform(-8.0, 8.0, 300)
    y = np.random.uniform(-8.0, 8.0, 300)
    z = np.full(300, 0.0)  # Perfectly flat ground
    cloud = PointCloud3D(np.column_stack([x, y, z]))

    elev.build_from_cloud(cloud)
    assert not np.all(np.isnan(elev.height_map))

    zones = elev.find_landing_zones(min_radius_m=1.0, max_slope_deg=5.0)
    assert len(zones) >= 1
    best_zone = zones[0]
    assert isinstance(best_zone, LandingZoneCandidate)
    assert best_zone.safety_score > 0.70
    assert best_zone.mean_slope_deg < 5.0


def test_pseudo_lidar_scan_generation():
    engine = Glob3RPerceptionEngine()
    drone_pose = StateFactory.create_initial("lidar_test").copy_with(z=-5.0)

    # Obstacle 4 meters straight ahead
    obs_cloud = PointCloud3D(np.array([
        [4.0, 0.0, -5.0],
        [4.1, 0.2, -5.0],
        [4.0, -0.2, -5.0],
    ]))

    scan = engine.generate_pseudo_lidar(obs_cloud, sensor_origin=drone_pose.position_ned())

    assert isinstance(scan, PseudoLidarScan)
    assert len(scan.ranges_m) == 3
    assert abs(scan.min_distance_m - 4.0) < 0.2
    assert abs(scan.closest_bearing_deg) < 5.0


def test_glob3r_perception_tick_end_to_end():
    engine = Glob3RPerceptionEngine()
    drone_pose = StateFactory.create_initial("glob3r_uav").copy_with(z=-10.0)

    results = engine.process_perception_tick(drone_pose)

    assert "point_cloud" in results
    assert "voxel_grid" in results
    assert "elevation_map" in results
    assert "landing_sites" in results
    assert "pseudo_lidar" in results

    assert isinstance(results["point_cloud"], PointCloud3D)
    assert isinstance(results["voxel_grid"], VoxelGrid3D)
    assert isinstance(results["pseudo_lidar"], PseudoLidarScan)
    assert len(results["point_cloud"]) > 0
