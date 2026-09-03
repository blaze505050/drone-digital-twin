"""
drone_sdk.perception_bridge.glob3r
==================================
Glob3R-Inspired 3D Vision Perception & Pseudo-LiDAR Mapping Engine.

Inspired by the global Structure-from-Motion and 3D vision paradigm (Glob3R):
implements a lightweight, classical geometric 3D vision, volumetric occupancy,
and pseudo-LiDAR mapping pipeline (independent algorithmic implementation
without requiring multi-gigabyte deep foundation model weights). Reconstructs
dense 3D environments, voxel occupancy grids, and elevation maps purely from
camera vision and poses without physical LiDAR hardware.

Core Outputs:
  1. PointCloud3D: Dense 3D pseudo-LiDAR point cloud with voxel downsampling,
     statistical outlier removal, and RANSAC ground plane segmentation.
  2. VoxelGrid3D: Volumetric 3D occupancy grid with log-odds ray-casting for
     obstacle avoidance and 3D flight corridor clearance checking.
  3. ElevationMap2D: 2.5D Digital Elevation Model (DEM) extracting terrain slope,
     roughness, and automated safe landing site candidates.
  4. PseudoLidarScan: Synthetic LiDAR range-azimuth-elevation scan generated from
     vision, providing direct LiDAR-equivalent telemetry to navigation systems.

Python version: 3.9+
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from drone_sdk.perception_bridge import CameraIntrinsics
from drone_sdk.state_manager.schema import DroneStateVector


@dataclass
class LandingZoneCandidate:
    """Evaluated safe landing zone on ground terrain."""
    center_ned:       np.ndarray   # [x, y, z] center coordinates
    radius_m:         float        # Clear landing pad radius
    mean_slope_deg:   float        # Terrain slope (< 5 deg is ideal)
    roughness_m:      float        # Surface roughness std dev (< 0.05m is smooth)
    safety_score:     float        # Composite score [0..1] where 1.0 is optimal

    def to_dict(self) -> dict:
        return {
            "center_ned":     [round(float(c), 3) for c in self.center_ned],
            "radius_m":       round(self.radius_m, 2),
            "slope_deg":      round(self.mean_slope_deg, 2),
            "roughness_cm":   round(self.roughness_m * 100.0, 1),
            "safety_score":   round(self.safety_score, 3),
        }


@dataclass
class PseudoLidarScan:
    """Synthetic LiDAR scan generated from vision."""
    timestamp:          float
    ranges_m:           np.ndarray   # (N,) array of distances
    azimuths_rad:       np.ndarray   # (N,) horizontal angles [-pi..pi]
    elevations_rad:     np.ndarray   # (N,) vertical angles [-pi/2..pi/2]
    intensities:        np.ndarray   # (N,) pseudo reflectance/confidence [0..1]
    min_distance_m:     float
    closest_bearing_deg: float
    hazard_detected:    bool

    def to_dict(self) -> dict:
        return {
            "timestamp":           round(self.timestamp, 3),
            "num_points":          len(self.ranges_m),
            "min_distance_m":      round(self.min_distance_m, 2),
            "closest_bearing_deg": round(self.closest_bearing_deg, 1),
            "hazard_detected":     self.hazard_detected,
        }


class PointCloud3D:
    """Dense 3D point cloud with geometric filtering and segmentation."""

    def __init__(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        confidences: Optional[np.ndarray] = None,
    ) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2 or (self.points.shape[1] != 3 and len(self.points) > 0):
            raise ValueError(f"points must have shape (N, 3), got {self.points.shape}")

        n = len(self.points)
        if colors is not None and len(colors) == n:
            self.colors = np.asarray(colors, dtype=np.uint8)
        else:
            self.colors = np.full((n, 3), 200, dtype=np.uint8)

        if confidences is not None and len(confidences) == n:
            self.confidences = np.asarray(confidences, dtype=np.float32)
        else:
            self.confidences = np.ones(n, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.points)

    def is_empty(self) -> bool:
        return len(self.points) == 0

    def voxel_downsample(self, voxel_size_m: float = 0.15) -> "PointCloud3D":
        """Voxel grid spatial downsampling."""
        if len(self.points) == 0 or voxel_size_m <= 0.0:
            return PointCloud3D(self.points.copy(), self.colors.copy(), self.confidences.copy())

        # Quantize points to voxel keys
        voxel_coords = np.floor(self.points / voxel_size_m).astype(np.int32)
        unique_voxels, indices = np.unique(voxel_coords, axis=0, return_inverse=True)

        down_pts = np.zeros((len(unique_voxels), 3), dtype=np.float64)
        down_cols = np.zeros((len(unique_voxels), 3), dtype=np.float64)
        down_confs = np.zeros(len(unique_voxels), dtype=np.float64)
        counts = np.bincount(indices, minlength=len(unique_voxels))

        for i in range(3):
            down_pts[:, i] = np.bincount(indices, weights=self.points[:, i], minlength=len(unique_voxels)) / counts
            down_cols[:, i] = np.bincount(indices, weights=self.colors[:, i], minlength=len(unique_voxels)) / counts
        down_confs = np.bincount(indices, weights=self.confidences, minlength=len(unique_voxels)) / counts

        return PointCloud3D(down_pts, down_cols.astype(np.uint8), down_confs.astype(np.float32))

    def filter_outliers(self, nb_neighbors: int = 12, std_ratio: float = 1.8) -> "PointCloud3D":
        """Statistical outlier removal to eliminate noisy visual depth artifacts."""
        if len(self.points) <= nb_neighbors:
            return PointCloud3D(self.points.copy(), self.colors.copy(), self.confidences.copy())

        pts = self.points
        # Subsample for distance estimation if cloud is large
        sample_size = min(len(pts), 500)
        indices = np.random.choice(len(pts), sample_size, replace=False)
        sample_pts = pts[indices]

        # Calculate mean distance to nearest neighbors
        dists = np.sqrt(np.sum((pts[:, np.newaxis, :] - sample_pts[np.newaxis, :, :]) ** 2, axis=2))
        dists.sort(axis=1)
        mean_k_dist = np.mean(dists[:, 1:min(nb_neighbors + 1, sample_size)], axis=1)

        thresh = np.mean(mean_k_dist) + std_ratio * np.std(mean_k_dist)
        mask = mean_k_dist < thresh

        return PointCloud3D(self.points[mask], self.colors[mask], self.confidences[mask])

    def segment_ground_plane(
        self,
        distance_threshold_m: float = 0.12,
        max_iterations: int = 60,
    ) -> Tuple["PointCloud3D", "PointCloud3D", np.ndarray]:
        """RANSAC ground plane segmentation separating terrain from vertical obstacles.

        Returns (ground_cloud, obstacle_cloud, plane_eq [a, b, c, d]).
        """
        if len(self.points) < 10:
            empty = PointCloud3D(np.empty((0, 3)))
            return empty, self, np.array([0.0, 0.0, 1.0, 0.0])

        best_inliers: np.ndarray = np.array([], dtype=bool)
        best_plane = np.array([0.0, 0.0, 1.0, 0.0])
        n = len(self.points)

        for _ in range(max_iterations):
            sample_idx = np.random.choice(n, 3, replace=False)
            p1, p2, p3 = self.points[sample_idx]

            # Normal vector
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal /= norm

            # Ground planes are predominantly vertical in NED (normal roughly along +Z or -Z)
            if abs(normal[2]) < 0.70:
                continue

            d = -float(np.dot(normal, p1))
            plane = np.array([normal[0], normal[1], normal[2], d])

            # Distances to plane
            distances = np.abs(self.points @ normal + d)
            inliers = distances < distance_threshold_m

            if np.sum(inliers) > np.sum(best_inliers):
                best_inliers = inliers
                best_plane = plane

        if len(best_inliers) == 0 or np.sum(best_inliers) < 5:
            # Fallback: ground is lowest 20% Z points
            z_thresh = np.percentile(self.points[:, 2], 80)
            best_inliers = self.points[:, 2] >= z_thresh

        ground_cloud = PointCloud3D(
            self.points[best_inliers], self.colors[best_inliers], self.confidences[best_inliers]
        )
        obstacle_cloud = PointCloud3D(
            self.points[~best_inliers], self.colors[~best_inliers], self.confidences[~best_inliers]
        )
        return ground_cloud, obstacle_cloud, best_plane

    def to_ply(self, filepath: str) -> None:
        """Export point cloud to standard Stanford PLY format."""
        header = (
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {len(self.points)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "property float confidence\n"
            "end_header\n"
        )
        lines = []
        for i in range(len(self.points)):
            p = self.points[i]
            c = self.colors[i]
            conf = self.confidences[i]
            lines.append(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]} {conf:.3f}\n")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header)
            f.writelines(lines)


class VoxelGrid3D:
    """Volumetric 3D probabilistic occupancy grid."""

    def __init__(
        self,
        bounds_x: Tuple[float, float] = (-20.0, 20.0),
        bounds_y: Tuple[float, float] = (-20.0, 20.0),
        bounds_z: Tuple[float, float] = (-15.0, 5.0),
        resolution_m: float = 0.40,
    ) -> None:
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.bounds_z = bounds_z
        self.res = resolution_m

        self.dim_x = int(math.ceil((bounds_x[1] - bounds_x[0]) / self.res))
        self.dim_y = int(math.ceil((bounds_y[1] - bounds_y[0]) / self.res))
        self.dim_z = int(math.ceil((bounds_z[1] - bounds_z[0]) / self.res))

        # Log-odds representation: 0.0 = prior P(occ)=0.5
        self.log_odds = np.zeros((self.dim_x, self.dim_y, self.dim_z), dtype=np.float32)

        self.l_occ = 0.85     # Log-odds increase on hit
        self.l_free = -0.40   # Log-odds decrease on miss (free ray)
        self.l_min = -3.5
        self.l_max = 3.5

    def _world_to_grid(self, pt: np.ndarray) -> Optional[Tuple[int, int, int]]:
        ix = int((pt[0] - self.bounds_x[0]) / self.res)
        iy = int((pt[1] - self.bounds_y[0]) / self.res)
        iz = int((pt[2] - self.bounds_z[0]) / self.res)
        if 0 <= ix < self.dim_x and 0 <= iy < self.dim_y and 0 <= iz < self.dim_z:
            return ix, iy, iz
        return None

    def _grid_to_world(self, ix: int, iy: int, iz: int) -> np.ndarray:
        return np.array([
            self.bounds_x[0] + (ix + 0.5) * self.res,
            self.bounds_y[0] + (iy + 0.5) * self.res,
            self.bounds_z[0] + (iz + 0.5) * self.res,
        ])

    def insert_point_cloud(self, cloud: PointCloud3D, sensor_origin: np.ndarray) -> None:
        """Raycast pseudo-LiDAR points into 3D voxel grid updating log-odds."""
        for pt in cloud.points:
            end_idx = self._world_to_grid(pt)
            if end_idx is None:
                continue

            # Update endpoint as occupied
            self.log_odds[end_idx] = min(self.l_max, self.log_odds[end_idx] + self.l_occ)

            # Raycast intermediate points as free space (subsample 4 ray steps)
            vec = pt - sensor_origin
            dist = float(np.linalg.norm(vec))
            if dist > self.res * 2.0:
                n_steps = max(2, min(8, int(dist / (self.res * 1.5))))
                for step in range(1, n_steps):
                    ray_pt = sensor_origin + vec * (step / n_steps)
                    mid_idx = self._world_to_grid(ray_pt)
                    if mid_idx is not None and mid_idx != end_idx:
                        self.log_odds[mid_idx] = max(self.l_min, self.log_odds[mid_idx] + self.l_free)

    def get_occupied_centroids(self, threshold_prob: float = 0.65) -> np.ndarray:
        """Return 3D coordinates of all occupied voxel centers."""
        # P = 1 / (1 + exp(-log_odds))
        thresh_lo = math.log(threshold_prob / (1.0 - threshold_prob))
        occupied_indices = np.argwhere(self.log_odds > thresh_lo)
        if len(occupied_indices) == 0:
            return np.empty((0, 3))

        centroids = []
        for idx in occupied_indices:
            centroids.append(self._grid_to_world(idx[0], idx[1], idx[2]))
        return np.array(centroids)

    def is_corridor_free(
        self,
        start_ned: np.ndarray,
        end_ned: np.ndarray,
        safety_radius_m: float = 0.5,
    ) -> bool:
        """Check if straight flight corridor between start and end is free of obstacles."""
        vec = end_ned - start_ned
        length = float(np.linalg.norm(vec))
        if length < 0.1:
            return True

        n_samples = max(4, int(length / (self.res * 0.75)))
        occupied_pts = self.get_occupied_centroids()
        if len(occupied_pts) == 0:
            return True

        for s in range(n_samples + 1):
            sample_pt = start_ned + vec * (s / n_samples)
            dists = np.linalg.norm(occupied_pts - sample_pt, axis=1)
            if np.any(dists < safety_radius_m):
                return False
        return True


class ElevationMap2D:
    """2.5D Digital Elevation Model (DEM) and terrain hazard analysis."""

    def __init__(
        self,
        bounds_x: Tuple[float, float] = (-20.0, 20.0),
        bounds_y: Tuple[float, float] = (-20.0, 20.0),
        cell_size_m: float = 0.50,
    ) -> None:
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.cell_size = cell_size_m

        self.nx = int(math.ceil((bounds_x[1] - bounds_x[0]) / cell_size_m))
        self.ny = int(math.ceil((bounds_y[1] - bounds_y[0]) / cell_size_m))

        self.height_map = np.full((self.nx, self.ny), np.nan, dtype=np.float32)
        self.slope_map = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.roughness_map = np.zeros((self.nx, self.ny), dtype=np.float32)

    def build_from_cloud(self, cloud: PointCloud3D) -> None:
        """Rasterize point cloud into 2.5D elevation and slope maps."""
        if len(cloud.points) == 0:
            return

        pts = cloud.points
        # Filter points within horizontal bounds
        mask = (
            (pts[:, 0] >= self.bounds_x[0]) & (pts[:, 0] < self.bounds_x[1]) &
            (pts[:, 1] >= self.bounds_y[0]) & (pts[:, 1] < self.bounds_y[1])
        )
        pts_in = pts[mask]
        if len(pts_in) == 0:
            return

        ix = np.floor((pts_in[:, 0] - self.bounds_x[0]) / self.cell_size).astype(np.int32)
        iy = np.floor((pts_in[:, 1] - self.bounds_y[0]) / self.cell_size).astype(np.int32)

        # Populate cell statistics (NED Z: ground is positive / down)
        cell_bins: Dict[Tuple[int, int], List[float]] = {}
        for i in range(len(pts_in)):
            key = (ix[i], iy[i])
            if key not in cell_bins:
                cell_bins[key] = []
            cell_bins[key].append(float(pts_in[i, 2]))

        for (cx, cy), z_vals in cell_bins.items():
            if 0 <= cx < self.nx and 0 <= cy < self.ny:
                self.height_map[cx, cy] = float(np.median(z_vals))
                self.roughness_map[cx, cy] = float(np.std(z_vals)) if len(z_vals) > 1 else 0.01

        # Calculate slope gradient
        valid_mask = ~np.isnan(self.height_map)
        mean_h = float(np.nanmean(self.height_map)) if np.any(valid_mask) else 0.0
        filled = np.where(valid_mask, self.height_map, mean_h)

        grad_x, grad_y = np.gradient(filled, self.cell_size)
        slope_rad = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
        self.slope_map = np.degrees(slope_rad).astype(np.float32)

    def find_landing_zones(
        self,
        min_radius_m: float = 1.0,
        max_slope_deg: float = 6.0,
        max_roughness_m: float = 0.08,
    ) -> List[LandingZoneCandidate]:
        """Detect flat, safe, and unobstructed landing zone candidates."""
        candidates: List[LandingZoneCandidate] = []
        cell_radius = max(1, int(math.ceil(min_radius_m / self.cell_size)))

        for i in range(cell_radius, self.nx - cell_radius, 2):
            for j in range(cell_radius, self.ny - cell_radius, 2):
                if np.isnan(self.height_map[i, j]):
                    continue

                h_sub = self.height_map[i - cell_radius: i + cell_radius + 1, j - cell_radius: j + cell_radius + 1]
                valid_sub = ~np.isnan(h_sub)
                if np.sum(valid_sub) < max(2, int(0.3 * h_sub.size)):
                    continue

                slope_sub = self.slope_map[i - cell_radius: i + cell_radius + 1, j - cell_radius: j + cell_radius + 1]
                rough_sub = self.roughness_map[i - cell_radius: i + cell_radius + 1, j - cell_radius: j + cell_radius + 1]

                mean_slope = float(np.mean(slope_sub[valid_sub]))
                max_slope = float(np.max(slope_sub[valid_sub]))
                mean_rough = float(np.mean(rough_sub[valid_sub]))

                if max_slope <= max_slope_deg and mean_rough <= max_roughness_m:
                    center_x = self.bounds_x[0] + (i + 0.5) * self.cell_size
                    center_y = self.bounds_y[0] + (j + 0.5) * self.cell_size
                    center_z = float(self.height_map[i, j])

                    # Safety score [0..1]
                    slope_factor = max(0.0, 1.0 - (mean_slope / max_slope_deg))
                    rough_factor = max(0.0, 1.0 - (mean_rough / max_roughness_m))
                    score = 0.6 * slope_factor + 0.4 * rough_factor

                    candidates.append(LandingZoneCandidate(
                        center_ned=np.array([center_x, center_y, center_z]),
                        radius_m=min_radius_m,
                        mean_slope_deg=mean_slope,
                        roughness_m=mean_rough,
                        safety_score=score,
                    ))

        candidates.sort(key=lambda c: c.safety_score, reverse=True)
        return candidates


class Glob3RPerceptionEngine:
    """Full Glob3R 3D Vision Perception & Pseudo-LiDAR Engine."""

    def __init__(
        self,
        camera: Optional[CameraIntrinsics] = None,
        voxel_resolution_m: float = 0.35,
    ) -> None:
        self.camera = camera or CameraIntrinsics()
        self.voxel_grid = VoxelGrid3D(resolution_m=voxel_resolution_m)
        self.elevation_map = ElevationMap2D()
        self.latest_cloud: Optional[PointCloud3D] = None
        self.latest_scan: Optional[PseudoLidarScan] = None

    def reconstruct_from_depth(
        self,
        depth_map: np.ndarray,
        drone_pose: DroneStateVector,
        rgb_image: Optional[np.ndarray] = None,
    ) -> PointCloud3D:
        """Backproject 2D dense depth map into 3D world NED point cloud."""
        cam = self.camera
        h, w = depth_map.shape
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

        z_cam = depth_map.flatten()
        valid = (z_cam > 0.3) & (z_cam < 50.0)

        x_cam = (u_coords.flatten()[valid] - cam.cx) * z_cam[valid] / cam.fx
        y_cam = (v_coords.flatten()[valid] - cam.cy) * z_cam[valid] / cam.fy
        z_cam_v = z_cam[valid]

        # Camera frame: X right, Y down, Z forward
        # Drone body frame: X forward, Y right, Z down
        pts_body = np.column_stack([z_cam_v, x_cam, y_cam])

        # Body to NED rotation from quaternion
        q0, q1, q2, q3 = drone_pose.q0, drone_pose.q1, drone_pose.q2, drone_pose.q3
        R_b2ned = np.array([
            [1.0 - 2.0*(q2*q2 + q3*q3), 2.0*(q1*q2 - q0*q3),       2.0*(q1*q3 + q0*q2)],
            [2.0*(q1*q2 + q0*q3),       1.0 - 2.0*(q1*q1 + q3*q3), 2.0*(q2*q3 - q0*q1)],
            [2.0*(q1*q3 - q0*q2),       2.0*(q2*q3 + q0*q1),       1.0 - 2.0*(q1*q1 + q2*q2)],
        ])

        drone_pos = drone_pose.position_ned()
        pts_ned = (R_b2ned @ pts_body.T).T + drone_pos

        colors = None
        if rgb_image is not None and rgb_image.shape[:2] == (h, w):
            colors = rgb_image.reshape(-1, 3)[valid]

        cloud = PointCloud3D(pts_ned, colors)
        self.latest_cloud = cloud
        return cloud

    def generate_synthetic_scene_cloud(
        self,
        drone_pose: DroneStateVector,
        n_points: int = 400,
        add_obstacles: bool = True,
    ) -> PointCloud3D:
        """Generate high-fidelity representative 3D point cloud for unit tests and headless SITL."""
        drone_pos = drone_pose.position_ned()

        # 1. Ground terrain points around vehicle
        gx = np.random.uniform(-15.0, 15.0, int(n_points * 0.7)) + drone_pos[0]
        gy = np.random.uniform(-15.0, 15.0, int(n_points * 0.7)) + drone_pos[1]
        # Ground at Z=0 (with gentle slope & roughness)
        gz = 0.02 * gx + 0.01 * gy + np.random.normal(0.0, 0.02, len(gx))
        ground_pts = np.column_stack([gx, gy, gz])

        # 2. Obstacle clusters (e.g. wall/building at x=10m)
        obs_pts = np.empty((0, 3))
        if add_obstacles:
            n_obs = int(n_points * 0.3)
            ox = np.full(n_obs, drone_pos[0] + 8.0) + np.random.normal(0.0, 0.2, n_obs)
            oy = np.random.uniform(-4.0, 4.0, n_obs) + drone_pos[1]
            oz = np.random.uniform(-8.0, 0.0, n_obs)  # Vertical obstacle
            obs_pts = np.column_stack([ox, oy, oz])

        all_pts = np.vstack([ground_pts, obs_pts])
        colors = np.full((len(all_pts), 3), 180, dtype=np.uint8)
        confidences = np.random.uniform(0.85, 0.98, len(all_pts)).astype(np.float32)

        cloud = PointCloud3D(all_pts, colors, confidences)
        self.latest_cloud = cloud
        return cloud

    def generate_pseudo_lidar(
        self,
        cloud: PointCloud3D,
        sensor_origin: np.ndarray,
        max_range_m: float = 30.0,
        fov_azimuth_deg: float = 120.0,
    ) -> PseudoLidarScan:
        """Compute spherical range, azimuth, and elevation pseudo-LiDAR scan."""
        if len(cloud.points) == 0:
            return PseudoLidarScan(
                timestamp=time.time(),
                ranges_m=np.array([]),
                azimuths_rad=np.array([]),
                elevations_rad=np.array([]),
                intensities=np.array([]),
                min_distance_m=float("inf"),
                closest_bearing_deg=0.0,
                hazard_detected=False,
            )

        vecs = cloud.points - sensor_origin
        dists = np.linalg.norm(vecs, axis=1)

        # Filter to field of view and max range
        half_fov_rad = math.radians(fov_azimuth_deg * 0.5)
        azimuths = np.arctan2(vecs[:, 1], vecs[:, 0])
        elevations = np.arcsin(np.clip(-vecs[:, 2] / np.maximum(1e-4, dists), -1.0, 1.0))

        mask = (dists <= max_range_m) & (np.abs(azimuths) <= half_fov_rad) & (dists > 0.2)
        if not np.any(mask):
            mask = dists > 0.0

        r_sub = dists[mask]
        az_sub = azimuths[mask]
        el_sub = elevations[mask]
        int_sub = cloud.confidences[mask]

        min_dist = float(np.min(r_sub)) if len(r_sub) > 0 else float("inf")
        min_idx = int(np.argmin(r_sub)) if len(r_sub) > 0 else 0
        closest_bearing = math.degrees(az_sub[min_idx]) if len(az_sub) > 0 else 0.0
        hazard = min_dist < 2.5  # Critical proximity hazard

        scan = PseudoLidarScan(
            timestamp=time.time(),
            ranges_m=r_sub,
            azimuths_rad=az_sub,
            elevations_rad=el_sub,
            intensities=int_sub,
            min_distance_m=min_dist,
            closest_bearing_deg=closest_bearing,
            hazard_detected=hazard,
        )
        self.latest_scan = scan
        return scan

    def process_perception_tick(
        self,
        drone_pose: DroneStateVector,
        depth_map: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[PointCloud3D, VoxelGrid3D, ElevationMap2D, PseudoLidarScan, List[LandingZoneCandidate]]]:
        """Execute full Glob3R pipeline for a single perception cycle.

        Returns full dictionary of 3D maps, pseudo-LiDAR scans, and landing evaluations.
        """
        sensor_pos = drone_pose.position_ned()

        # 1. 3D Point cloud extraction
        if depth_map is not None:
            cloud = self.reconstruct_from_depth(depth_map, drone_pose)
        else:
            cloud = self.generate_synthetic_scene_cloud(drone_pose)

        # 2. Clean cloud with voxel downsampling and outlier removal
        filtered_cloud = cloud.voxel_downsample(voxel_size_m=0.20).filter_outliers()

        # 3. Ground plane segmentation
        ground_cloud, obstacle_cloud, _ = filtered_cloud.segment_ground_plane()

        # 4. Update 3D volumetric occupancy voxel grid
        self.voxel_grid.insert_point_cloud(obstacle_cloud, sensor_pos)

        # 5. Build 2.5D elevation map and detect landing zones
        self.elevation_map.build_from_cloud(ground_cloud)
        landing_sites = self.elevation_map.find_landing_zones(min_radius_m=1.0)

        # 6. Generate pseudo-LiDAR scan
        pseudo_scan = self.generate_pseudo_lidar(obstacle_cloud, sensor_pos)

        return {
            "point_cloud":    filtered_cloud,
            "ground_cloud":   ground_cloud,
            "obstacle_cloud": obstacle_cloud,
            "voxel_grid":     self.voxel_grid,
            "elevation_map":  self.elevation_map,
            "landing_sites":  landing_sites,
            "pseudo_lidar":   pseudo_scan,
        }
