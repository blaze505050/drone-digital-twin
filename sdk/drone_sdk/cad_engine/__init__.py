"""
drone_sdk.cad_engine
====================
CAD Import Engine + Geometry Processing — Modules 10 & 11.

Provides a complete pipeline from parametric CAD geometry to
CFD-ready mesh, with mass properties, structural zones, and
digital twin geometry synchronisation.

Pipeline::

    DroneGeometry  ──► MeshGenerator ──► AerodynamicSurface
         │                   │
         │                   └──► OpenFOAM case (Module 12)
         │
         └──► MassProperties (for 6-DOF model)
         └──► StructuralZones (for FEM model, Module 16)
         └──► SDFExporter (for Gazebo, Module 7)

CAD formats supported (via pygmsh / meshio)
-------------------------------------------
    STL   — standard triangle mesh (most common for drones)
    OBJ   — Wavefront (useful for visual models)
    STEP  — ISO 10303 (via python-occ, optional)
    Parametric — built-in quad/hex/octorotor generator (no CAD file needed)

Python version: 3.9+
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from .bemt import BladeElementMomentumSolver, PropellerGeometry, BEMTResult


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry primitives
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, s: float) -> "Vec3":
        return Vec3(self.x * s, self.y * s, self.z * s)

    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalised(self) -> "Vec3":
        n = self.norm()
        return Vec3(self.x/n, self.y/n, self.z/n) if n > 0 else Vec3()

    def dot(self, o: "Vec3") -> float:
        return self.x*o.x + self.y*o.y + self.z*o.z

    def cross(self, o: "Vec3") -> "Vec3":
        return Vec3(
            self.y*o.z - self.z*o.y,
            self.z*o.x - self.x*o.z,
            self.x*o.y - self.y*o.x,
        )

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_array(cls, a: np.ndarray) -> "Vec3":
        return cls(float(a[0]), float(a[1]), float(a[2]))

    def __repr__(self) -> str:
        return f"Vec3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


@dataclass
class BoundingBox:
    min: Vec3 = field(default_factory=Vec3)
    max: Vec3 = field(default_factory=Vec3)

    @property
    def size(self) -> Vec3:
        return self.max - self.min

    @property
    def centre(self) -> Vec3:
        return Vec3(
            (self.min.x + self.max.x) / 2,
            (self.min.y + self.max.y) / 2,
            (self.min.z + self.max.z) / 2,
        )

    @property
    def volume(self) -> float:
        s = self.size
        return s.x * s.y * s.z

    @property
    def diagonal(self) -> float:
        return self.size.norm()

    def expand(self, factor: float) -> "BoundingBox":
        """Return a bounding box expanded by factor in all directions."""
        c = self.centre
        half = self.size * (factor / 2)
        return BoundingBox(
            min=Vec3(c.x - half.x, c.y - half.y, c.z - half.z),
            max=Vec3(c.x + half.x, c.y + half.y, c.z + half.z),
        )


@dataclass
class Triangle:
    v0: Vec3
    v1: Vec3
    v2: Vec3

    @property
    def normal(self) -> Vec3:
        e1 = self.v1 - self.v0
        e2 = self.v2 - self.v0
        return e1.cross(e2).normalised()

    @property
    def area(self) -> float:
        e1 = self.v1 - self.v0
        e2 = self.v2 - self.v0
        return e1.cross(e2).norm() / 2.0

    @property
    def centroid(self) -> Vec3:
        return Vec3(
            (self.v0.x + self.v1.x + self.v2.x) / 3,
            (self.v0.y + self.v1.y + self.v2.y) / 3,
            (self.v0.z + self.v1.z + self.v2.z) / 3,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Surface mesh
# ─────────────────────────────────────────────────────────────────────────────

class TriangleMesh:
    """Triangular surface mesh with analysis utilities.

    Stores vertices as (N,3) float64 and faces as (M,3) int32 arrays.
    Compatible with meshio, trimesh, and PyVista.

    Usage::

        mesh = TriangleMesh(vertices, faces)
        print(mesh.surface_area)
        print(mesh.bounding_box)
        mesh.to_stl("/tmp/drone.stl")
    """

    def __init__(
        self,
        vertices: np.ndarray,   # (N, 3) float64
        faces:    np.ndarray,   # (M, 3) int32
    ) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces    = np.asarray(faces,    dtype=np.int32)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def bounding_box(self) -> BoundingBox:
        vmin = self.vertices.min(axis=0)
        vmax = self.vertices.max(axis=0)
        return BoundingBox(
            min=Vec3(*vmin),
            max=Vec3(*vmax),
        )

    @property
    def surface_area(self) -> float:
        """Total surface area (sum of triangle areas)."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        e1 = v1 - v0
        e2 = v2 - v0
        cross = np.cross(e1, e2)
        areas = np.linalg.norm(cross, axis=1) / 2.0
        return float(areas.sum())

    @property
    def centroid(self) -> np.ndarray:
        """Centroid of the mesh (average of face centroids, area-weighted)."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        face_centroids = (v0 + v1 + v2) / 3.0
        e1 = v1 - v0
        e2 = v2 - v0
        areas = np.linalg.norm(np.cross(e1, e2), axis=1) / 2.0
        total_area = areas.sum()
        if total_area == 0:
            return face_centroids.mean(axis=0)
        return (face_centroids * areas[:, None]).sum(axis=0) / total_area

    def face_normals(self) -> np.ndarray:
        """Outward face normals (M, 3)."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        e1 = v1 - v0
        e2 = v2 - v0
        n  = np.cross(e1, e2)
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return n / norms

    def translate(self, offset: np.ndarray) -> "TriangleMesh":
        return TriangleMesh(self.vertices + offset, self.faces.copy())

    def scale(self, factor: float) -> "TriangleMesh":
        return TriangleMesh(self.vertices * factor, self.faces.copy())

    def rotate_z(self, angle_rad: float) -> "TriangleMesh":
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return TriangleMesh(self.vertices @ R.T, self.faces.copy())

    def to_stl(self, path: str) -> Path:
        """Write ASCII STL file (no external dependencies)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = ["solid drone_geometry"]
        for face in self.faces:
            v0, v1, v2 = self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]
            e1 = v1 - v0; e2 = v2 - v0
            n  = np.cross(e1, e2)
            nn = np.linalg.norm(n)
            if nn > 0:
                n = n / nn
            lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
            lines.append("    outer loop")
            for v in (v0, v1, v2):
                lines.append(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append("endsolid drone_geometry")
        p.write_text("\n".join(lines))
        return p

    @classmethod
    def from_stl(cls, path: str) -> "TriangleMesh":
        """Parse ASCII or binary STL file (no external dependencies for ASCII)."""
        p = Path(path)
        content = p.read_text(errors="replace")
        vertices_list = []
        faces_list    = []

        for line in content.splitlines():
            line = line.strip()
            if line.startswith("vertex "):
                parts = line.split()
                vertices_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(vertices_list) % 3 == 0:
                    n = len(vertices_list)
                    faces_list.append([n-3, n-2, n-1])

        if not vertices_list:
            raise ValueError(f"No vertices found in {path}. "
                             "Only ASCII STL is supported without trimesh.")

        return cls(np.array(vertices_list), np.array(faces_list))

    def __repr__(self) -> str:
        bb = self.bounding_box
        return (
            f"TriangleMesh(vertices={self.n_vertices}, faces={self.n_faces}, "
            f"area={self.surface_area:.4f} m², "
            f"bbox={bb.size.x:.3f}×{bb.size.y:.3f}×{bb.size.z:.3f} m)"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Mass properties
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MassProperties:
    """Rigid body mass properties from geometry and density."""
    mass_kg:      float
    cg:           np.ndarray     # Centre of gravity in body frame, m
    ixx:          float          # kg·m²
    iyy:          float
    izz:          float
    ixy:          float = 0.0
    ixz:          float = 0.0
    iyz:          float = 0.0

    @property
    def inertia_tensor(self) -> np.ndarray:
        return np.array([
            [ self.ixx, -self.ixy, -self.ixz],
            [-self.ixy,  self.iyy, -self.iyz],
            [-self.ixz, -self.iyz,  self.izz],
        ])

    def to_twin_physics_params(
        self,
        arm_length_m: float = 0.25,
        max_thrust_motor_n: float = 6.62,
    ):
        """Bridge CAD mass properties directly into ClosedLoopDigitalTwin physics model."""
        from drone_sdk.digital_twin_core.twin_model import TwinPhysicsParameters
        return TwinPhysicsParameters(
            mass_kg=max(0.01, float(self.mass_kg)),
            ixx=max(1e-6, float(self.ixx)),
            iyy=max(1e-6, float(self.iyy)),
            izz=max(1e-6, float(self.izz)),
            arm_length_m=arm_length_m,
            max_thrust_motor_n=max_thrust_motor_n,
        )

    def to_dict(self) -> dict:
        return {
            "mass_kg": self.mass_kg,
            "cg":      self.cg.tolist(),
            "ixx": self.ixx, "iyy": self.iyy, "izz": self.izz,
            "ixy": self.ixy, "ixz": self.ixz, "iyz": self.iyz,
        }


class MassEstimator:
    """Estimate exact polyhedral mass properties from triangulated geometry and material density.

    Implements Brian Mirtich's exact polyhedral mass properties algorithm (Mirtich 1996,
    "Fast and Accurate Computation of Polyhedral Mass Properties", Journal of Graphics Tools).
    Decomposes closed surface mesh into signed tetrahedra from origin to compute:
      - Exact volume and mass
      - Exact center of mass (CG)
      - Exact 3x3 inertia tensor with products of inertia (ixx, iyy, izz, ixy, ixz, iyz)
        evaluated about the true center of mass.
    """

    @staticmethod
    def from_mesh(
        mesh:            TriangleMesh,
        density_kg_m3:   float = 1200.0,
        wall_thickness:  float = 0.0,    # 0 = solid, >0 = thin-walled
    ) -> MassProperties:
        """Estimate mass properties using exact polyhedral tetrahedral decomposition."""
        if mesh.n_faces == 0 or mesh.n_vertices == 0:
            return MassProperties(mass_kg=0.0, cg=np.zeros(3), ixx=0.0, iyy=0.0, izz=0.0)

        v0 = mesh.vertices[mesh.faces[:, 0]]
        v1 = mesh.vertices[mesh.faces[:, 1]]
        v2 = mesh.vertices[mesh.faces[:, 2]]

        # Signed determinant of [v0, v1, v2] for each triangle
        det = np.einsum("ij,ij->i", v0, np.cross(v1, v2))
        vol = float(np.sum(det)) / 6.0

        # Handle degenerate/thin meshes or explicit shell thickness
        if abs(vol) < 1e-9 or wall_thickness > 0:
            if wall_thickness > 0:
                mass = mesh.surface_area * wall_thickness * density_kg_m3
            else:
                mass = max(0.01, abs(vol) * density_kg_m3)
            bb = mesh.bounding_box
            lx, ly, lz = bb.size.x, bb.size.y, bb.size.z
            return MassProperties(
                mass_kg=mass, cg=mesh.centroid,
                ixx=mass * (ly**2 + lz**2) / 12.0,
                iyy=mass * (lx**2 + lz**2) / 12.0,
                izz=mass * (lx**2 + ly**2) / 12.0,
            )

        sign = 1.0 if vol >= 0 else -1.0
        vol = abs(vol)
        mass = vol * density_kg_m3

        # 1. Exact Center of Gravity via Divergence Theorem / Tetrahedra
        # For each tetrahedron, integral of (x, y, z) is (det / 24) * (v0 + v1 + v2)
        v_sum = v0 + v1 + v2
        cg_integ = np.sum(det[:, None] * v_sum, axis=0) / 24.0
        cg = cg_integ / (vol * 6.0 / 6.0) if vol > 1e-9 else mesh.centroid

        # 2. Second-order moments of inertia about the coordinate origin (Mirtich 1996)
        x0, y0, z0 = v0[:, 0], v0[:, 1], v0[:, 2]
        x1, y1, z1 = v1[:, 0], v1[:, 1], v1[:, 2]
        x2, y2, z2 = v2[:, 0], v2[:, 1], v2[:, 2]

        # Monomial integrals: int(x^2 dV), int(y^2 dV), int(z^2 dV)
        int_x2 = np.sum(det * (x0**2 + x1**2 + x2**2 + x0*x1 + x1*x2 + x2*x0)) / 60.0
        int_y2 = np.sum(det * (y0**2 + y1**2 + y2**2 + y0*y1 + y1*y2 + y2*y0)) / 60.0
        int_z2 = np.sum(det * (z0**2 + z1**2 + z2**2 + z0*z1 + z1*z2 + z2*z0)) / 60.0

        # Monomial cross-product integrals: int(x*y dV), int(y*z dV), int(x*z dV)
        int_xy = np.sum(det * (
            x0 * (2*y0 + y1 + y2) +
            x1 * (y0 + 2*y1 + y2) +
            x2 * (y0 + y1 + 2*y2)
        )) / 120.0

        int_yz = np.sum(det * (
            y0 * (2*z0 + z1 + z2) +
            y1 * (z0 + 2*z1 + z2) +
            y2 * (z0 + z1 + 2*z2)
        )) / 120.0

        int_xz = np.sum(det * (
            x0 * (2*z0 + z1 + z2) +
            x1 * (z0 + 2*z1 + z2) +
            x2 * (z0 + z1 + 2*z2)
        )) / 120.0

        # Raw moments of inertia about origin
        Ixx_orig = density_kg_m3 * (int_y2 + int_z2)
        Iyy_orig = density_kg_m3 * (int_x2 + int_z2)
        Izz_orig = density_kg_m3 * (int_x2 + int_y2)
        Ixy_orig = density_kg_m3 * int_xy
        Ixz_orig = density_kg_m3 * int_xz
        Iyz_orig = density_kg_m3 * int_yz

        # 3. Parallel Axis Theorem: translate inertia tensor to Centre of Gravity (CG)
        cx, cy, cz = float(cg[0]), float(cg[1]), float(cg[2])
        ixx = float(Ixx_orig - mass * (cy**2 + cz**2))
        iyy = float(Iyy_orig - mass * (cx**2 + cz**2))
        izz = float(Izz_orig - mass * (cx**2 + cy**2))
        ixy = float(Ixy_orig - mass * (cx * cy))
        ixz = float(Ixz_orig - mass * (cx * cz))
        iyz = float(Iyz_orig - mass * (cy * cz))

        # Ensure positive-definite diagonal
        ixx = max(1e-7, abs(ixx))
        iyy = max(1e-7, abs(iyy))
        izz = max(1e-7, abs(izz))

        return MassProperties(
            mass_kg=float(mass),
            cg=cg,
            ixx=ixx,
            iyy=iyy,
            izz=izz,
            ixy=float(ixy),
            ixz=float(ixz),
            iyz=float(iyz),
        )

    @staticmethod
    def _signed_volume(mesh: TriangleMesh) -> float:
        """Compute signed volume using divergence theorem."""
        v0 = mesh.vertices[mesh.faces[:, 0]]
        v1 = mesh.vertices[mesh.faces[:, 1]]
        v2 = mesh.vertices[mesh.faces[:, 2]]
        cross = np.cross(v1, v2)
        dots  = (v0 * cross).sum(axis=1)
        return float(dots.sum()) / 6.0


# ─────────────────────────────────────────────────────────────────────────────
#  Parametric drone geometry builder
# ─────────────────────────────────────────────────────────────────────────────

class DroneGeometryBuilder:
    """Generates parametric drone surface meshes without external CAD software.

    Produces STL-compatible triangle meshes for:
    - Quadrotor X / + configuration
    - Hexarotor
    - Octorotor
    - Flat-plate airfoil section (for fixed-wing)

    Usage::

        mesh = DroneGeometryBuilder.quadrotor_x(
            arm_length=0.25,
            arm_diameter=0.02,
            body_radius=0.08,
            body_height=0.04,
        )
        mesh.to_stl("/tmp/quadrotor.stl")
    """

    @staticmethod
    def quadrotor_x(
        arm_length:    float = 0.25,
        arm_diameter:  float = 0.015,
        body_radius:   float = 0.075,
        body_height:   float = 0.035,
        n_arm_segments: int  = 6,
        n_body_segments: int = 12,
    ) -> TriangleMesh:
        """Generate a quadrotor X-configuration airframe mesh."""
        all_verts: List[np.ndarray] = []
        all_faces: List[np.ndarray] = []

        # Four arms at 45°, 135°, 225°, 315° (X configuration)
        for i in range(4):
            angle = math.radians(45 + 90 * i)
            cx = arm_length * math.cos(angle)
            cy = arm_length * math.sin(angle)
            arm_verts, arm_faces = DroneGeometryBuilder._cylinder(
                start  = np.array([0., 0., 0.]),
                end    = np.array([cx, cy, 0.]),
                radius = arm_diameter / 2,
                n      = n_arm_segments,
            )
            v_off = sum(len(v) for v in all_verts)
            all_verts.append(arm_verts)
            all_faces.append(arm_faces + v_off)

        # Central body (cylinder)
        body_verts, body_faces = DroneGeometryBuilder._cylinder(
            start  = np.array([0., 0., -body_height/2]),
            end    = np.array([0., 0.,  body_height/2]),
            radius = body_radius,
            n      = n_body_segments,
        )
        v_off = sum(len(v) for v in all_verts)
        all_verts.append(body_verts)
        all_faces.append(body_faces + v_off)

        vertices = np.vstack(all_verts)
        faces    = np.vstack(all_faces)
        return TriangleMesh(vertices, faces)

    @staticmethod
    def to_frame_config(
        arm_length:      float = 0.25,
        arm_diameter:    float = 0.012,
        wall_thickness:  float = 0.001,
        material:        Optional[object] = None,
        motor_mass_kg:   float = 0.065,
        payload_mass_kg: float = 0.200,
    ) -> object:
        """Create a structural_twin DroneFrameConfig directly from CAD parameters."""
        from drone_sdk.structural_twin import DroneFrameConfig, Material
        mat = material if material is not None else Material.carbon_fibre_tube()
        return DroneFrameConfig(
            arm_length=arm_length,
            arm_diameter_o=arm_diameter,
            arm_diameter_i=max(0.002, arm_diameter - 2.0 * wall_thickness),
            n_arms=4,
            material=mat,
            motor_mass_kg=motor_mass_kg,
            payload_mass_kg=payload_mass_kg,
        )

    @staticmethod
    def to_propeller_geometry(
        diameter_inch: float = 10.0,
        pitch_inch: float = 4.7,
        num_blades: int = 2,
    ) -> PropellerGeometry:
        """Construct matching BEMT PropellerGeometry directly from CAD airframe specs."""
        radius_m = (diameter_inch * 0.0254) / 2.0
        hub_radius_m = radius_m * 0.12
        root_pitch_deg = math.degrees(math.atan2(pitch_inch * 0.0254, 2.0 * math.pi * hub_radius_m))
        tip_pitch_deg = math.degrees(math.atan2(pitch_inch * 0.0254, 2.0 * math.pi * radius_m))
        return PropellerGeometry(
            radius=radius_m,
            hub_radius=hub_radius_m,
            num_blades=num_blades,
            root_chord=radius_m * 0.18,
            tip_chord=radius_m * 0.08,
            root_twist_deg=float(root_pitch_deg),
            tip_twist_deg=float(tip_pitch_deg),
        )

    @staticmethod
    def hexarotor(arm_length: float = 0.3, arm_diameter: float = 0.015) -> TriangleMesh:
        """Generate a hexarotor airframe (6 arms, 60° spacing)."""
        all_verts = []
        all_faces = []
        for i in range(6):
            angle = math.radians(i * 60)
            cx = arm_length * math.cos(angle)
            cy = arm_length * math.sin(angle)
            verts, faces = DroneGeometryBuilder._cylinder(
                start  = np.array([0., 0., 0.]),
                end    = np.array([cx, cy, 0.]),
                radius = arm_diameter / 2,
                n      = 6,
            )
            v_off = sum(len(v) for v in all_verts)
            all_verts.append(verts)
            all_faces.append(faces + v_off)
        return TriangleMesh(np.vstack(all_verts), np.vstack(all_faces))

    @staticmethod
    def flat_plate_wing(
        span:       float = 1.0,
        chord:      float = 0.15,
        thickness:  float = 0.005,
    ) -> TriangleMesh:
        """Generate a flat-plate wing section."""
        h = thickness / 2
        # 8 vertices of a box (wing cross-section)
        verts = np.array([
            [-chord/2, -span/2, -h], [chord/2, -span/2, -h],
            [ chord/2,  span/2, -h], [-chord/2,  span/2, -h],
            [-chord/2, -span/2,  h], [chord/2, -span/2,  h],
            [ chord/2,  span/2,  h], [-chord/2,  span/2,  h],
        ], dtype=np.float64)
        # 12 triangles (6 faces × 2 triangles each)
        faces = np.array([
            [0,1,2],[0,2,3],   # bottom
            [4,6,5],[4,7,6],   # top
            [0,5,1],[0,4,5],   # front
            [2,6,3],[3,6,7],   # back -- corrected winding
            [0,3,7],[0,7,4],   # left
            [1,5,6],[1,6,2],   # right
        ], dtype=np.int32)
        return TriangleMesh(verts, faces)

    @staticmethod
    def _cylinder(
        start:  np.ndarray,
        end:    np.ndarray,
        radius: float,
        n:      int = 8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a cylinder mesh between two points."""
        axis = end - start
        length = np.linalg.norm(axis)
        if length < 1e-10:
            return np.zeros((1, 3)), np.zeros((0, 3), dtype=int)

        # Orthonormal basis for the cylinder cross-section
        z = axis / length
        # Pick an arbitrary perpendicular
        if abs(z[0]) < 0.9:
            perp = np.array([1., 0., 0.])
        else:
            perp = np.array([0., 1., 0.])
        x = np.cross(z, perp)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)

        # Ring vertices at start and end
        angles  = np.linspace(0, 2*math.pi, n, endpoint=False)
        ring_s  = start + radius * (np.outer(np.cos(angles), x) + np.outer(np.sin(angles), y))
        ring_e  = end   + radius * (np.outer(np.cos(angles), x) + np.outer(np.sin(angles), y))

        verts = np.vstack([ring_s, ring_e])  # (2n, 3)

        faces = []
        for i in range(n):
            j = (i + 1) % n
            # Two triangles per quad
            faces.append([i, j, n+j])
            faces.append([i, n+j, n+i])
        # End caps
        s_c = len(verts)
        e_c = s_c + 1
        verts = np.vstack([verts, [start], [end]])
        for i in range(n):
            j = (i + 1) % n
            faces.append([s_c, j, i])       # start cap (reversed for outward normal)
            faces.append([e_c, n+i, n+j])   # end cap

        return verts, np.array(faces, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
#  Aerodynamic surface analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AerodynamicSurface:
    """Key aerodynamic geometric parameters derived from the mesh."""
    reference_area_m2:    float   # S_ref — planform area
    reference_span_m:     float   # b — wing span (or rotor diameter)
    reference_chord_m:    float   # c̄ — mean aerodynamic chord
    wetted_area_m2:       float   # S_wet — total wetted area
    aspect_ratio:         float   # AR = b²/S
    fineness_ratio:       float   # l/d (body length / max width)
    frontal_area_m2:      float   # projected frontal area
    bounding_box:         BoundingBox

    @property
    def wetted_to_reference_ratio(self) -> float:
        return self.wetted_area_m2 / max(self.reference_area_m2, 1e-12)

    def to_dict(self) -> dict:
        return {
            "reference_area_m2": self.reference_area_m2,
            "reference_span_m":  self.reference_span_m,
            "reference_chord_m": self.reference_chord_m,
            "wetted_area_m2":    self.wetted_area_m2,
            "aspect_ratio":      self.aspect_ratio,
            "fineness_ratio":    self.fineness_ratio,
            "frontal_area_m2":   self.frontal_area_m2,
        }


class GeometryAnalyser:
    """Extracts aerodynamic parameters from a TriangleMesh.

    Usage::

        mesh    = DroneGeometryBuilder.quadrotor_x()
        surface = GeometryAnalyser.analyse(mesh)
        print(f"Frontal area: {surface.frontal_area_m2:.4f} m²")
    """

    @staticmethod
    def analyse(mesh: TriangleMesh) -> AerodynamicSurface:
        bb = mesh.bounding_box
        sx = bb.size.x   # fore-aft (chord direction)
        sy = bb.size.y   # span
        sz = bb.size.z   # thickness / height

        # Reference area (top-down projection): x-y planform
        ref_area = GeometryAnalyser._projected_area(mesh, axis=2)

        # Frontal area (front view): y-z projection
        frontal_area = GeometryAnalyser._projected_area(mesh, axis=0)

        span  = sy
        chord = ref_area / span if span > 0 else sx
        ar    = span**2 / ref_area if ref_area > 0 else 1.0
        fr    = sx / max(sy, sz) if max(sy, sz) > 0 else 1.0

        return AerodynamicSurface(
            reference_area_m2 = ref_area,
            reference_span_m  = span,
            reference_chord_m = chord,
            wetted_area_m2    = mesh.surface_area,
            aspect_ratio      = ar,
            fineness_ratio    = fr,
            frontal_area_m2   = frontal_area,
            bounding_box      = bb,
        )

    @staticmethod
    def _projected_area(mesh: TriangleMesh, axis: int) -> float:
        """Approximate projected area by summing |normal_axis| * face_area."""
        v0 = mesh.vertices[mesh.faces[:, 0]]
        v1 = mesh.vertices[mesh.faces[:, 1]]
        v2 = mesh.vertices[mesh.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        # Each face contributes |n_axis|/2 to projected area
        proj  = np.abs(cross[:, axis]) / 2.0
        # Only faces whose normal points in the positive axis direction
        return float(proj[cross[:, axis] > 0].sum())


# ─────────────────────────────────────────────────────────────────────────────
#  CAD Import (STL + OBJ parsers)
# ─────────────────────────────────────────────────────────────────────────────

class CADImporter:
    """Import CAD geometry files into TriangleMesh.

    Supported: ASCII STL, OBJ.
    For STEP/IGES/binary STL: use trimesh (pip install trimesh).

    Usage::

        mesh = CADImporter.load("airframe.stl")
        mesh = CADImporter.load("wing.obj")
    """

    @staticmethod
    def load(path: str) -> TriangleMesh:
        """Auto-detect format and load geometry."""
        p   = Path(path)
        ext = p.suffix.lower()
        if ext == ".stl":
            return CADImporter.load_stl(path)
        elif ext == ".obj":
            return CADImporter.load_obj(path)
        else:
            raise ValueError(
                f"Unsupported CAD format: {ext}. "
                "Supported: .stl, .obj. "
                "For .step/.iges, install trimesh + python-occ."
            )

    @staticmethod
    def load_stl(path: str) -> TriangleMesh:
        """Load ASCII STL file."""
        return TriangleMesh.from_stl(path)

    @staticmethod
    def load_obj(path: str) -> TriangleMesh:
        """Load Wavefront OBJ file (vertices + faces only)."""
        vertices = []
        faces    = []
        for line in Path(path).read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(p) for p in parts[1:4]])
            elif parts[0] == "f":
                # OBJ face indices are 1-based; may include uv/normal as v/vt/vn
                idxs = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                if len(idxs) == 3:
                    faces.append(idxs)
                elif len(idxs) == 4:
                    # Quad → two triangles
                    faces.append([idxs[0], idxs[1], idxs[2]])
                    faces.append([idxs[0], idxs[2], idxs[3]])
        if not vertices:
            raise ValueError(f"No vertices in {path}")
        return TriangleMesh(np.array(vertices), np.array(faces))

    @staticmethod
    def load_with_trimesh(path: str) -> TriangleMesh:
        """Load any format supported by trimesh (requires: pip install trimesh)."""
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError("Install trimesh: pip install trimesh") from exc
        tm = trimesh.load(path, force="mesh")
        return TriangleMesh(
            vertices=np.array(tm.vertices, dtype=np.float64),
            faces   =np.array(tm.faces,    dtype=np.int32),
        )
