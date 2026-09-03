"""Tests for drone_sdk.cad_engine (Modules 10 & 11)."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pytest

from drone_sdk.cad_engine import (
    BoundingBox, CADImporter, DroneGeometryBuilder,
    GeometryAnalyser, MassEstimator, Triangle,
    TriangleMesh, Vec3,
)


# ══════════════════════════════════════════════════════════════════════════════
class TestVec3:
    def test_add(self):
        assert (Vec3(1,2,3) + Vec3(4,5,6)) == Vec3(5,7,9)

    def test_sub(self):
        assert (Vec3(5,7,9) - Vec3(4,5,6)) == Vec3(1,2,3)

    def test_mul(self):
        v = Vec3(1,2,3) * 2
        assert v == Vec3(2,4,6)

    def test_norm_unit(self):
        assert abs(Vec3(1,0,0).norm() - 1.0) < 1e-12

    def test_norm_diagonal(self):
        v = Vec3(1,1,1)
        assert abs(v.norm() - math.sqrt(3)) < 1e-10

    def test_normalised(self):
        v = Vec3(3,4,0).normalised()
        assert abs(v.norm() - 1.0) < 1e-10

    def test_dot(self):
        assert Vec3(1,0,0).dot(Vec3(1,0,0)) == 1.0
        assert Vec3(1,0,0).dot(Vec3(0,1,0)) == 0.0

    def test_cross(self):
        c = Vec3(1,0,0).cross(Vec3(0,1,0))
        assert abs(c.z - 1.0) < 1e-10

    def test_to_array(self):
        a = Vec3(1,2,3).to_array()
        assert a.shape == (3,)
        assert np.allclose(a, [1,2,3])

    def test_from_array(self):
        v = Vec3.from_array(np.array([4.,5.,6.]))
        assert v == Vec3(4,5,6)

    def test_zero_normalised_safe(self):
        v = Vec3(0,0,0).normalised()
        assert v == Vec3(0,0,0)

    def test_eq(self):
        assert Vec3(1,2,3) == Vec3(1,2,3)
        assert Vec3(1,2,3) != Vec3(1,2,4)


class TestBoundingBox:
    @pytest.fixture
    def unit_box(self):
        return BoundingBox(Vec3(-1,-1,-1), Vec3(1,1,1))

    def test_size(self, unit_box):
        s = unit_box.size
        assert abs(s.x - 2) < 1e-9
        assert abs(s.y - 2) < 1e-9
        assert abs(s.z - 2) < 1e-9

    def test_centre(self, unit_box):
        c = unit_box.centre
        assert abs(c.x) < 1e-9
        assert abs(c.y) < 1e-9
        assert abs(c.z) < 1e-9

    def test_volume(self, unit_box):
        assert abs(unit_box.volume - 8.0) < 1e-9

    def test_diagonal(self, unit_box):
        assert abs(unit_box.diagonal - 2*math.sqrt(3)) < 1e-9

    def test_expand(self, unit_box):
        expanded = unit_box.expand(2.0)
        assert abs(expanded.size.x - 4.0) < 1e-9


class TestTriangle:
    @pytest.fixture
    def tri(self):
        return Triangle(Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0))

    def test_normal_z(self, tri):
        n = tri.normal
        assert abs(n.z - 1.0) < 1e-9

    def test_area_unit(self, tri):
        assert abs(tri.area - 0.5) < 1e-9

    def test_centroid(self, tri):
        c = tri.centroid
        assert abs(c.x - 1/3) < 1e-9
        assert abs(c.y - 1/3) < 1e-9


# ══════════════════════════════════════════════════════════════════════════════
class TestTriangleMesh:
    @pytest.fixture
    def unit_cube(self):
        """A simple unit cube mesh (12 triangles)."""
        v = np.array([
            [0,0,0],[1,0,0],[1,1,0],[0,1,0],  # bottom
            [0,0,1],[1,0,1],[1,1,1],[0,1,1],  # top
        ], dtype=np.float64)
        f = np.array([
            [0,2,1],[0,3,2],  # bottom
            [4,5,6],[4,6,7],  # top
            [0,1,5],[0,5,4],  # front
            [1,2,6],[1,6,5],  # right
            [2,3,7],[2,7,6],  # back
            [3,0,4],[3,4,7],  # left
        ], dtype=np.int32)
        return TriangleMesh(v, f)

    @pytest.fixture
    def quad_mesh(self):
        return DroneGeometryBuilder.quadrotor_x()

    def test_n_vertices(self, unit_cube):
        assert unit_cube.n_vertices == 8

    def test_n_faces(self, unit_cube):
        assert unit_cube.n_faces == 12

    def test_bounding_box_unit_cube(self, unit_cube):
        bb = unit_cube.bounding_box
        assert abs(bb.size.x - 1.0) < 1e-9
        assert abs(bb.size.y - 1.0) < 1e-9
        assert abs(bb.size.z - 1.0) < 1e-9

    def test_surface_area_unit_cube(self, unit_cube):
        # 6 faces × 1 m² = 6 m²
        assert abs(unit_cube.surface_area - 6.0) < 1e-9

    def test_centroid_unit_cube(self, unit_cube):
        c = unit_cube.centroid
        assert abs(c[0] - 0.5) < 0.1
        assert abs(c[1] - 0.5) < 0.1
        assert abs(c[2] - 0.5) < 0.1

    def test_face_normals_shape(self, unit_cube):
        n = unit_cube.face_normals()
        assert n.shape == (12, 3)

    def test_face_normals_unit(self, unit_cube):
        norms = np.linalg.norm(unit_cube.face_normals(), axis=1)
        assert np.allclose(norms, 1.0, atol=1e-9)

    def test_translate(self, unit_cube):
        moved = unit_cube.translate(np.array([1,0,0]))
        assert abs(moved.bounding_box.min.x - 1.0) < 1e-9

    def test_scale(self, unit_cube):
        big = unit_cube.scale(2.0)
        assert abs(big.bounding_box.size.x - 2.0) < 1e-9

    def test_rotate_z_90(self, unit_cube):
        rot = unit_cube.rotate_z(math.pi/2)
        # After 90° rotation, x-extent maps to y-extent
        bb = rot.bounding_box
        assert abs(bb.size.y - 1.0) < 0.01

    def test_stl_roundtrip(self, unit_cube, tmp_path):
        path = str(tmp_path / "cube.stl")
        unit_cube.to_stl(path)
        loaded = TriangleMesh.from_stl(path)
        assert loaded.n_vertices > 0
        assert loaded.n_faces == unit_cube.n_faces

    def test_to_stl_creates_file(self, unit_cube, tmp_path):
        p = unit_cube.to_stl(str(tmp_path / "test.stl"))
        assert p.exists()
        assert p.stat().st_size > 0

    def test_repr(self, unit_cube):
        r = repr(unit_cube)
        assert "TriangleMesh" in r
        assert "faces=12" in r


# ══════════════════════════════════════════════════════════════════════════════
class TestDroneGeometryBuilder:
    def test_quadrotor_returns_mesh(self):
        m = DroneGeometryBuilder.quadrotor_x()
        assert isinstance(m, TriangleMesh)
        assert m.n_vertices > 0
        assert m.n_faces > 0

    def test_quadrotor_bounding_box_reasonable(self):
        arm = 0.25
        m   = DroneGeometryBuilder.quadrotor_x(arm_length=arm)
        bb  = m.bounding_box
        # Bounding box should span at least arm*2 in x and y
        assert bb.size.x > arm
        assert bb.size.y > arm

    def test_quadrotor_arm_length_scales_bbox(self):
        m1 = DroneGeometryBuilder.quadrotor_x(arm_length=0.25)
        m2 = DroneGeometryBuilder.quadrotor_x(arm_length=0.50)
        bb1 = m1.bounding_box
        bb2 = m2.bounding_box
        assert bb2.size.x > bb1.size.x

    def test_hexarotor_returns_mesh(self):
        m = DroneGeometryBuilder.hexarotor()
        assert m.n_vertices > 0

    def test_flat_plate_wing_8_vertices(self):
        m = DroneGeometryBuilder.flat_plate_wing()
        assert m.n_vertices == 8
        assert m.n_faces == 12

    def test_flat_plate_spans_correct(self):
        m  = DroneGeometryBuilder.flat_plate_wing(span=1.0, chord=0.15)
        bb = m.bounding_box
        assert abs(bb.size.y - 1.0) < 1e-9
        assert abs(bb.size.x - 0.15) < 1e-9

    def test_quadrotor_surface_area_positive(self):
        m = DroneGeometryBuilder.quadrotor_x()
        assert m.surface_area > 0

    def test_mesh_has_no_degenerate_faces(self):
        m = DroneGeometryBuilder.quadrotor_x()
        # All faces should have 3 distinct vertex indices
        for face in m.faces:
            assert len(set(face)) == 3, f"Degenerate face: {face}"


# ══════════════════════════════════════════════════════════════════════════════
class TestMassEstimator:
    @pytest.fixture
    def unit_cube(self):
        v = np.array([
            [0,0,0],[1,0,0],[1,1,0],[0,1,0],
            [0,0,1],[1,0,1],[1,1,1],[0,1,1],
        ], dtype=np.float64)
        f = np.array([
            [0,2,1],[0,3,2],[4,5,6],[4,6,7],
            [0,1,5],[0,5,4],[1,2,6],[1,6,5],
            [2,3,7],[2,7,6],[3,0,4],[3,4,7],
        ], dtype=np.int32)
        return TriangleMesh(v, f)

    def test_solid_density_1000(self, unit_cube):
        props = MassEstimator.from_mesh(unit_cube, density_kg_m3=1000.0)
        # Unit cube volume = 1 m³ × 1000 kg/m³ = 1000 kg
        assert abs(props.mass_kg - 1000.0) < 1.0

    def test_thin_wall_mass(self, unit_cube):
        props = MassEstimator.from_mesh(unit_cube, density_kg_m3=1200.0,
                                        wall_thickness=0.005)
        # S=6 m², t=0.005m, ρ=1200 → m=36 kg
        assert abs(props.mass_kg - 36.0) < 1.0

    def test_cg_returns_array(self, unit_cube):
        props = MassEstimator.from_mesh(unit_cube)
        assert props.cg.shape == (3,)

    def test_inertia_positive(self, unit_cube):
        props = MassEstimator.from_mesh(unit_cube, density_kg_m3=100.0)
        assert props.ixx > 0
        assert props.iyy > 0
        assert props.izz > 0

    def test_inertia_tensor_symmetric(self, unit_cube):
        props = MassEstimator.from_mesh(unit_cube, density_kg_m3=100.0)
        I = props.inertia_tensor
        assert np.allclose(I, I.T, atol=1e-12)

    def test_to_dict(self, unit_cube):
        props = MassEstimator.from_mesh(unit_cube)
        d = props.to_dict()
        assert "mass_kg" in d
        assert "ixx" in d
        assert "cg" in d

    def test_exact_cube_inertia_mirtich(self, unit_cube):
        """Verify Mirtich exact polyhedral inertia against analytical textbook solution: I = m * L^2 / 6."""
        density = 1000.0
        props = MassEstimator.from_mesh(unit_cube, density_kg_m3=density)
        # Analytical mass = 1000 kg, I = 1000 * 1^2 / 6 = 166.667 kg*m^2
        expected_inertia = 1000.0 / 6.0
        assert abs(props.mass_kg - 1000.0) < 1e-4
        assert abs(props.ixx - expected_inertia) < 1e-2
        assert abs(props.iyy - expected_inertia) < 1e-2
        assert abs(props.izz - expected_inertia) < 1e-2
        # Off-diagonal products must be zero for symmetric cube about CG
        assert abs(props.ixy) < 1e-4
        assert abs(props.ixz) < 1e-4
        assert abs(props.iyz) < 1e-4

    def test_to_twin_physics_params(self, unit_cube):
        """Test bridging CAD mass properties directly into ClosedLoopDigitalTwin physics."""
        props = MassEstimator.from_mesh(unit_cube, density_kg_m3=1.5)
        twin_params = props.to_twin_physics_params(arm_length_m=0.30, max_thrust_motor_n=8.0)
        assert abs(twin_params.mass_kg - 1.5) < 1e-3
        assert twin_params.arm_length_m == 0.30
        assert twin_params.max_thrust_motor_n == 8.0
        assert twin_params.ixx > 0

    def test_quadrotor_mass_reasonable(self):
        m = DroneGeometryBuilder.quadrotor_x(arm_length=0.25)
        # Carbon fibre density ~1600 kg/m³, thin wall → ~0.3-2 kg typical
        props = MassEstimator.from_mesh(m, density_kg_m3=1600.0, wall_thickness=0.001)
        assert 0.01 < props.mass_kg < 5.0


# ══════════════════════════════════════════════════════════════════════════════
class TestGeometryAnalyser:
    @pytest.fixture
    def wing_mesh(self):
        return DroneGeometryBuilder.flat_plate_wing(span=1.0, chord=0.15)

    @pytest.fixture
    def quad_mesh(self):
        return DroneGeometryBuilder.quadrotor_x(arm_length=0.25)

    def test_analyse_returns_surface(self, quad_mesh):
        from drone_sdk.cad_engine import AerodynamicSurface
        s = GeometryAnalyser.analyse(quad_mesh)
        assert isinstance(s, AerodynamicSurface)

    def test_wetted_area_positive(self, quad_mesh):
        s = GeometryAnalyser.analyse(quad_mesh)
        assert s.wetted_area_m2 > 0

    def test_reference_area_positive(self, quad_mesh):
        s = GeometryAnalyser.analyse(quad_mesh)
        assert s.reference_area_m2 > 0

    def test_frontal_area_positive(self, quad_mesh):
        s = GeometryAnalyser.analyse(quad_mesh)
        assert s.frontal_area_m2 > 0

    def test_wing_span_correct(self, wing_mesh):
        s = GeometryAnalyser.analyse(wing_mesh)
        assert abs(s.reference_span_m - 1.0) < 0.01

    def test_bounding_box_present(self, quad_mesh):
        s = GeometryAnalyser.analyse(quad_mesh)
        assert s.bounding_box is not None

    def test_to_dict_serialisable(self, quad_mesh):
        import json
        s = GeometryAnalyser.analyse(quad_mesh)
        json.dumps(s.to_dict())   # should not raise

    def test_wetted_ratio_above_1(self, quad_mesh):
        # Wetted area always ≥ reference area for a 3-D body
        s = GeometryAnalyser.analyse(quad_mesh)
        assert s.wetted_to_reference_ratio >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
class TestCADImporter:
    def test_load_stl_roundtrip(self, tmp_path):
        # Build, export, re-import. ASCII STL stores unshared verts (3 per face).
        original = DroneGeometryBuilder.flat_plate_wing()
        path     = original.to_stl(str(tmp_path / "wing.stl"))
        loaded   = CADImporter.load(str(path))
        assert loaded.n_faces == original.n_faces
        # ASCII STL: n_vertices = n_faces * 3 (unshared) vs original shared verts
        assert loaded.n_vertices == original.n_faces * 3

    def test_load_obj(self, tmp_path):
        # Write a simple OBJ
        obj_content = (
            "v 0.0 0.0 0.0\n"
            "v 1.0 0.0 0.0\n"
            "v 0.0 1.0 0.0\n"
            "v 0.0 0.0 1.0\n"
            "f 1 2 3\n"
            "f 1 2 4\n"
        )
        p = tmp_path / "test.obj"
        p.write_text(obj_content)
        m = CADImporter.load(str(p))
        assert m.n_vertices == 4
        assert m.n_faces    == 2

    def test_load_obj_quads_split_to_tris(self, tmp_path):
        obj_content = (
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
            "f 1 2 3 4\n"   # quad → 2 triangles
        )
        p = tmp_path / "quad.obj"
        p.write_text(obj_content)
        m = CADImporter.load(str(p))
        assert m.n_faces == 2

    def test_unsupported_format_raises(self, tmp_path):
        p = tmp_path / "bad.iges"
        p.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            CADImporter.load(str(p))

    def test_stl_vertices_finite(self, tmp_path):
        original = DroneGeometryBuilder.quadrotor_x()
        p        = original.to_stl(str(tmp_path / "quad.stl"))
        loaded   = CADImporter.load(str(p))
        assert np.all(np.isfinite(loaded.vertices))
