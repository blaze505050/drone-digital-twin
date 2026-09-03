"""Tests for drone_sdk.openfoam_bridge (Module 12)."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pytest

from drone_sdk.openfoam_bridge import (
    AeroDatabase, AeroForces, FlowCondition,
    FoamDictWriter, OpenFOAMCase,
)


# ══════════════════════════════════════════════════════════════════════════════
class TestFlowCondition:
    @pytest.fixture
    def fc(self):
        return FlowCondition(velocity_ms=10.0, aoa_deg=0.0, altitude_m=100.0)

    def test_density_sea_level(self):
        fc = FlowCondition(altitude_m=0.0)
        assert abs(fc.density - 1.225) < 0.01

    def test_density_decreases_with_altitude(self):
        lo = FlowCondition(altitude_m=0.0)
        hi = FlowCondition(altitude_m=5000.0)
        assert hi.density < lo.density

    def test_nu_positive(self, fc):
        assert fc.kinematic_viscosity > 0

    def test_nu_increases_with_altitude(self):
        lo = FlowCondition(altitude_m=0.0)
        hi = FlowCondition(altitude_m=5000.0)
        assert hi.kinematic_viscosity > lo.kinematic_viscosity

    def test_reynolds_scales_with_velocity(self):
        fc1 = FlowCondition(velocity_ms=10.0)
        fc2 = FlowCondition(velocity_ms=20.0)
        assert abs(fc2.reynolds_number(0.25) / fc1.reynolds_number(0.25) - 2.0) < 0.01

    def test_velocity_vector_zero_aoa(self, fc):
        v = fc.velocity_vector()
        assert v.shape == (3,)
        assert abs(np.linalg.norm(v) - 10.0) < 0.01
        assert abs(v[0] - 10.0) < 0.01
        assert abs(v[1]) < 0.001
        assert abs(v[2]) < 0.001

    def test_velocity_vector_nonzero_aoa(self):
        fc = FlowCondition(velocity_ms=10.0, aoa_deg=10.0)
        v  = fc.velocity_vector()
        assert abs(np.linalg.norm(v) - 10.0) < 0.01
        assert v[2] > 0.0      # positive z component for positive AoA

    def test_velocity_vector_magnitude_preserved(self):
        for aoa in range(-20, 21, 5):
            fc = FlowCondition(velocity_ms=15.0, aoa_deg=float(aoa))
            assert abs(np.linalg.norm(fc.velocity_vector()) - 15.0) < 0.01

    def test_to_dict(self, fc):
        d = fc.to_dict()
        assert "velocity_ms"  in d
        assert "aoa_deg"      in d
        assert "density"      in d
        assert "nu"           in d

    def test_turbulence_k_positive(self):
        fc = FlowCondition(turbulence_intensity=0.05)
        k  = 1.5 * (fc.velocity_ms * fc.turbulence_intensity) ** 2
        assert k > 0


# ══════════════════════════════════════════════════════════════════════════════
class TestFoamDictWriter:
    @pytest.fixture
    def writer(self):
        return FoamDictWriter()

    @pytest.fixture
    def case_dir(self, tmp_path):
        return tmp_path / "test_case"

    def test_foam_header_contains_class(self, writer):
        h = writer.foam_header("dictionary", "system", "controlDict")
        assert "dictionary" in h
        assert "controlDict" in h
        assert "FoamFile" in h

    def test_write_control_dict(self, writer, case_dir):
        p = writer.write_control_dict(case_dir, end_time=300)
        assert p.exists()
        content = p.read_text()
        assert "simpleFoam" in content
        assert "300" in content
        assert "forces" in content

    def test_write_fv_solution(self, writer, case_dir):
        p = writer.write_fv_solution(case_dir)
        assert p.exists()
        content = p.read_text()
        assert "SIMPLE" in content
        assert "GAMG" in content

    def test_write_fv_schemes(self, writer, case_dir):
        p = writer.write_fv_schemes(case_dir)
        assert p.exists()
        content = p.read_text()
        assert "steadyState" in content
        assert "Gauss" in content

    def test_write_block_mesh_dict(self, writer, case_dir):
        p = writer.write_block_mesh_dict(case_dir, domain=(20.0, 10.0, 10.0))
        assert p.exists()
        content = p.read_text()
        assert "vertices" in content
        assert "blocks"   in content
        assert "boundary" in content
        assert "inlet"    in content
        assert "outlet"   in content

    def test_block_mesh_domain_size_in_file(self, writer, case_dir):
        writer.write_block_mesh_dict(case_dir, domain=(30.0, 15.0, 15.0))
        content = (case_dir / "system" / "blockMeshDict").read_text()
        assert "15.0" in content or "15.00" in content

    def test_write_snappy_hex_mesh_dict(self, writer, case_dir):
        p = writer.write_snappy_hex_mesh_dict(
            case_dir, stl_filename="drone.stl", n_refinement=4
        )
        assert p.exists()
        content = p.read_text()
        assert "drone.stl" in content
        assert "snappyHexMeshDict" in content

    def test_write_initial_conditions_creates_files(self, writer, case_dir):
        flow  = FlowCondition(velocity_ms=10.0, aoa_deg=5.0)
        paths = writer.write_initial_conditions(case_dir, flow)
        assert "U"     in paths
        assert "p"     in paths
        assert "k"     in paths
        assert "omega" in paths
        assert "nut"   in paths
        for p in paths.values():
            assert p.exists()

    def test_U_file_contains_velocity(self, writer, case_dir):
        flow = FlowCondition(velocity_ms=15.0, aoa_deg=0.0)
        paths = writer.write_initial_conditions(case_dir, flow)
        content = paths["U"].read_text()
        assert "15.0" in content or "15." in content
        assert "inlet" in content

    def test_k_file_contains_turbulence(self, writer, case_dir):
        flow    = FlowCondition(velocity_ms=10.0, turbulence_intensity=0.05)
        paths   = writer.write_initial_conditions(case_dir, flow)
        content = paths["k"].read_text()
        k_expected = 1.5 * (10.0 * 0.05) ** 2   # = 0.1875
        assert str(round(k_expected, 2))[:3] in content

    def test_write_transport_properties(self, writer, case_dir):
        fc = FlowCondition(velocity_ms=10.0, altitude_m=0.0)
        p  = writer.write_transport_properties(case_dir, fc.kinematic_viscosity)
        assert p.exists()
        content = p.read_text()
        assert "Newtonian" in content
        assert "nu" in content

    def test_write_turbulence_properties(self, writer, case_dir):
        p = writer.write_turbulence_properties(case_dir)
        assert p.exists()
        content = p.read_text()
        assert "kOmegaSST" in content
        assert "RAS" in content


# ══════════════════════════════════════════════════════════════════════════════
class TestOpenFOAMCase:
    @pytest.fixture
    def flow(self):
        return FlowCondition(velocity_ms=10.0, aoa_deg=0.0)

    @pytest.fixture
    def case(self, tmp_path, flow):
        return OpenFOAMCase(str(tmp_path / "foam_case"), flow)

    def test_setup_creates_directory(self, case):
        case.setup()
        assert case.case_dir.exists()

    def test_setup_creates_system_dir(self, case):
        case.setup()
        assert (case.case_dir / "system").is_dir()

    def test_setup_creates_constant_dir(self, case):
        case.setup()
        assert (case.case_dir / "constant").is_dir()

    def test_setup_creates_0_dir(self, case):
        case.setup()
        assert (case.case_dir / "0").is_dir()

    def test_setup_writes_all_system_files(self, case):
        case.setup()
        system = case.case_dir / "system"
        assert (system / "controlDict").exists()
        assert (system / "fvSolution").exists()
        assert (system / "fvSchemes").exists()
        assert (system / "blockMeshDict").exists()

    def test_setup_writes_constant_files(self, case):
        case.setup()
        const = case.case_dir / "constant"
        assert (const / "transportProperties").exists()
        assert (const / "turbulenceProperties").exists()

    def test_setup_writes_initial_conditions(self, case):
        case.setup()
        zero = case.case_dir / "0"
        for fname in ["U", "p", "k", "omega", "nut"]:
            assert (zero / fname).exists()

    def test_setup_copies_stl(self, tmp_path, flow):
        # Create a dummy STL file
        stl_path = tmp_path / "airframe.stl"
        stl_path.write_text("solid test\nendsolid test\n")
        case = OpenFOAMCase(str(tmp_path / "case_stl"), flow)
        case.setup(str(stl_path))
        dest = case.case_dir / "constant" / "triSurface" / "airframe.stl"
        assert dest.exists()

    def test_setup_writes_snappy_when_stl_given(self, tmp_path, flow):
        stl_path = tmp_path / "drone.stl"
        stl_path.write_text("solid drone\nendsolid drone\n")
        case = OpenFOAMCase(str(tmp_path / "case_snappy"), flow)
        case.setup(str(stl_path))
        assert (case.case_dir / "system" / "snappyHexMeshDict").exists()

    def test_run_raises_without_openfoam(self, case):
        case.setup()
        from unittest.mock import patch
        with patch.object(OpenFOAMCase, "_openfoam_available", return_value=False):
            with pytest.raises(RuntimeError, match="OpenFOAM"):
                case.run()

    def test_parse_missing_postprocessing_returns_unconverged(self, case):
        case.setup()
        result = case._parse_results()
        assert result.converged is False
        assert result.aoa_deg == 0.0

    def test_parse_coeff_file(self, case, tmp_path):
        # Write a mock forceCoeffs.dat
        pp_dir = case.case_dir / "postProcessing" / "forceCoeffs" / "500"
        pp_dir.mkdir(parents=True)
        coeff_file = pp_dir / "forceCoeffs.dat"
        coeff_file.write_text(
            "# Time  Cm  Cd  Cl\n"
            "499 -0.05 0.12 0.45\n"
            "500 -0.04 0.11 0.47\n"
        )
        result = case._parse_coeff_file(coeff_file)
        assert abs(result.CL - 0.47) < 1e-6
        assert abs(result.CD - 0.11) < 1e-6
        assert result.converged is True


# ══════════════════════════════════════════════════════════════════════════════
class TestAeroForces:
    def test_l_d_computed(self):
        f = AeroForces(aoa_deg=5.0, velocity_ms=10.0, CL=0.5, CD=0.05)
        assert abs(f.L_D - 10.0) < 1e-9

    def test_zero_cd_l_d_zero(self):
        f = AeroForces(aoa_deg=0.0, velocity_ms=10.0, CL=0.3, CD=0.0)
        assert f.L_D == 0.0

    def test_to_dict(self):
        f = AeroForces(aoa_deg=5.0, velocity_ms=10.0, CL=0.5, CD=0.05)
        d = f.to_dict()
        assert "CL" in d
        assert "CD" in d
        assert "L_D" in d
        assert "converged" in d

    def test_not_converged_by_default(self):
        f = AeroForces(aoa_deg=0.0, velocity_ms=5.0)
        assert f.converged is False

    def test_json_serialisable(self):
        f = AeroForces(5.0, 10.0, CL=0.4, CD=0.04, converged=True)
        json.dumps(f.to_dict())   # must not raise


# ══════════════════════════════════════════════════════════════════════════════
class TestAeroDatabase:
    @pytest.fixture
    def db_with_results(self, tmp_path):
        db = AeroDatabase(str(tmp_path / "db"))
        db._results = [
            AeroForces(aoa_deg=a, velocity_ms=10.0,
                       CL=0.1*a, CD=0.02 + 0.001*a**2,
                       converged=True)
            for a in range(-5, 16, 5)
        ]
        return db

    def test_len(self, db_with_results):
        assert len(db_with_results) == 5

    def test_to_numpy_shape(self, db_with_results):
        arr = db_with_results.to_numpy()
        assert arr.shape == (5, 7)

    def test_to_numpy_columns(self, db_with_results):
        arr = db_with_results.to_numpy()
        # Column 0 = velocity, all should be 10.0
        assert np.allclose(arr[:, 0], 10.0)

    def test_save_and_load(self, db_with_results, tmp_path):
        path = db_with_results.save(str(tmp_path / "aero.json"))
        assert path.exists()
        db2 = AeroDatabase.load(str(path))
        assert len(db2) == len(db_with_results)

    def test_load_preserves_coefficients(self, db_with_results, tmp_path):
        path = db_with_results.save(str(tmp_path / "aero2.json"))
        db2  = AeroDatabase.load(str(path))
        for r1, r2 in zip(db_with_results._results, db2._results):
            assert abs(r1.CL - r2.CL) < 1e-9
            assert abs(r1.CD - r2.CD) < 1e-9

    def test_get_polar(self, db_with_results):
        aoas, coef = db_with_results.get_polar(10.0)
        assert len(aoas) == 5
        assert coef.shape == (5, 2)

    def test_get_polar_sorted(self, db_with_results):
        aoas, _ = db_with_results.get_polar(10.0)
        assert np.all(np.diff(aoas) > 0)

    def test_get_polar_empty_for_missing_velocity(self, db_with_results):
        aoas, coef = db_with_results.get_polar(99.0)
        assert len(aoas) == 0

    def test_repr(self, db_with_results):
        r = repr(db_with_results)
        assert "AeroDatabase" in r
        assert "5 cases" in r

    def test_run_sweep_no_openfoam(self, tmp_path):
        """run_sweep should gracefully handle missing OpenFOAM."""
        from unittest.mock import patch
        with patch.object(OpenFOAMCase, "_openfoam_available", return_value=False):
            db = AeroDatabase(str(tmp_path / "sweep"))
            results = db.run_sweep(
                velocities=[10.0],
                aoa_range=(0.0, 10.0, 5.0),
                n_iterations=50,
            )
        # Should return placeholder (not converged) results without crashing
        assert len(results) == 2   # 0° and 5°
        assert all(not r.converged for r in results)

    def test_empty_database_numpy(self, tmp_path):
        db  = AeroDatabase(str(tmp_path))
        arr = db.to_numpy()
        assert arr.shape == (0, 7)
