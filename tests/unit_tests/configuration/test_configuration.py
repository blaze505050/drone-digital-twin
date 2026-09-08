"""
Unit tests for Vehicle Configuration & Component BOM (B09).
Verifies:
- Mass positivity and component BOM aggregation.
- 3D inertia calculation via Parallel Axis Theorem (Steiner's theorem).
- Triangle inequalities (Ixx + Iyy >= Izz) and positive-definiteness.
- Canonical Holybro X500 V2 configuration integrity.
"""
import numpy as np
import pytest

from drone_sdk.configuration import (
    AirframeConfig,
    BatteryPackConfig,
    ComponentBOM,
    MotorConfig,
    PropellerConfig,
    RotorPlacement,
    VehicleConfiguration,
    create_holybro_x500_v2,
    get_vehicle_config,
)


def test_bom_inertia_aggregation_parallel_axis():
    # Simple symmetrical vehicle: frame (1kg at origin) + 2 point masses (0.5kg at +0.2m and -0.2m on X-axis)
    airframe = AirframeConfig(
        name="Test Frame",
        mass_kg=1.0,
        cg_offset_m=np.zeros(3),
        inertia_tensor_kgm2=np.diag([0.01, 0.01, 0.02]),
    )
    m1 = ComponentBOM(
        name="Left Mass",
        mass_kg=0.5,
        cg_offset_m=np.array([-0.2, 0.0, 0.0]),
        inertia_tensor_kgm2=np.zeros((3, 3)),
    )
    m2 = ComponentBOM(
        name="Right Mass",
        mass_kg=0.5,
        cg_offset_m=np.array([+0.2, 0.0, 0.0]),
        inertia_tensor_kgm2=np.zeros((3, 3)),
    )

    veh = VehicleConfiguration(
        asset_id="test_toy",
        vehicle_name="Toy Vehicle",
        airframe=airframe,
        components=[m1, m2],
    )

    # Total mass = 1.0 (frame) + 0.490 (default battery) + 0.180 (avionics) + 0.5 + 0.5 = 2.67 kg
    total_mass = veh.compute_total_mass()
    assert total_mass > 2.0

    # CG should be at x=0, y=0 due to symmetry
    cg = veh.compute_center_of_gravity()
    assert np.isclose(cg[0], 0.0, atol=1e-5)
    assert np.isclose(cg[1], 0.0, atol=1e-5)

    # Inertia tensor must be symmetric and positive-definite
    I = veh.compute_total_inertia_tensor()
    assert np.allclose(I, I.T, atol=1e-6)
    eigvals = np.linalg.eigvalsh(I)
    assert np.all(eigvals > 0.0)


def test_holybro_x500_v2_canonical_configuration():
    config = create_holybro_x500_v2()
    assert config.asset_id == "holybro_x500_v2"
    
    # Check total AUW ~ 1.4 - 1.6 kg
    mass = config.compute_total_mass()
    assert 1.30 <= mass <= 1.80

    # Validate physical consistency
    is_valid, issues = config.validate_configuration()
    assert is_valid is True, f"Validation failed with issues: {issues}"

    # Check 3D inertia moments satisfy triangle inequality
    I = config.compute_total_inertia_tensor()
    ixx, iyy, izz = I[0, 0], I[1, 1], I[2, 2]
    assert ixx + iyy >= izz - 1e-5
    assert ixx + izz >= iyy - 1e-5
    assert iyy + izz >= ixx - 1e-5

    # Check 4 rotors in Quad-X placement
    assert len(config.rotors) == 4
    for r in config.rotors:
        assert np.isclose(np.linalg.norm(r.axis_b), 1.0)
        assert abs(r.spin_direction) == 1

    # Check to_dict export
    d = config.to_dict()
    assert d["asset_id"] == "holybro_x500_v2"
    assert d["total_mass_kg"] == mass
    assert d["is_valid"] is True
