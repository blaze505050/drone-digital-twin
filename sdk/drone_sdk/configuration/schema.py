"""
drone_sdk.configuration.schema
==============================
Asset-Specific Vehicle Configuration & Component Bill-of-Materials (BOM).
Implements Backlog Item B09 & PRD CFG-01/02.

Provides:
- Typed component schemas (Airframe, Motor, Propeller, Battery, Avionics).
- Parallel Axis Theorem 3D inertia aggregation for multi-component assemblies.
- Asset validation (mass positivity, symmetric positive-definite inertia, rotor sanity).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ComponentBOM:
    """Individual hardware component with mass, CG offset, and local inertia tensor."""
    name: str
    mass_kg: float
    cg_offset_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia_tensor_kgm2: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    description: str = ""
    is_measured: bool = True

    def __post_init__(self) -> None:
        self.cg_offset_m = np.asarray(self.cg_offset_m, dtype=float)
        self.inertia_tensor_kgm2 = np.asarray(self.inertia_tensor_kgm2, dtype=float)
        if self.inertia_tensor_kgm2.shape != (3, 3):
            if self.inertia_tensor_kgm2.size == 3:
                self.inertia_tensor_kgm2 = np.diag(self.inertia_tensor_kgm2.flatten())
            elif self.inertia_tensor_kgm2.size == 0:
                self.inertia_tensor_kgm2 = np.zeros((3, 3))
            else:
                raise ValueError(f"Inertia tensor for {self.name} must be (3, 3), got {self.inertia_tensor_kgm2.shape}")


@dataclass
class MotorConfig:
    """Brushless DC motor electro-mechanical specifications."""
    name: str = "BLDC 2216"
    kv: float = 880.0                # RPM / Volt
    r_m: float = 0.12                # Internal winding resistance (Ohms)
    i_0: float = 0.6                 # No-load idle current (A)
    max_current_a: float = 25.0      # Maximum continuous current (A)
    tau_m: float = 0.035             # Motor + ESC first-order response time constant (s)
    mass_kg: float = 0.075           # Motor mass (kg)
    i_rotor_kgm2: float = 2.5e-5     # Rotor polar inertia (kg*m^2)


@dataclass
class PropellerConfig:
    """Propeller geometric and aerodynamic specifications."""
    name: str = "APC 10x4.5 MR"
    diameter_m: float = 0.254        # 10 inches = 0.254 m
    pitch_m: float = 0.1143          # 4.5 inches = 0.1143 m
    blade_count: int = 2
    mass_kg: float = 0.015           # Propeller mass (kg)
    i_prop_kgm2: float = 4.2e-5      # Propeller polar rotational inertia (kg*m^2)
    polar_dataset: str = "uiuc_apc_10x4.5"
    ct0: float = 0.110               # Static thrust coefficient at J=0
    cp0: float = 0.048               # Static power coefficient at J=0


@dataclass
class BatteryPackConfig:
    """LiPo / Li-ion battery pack configuration."""
    name: str = "4S 5000mAh 60C LiPo"
    cells_in_series: int = 4         # Ns (e.g. 4S)
    cells_in_parallel: int = 1       # Np (e.g. 1P)
    cell_capacity_ah: float = 5.0    # Ah per cell
    nominal_cell_voltage: float = 3.7# V
    max_cell_voltage: float = 4.2    # V fully charged
    min_cell_voltage: float = 3.2    # V cutoff
    cell_r0: float = 0.012           # Ohms series resistance per cell
    cell_r1: float = 0.008           # Ohms diffusion resistance per cell
    cell_c1: float = 1800.0          # Farads diffusion capacitance per cell
    pack_mass_kg: float = 0.490      # Total pack mass with wiring/casing (kg)

    @property
    def nominal_pack_voltage(self) -> float:
        return self.cells_in_series * self.nominal_cell_voltage

    @property
    def max_pack_voltage(self) -> float:
        return self.cells_in_series * self.max_cell_voltage

    @property
    def min_pack_voltage(self) -> float:
        return self.cells_in_series * self.min_cell_voltage

    @property
    def total_capacity_ah(self) -> float:
        return self.cells_in_parallel * self.cell_capacity_ah

    @property
    def nominal_energy_wh(self) -> float:
        return self.nominal_pack_voltage * self.total_capacity_ah


@dataclass
class AirframeConfig:
    """Airframe structural frame parameters."""
    name: str = "Holybro X500 V2 Frame"
    frame_type: str = "quad_x"       # quad_x, quad_plus, hex_x, octo_x
    wheelbase_m: float = 0.500       # 500 mm diagonal motor-to-motor
    mass_kg: float = 0.460           # Frame alone with arms/landing gear (kg)
    cg_offset_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia_tensor_kgm2: np.ndarray = field(
        default_factory=lambda: np.diag([0.0065, 0.0065, 0.0120])
    )


@dataclass
class AvionicsConfig:
    """Flight computer, sensors, GPS, telemetry, ESC payload."""
    name: str = "Pixhawk 6C + ESCs + GPS"
    mass_kg: float = 0.180
    cg_offset_m: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -0.02]))
    idle_power_w: float = 8.5        # Base avionics / flight controller power draw (Watts)
    inertia_tensor_kgm2: np.ndarray = field(
        default_factory=lambda: np.diag([0.0003, 0.0003, 0.0005])
    )


@dataclass
class RotorPlacement:
    """Position, orientation, and spin direction for a single rotor in FRD body frame."""
    rotor_id: int
    position_b: np.ndarray           # [x_forward, y_right, z_down] in metres
    axis_b: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0])) # Thrust vector in body frame
    spin_direction: int = 1          # +1 for CW (viewed from top), -1 for CCW
    motor_index: int = 0
    propeller_index: int = 0

    def __post_init__(self) -> None:
        self.position_b = np.asarray(self.position_b, dtype=float)
        self.axis_b = np.asarray(self.axis_b, dtype=float)
        norm = np.linalg.norm(self.axis_b)
        if norm > 1e-6:
            self.axis_b = self.axis_b / norm


@dataclass
class VehicleConfiguration:
    """
    Complete asset-specific vehicle configuration and assembly model.
    Encapsulates all physical parameters, BOM components, and geometry.
    """
    asset_id: str = "holybro_x500_v2"
    vehicle_name: str = "Holybro X500 V2 Reference Quadrotor"
    airframe: AirframeConfig = field(default_factory=AirframeConfig)
    battery: BatteryPackConfig = field(default_factory=BatteryPackConfig)
    avionics: AvionicsConfig = field(default_factory=AvionicsConfig)
    motors: List[MotorConfig] = field(default_factory=lambda: [MotorConfig()])
    propellers: List[PropellerConfig] = field(default_factory=lambda: [PropellerConfig()])
    rotors: List[RotorPlacement] = field(default_factory=list)
    components: List[ComponentBOM] = field(default_factory=list)

    def get_all_bom_components(self) -> List[ComponentBOM]:
        """Collect all BOM items including airframe, battery, avionics, motors, and props."""
        items: List[ComponentBOM] = []
        
        # 1. Airframe
        items.append(ComponentBOM(
            name=self.airframe.name,
            mass_kg=self.airframe.mass_kg,
            cg_offset_m=self.airframe.cg_offset_m,
            inertia_tensor_kgm2=self.airframe.inertia_tensor_kgm2,
            description="Airframe chassis and arms",
        ))
        
        # 2. Battery pack
        items.append(ComponentBOM(
            name=self.battery.name,
            mass_kg=self.battery.pack_mass_kg,
            cg_offset_m=np.array([0.0, 0.0, 0.01]),  # Battery typically slung underneath/centered
            inertia_tensor_kgm2=np.diag([0.0008, 0.0018, 0.0022]),
            description="LiPo Battery Pack",
        ))

        # 3. Avionics & FC
        items.append(ComponentBOM(
            name=self.avionics.name,
            mass_kg=self.avionics.mass_kg,
            cg_offset_m=self.avionics.cg_offset_m,
            inertia_tensor_kgm2=self.avionics.inertia_tensor_kgm2,
            description="Avionics & Flight Controller",
        ))

        # 4. Motors & Propellers per rotor station
        for r in self.rotors:
            m_cfg = self.motors[min(r.motor_index, len(self.motors) - 1)]
            p_cfg = self.propellers[min(r.propeller_index, len(self.propellers) - 1)]
            
            # Motor at rotor station
            items.append(ComponentBOM(
                name=f"Motor #{r.rotor_id} ({m_cfg.name})",
                mass_kg=m_cfg.mass_kg,
                cg_offset_m=r.position_b,
                inertia_tensor_kgm2=np.diag([1e-5, 1e-5, m_cfg.i_rotor_kgm2]),
                description=f"BLDC motor at rotor {r.rotor_id}",
            ))
            
            # Propeller at rotor station
            items.append(ComponentBOM(
                name=f"Propeller #{r.rotor_id} ({p_cfg.name})",
                mass_kg=p_cfg.mass_kg,
                cg_offset_m=r.position_b + np.array([0.0, 0.0, -0.015]),
                inertia_tensor_kgm2=np.diag([1e-5, 1e-5, p_cfg.i_prop_kgm2]),
                description=f"Propeller at rotor {r.rotor_id}",
            ))

        # 5. Additional custom components
        items.extend(self.components)
        return items

    def compute_total_mass(self) -> float:
        """Compute aggregate vehicle all-up mass (AUW) in kg."""
        return float(sum(item.mass_kg for item in self.get_all_bom_components()))

    def compute_center_of_gravity(self) -> np.ndarray:
        """
        Compute vehicle Center of Gravity (CG) in FRD body frame [x_cg, y_cg, z_cg] (m).
        r_CG = sum(m_i * r_i) / sum(m_i)
        """
        items = self.get_all_bom_components()
        total_m = sum(item.mass_kg for item in items)
        if total_m <= 1e-9:
            return np.zeros(3)
        cg = sum(item.mass_kg * item.cg_offset_m for item in items) / total_m
        return np.asarray(cg, dtype=float)

    def compute_total_inertia_tensor(self) -> np.ndarray:
        """
        Compute total 3x3 inertia tensor about assembly Center of Gravity (CG) in FRD frame.
        Uses the Parallel Axis Theorem (Steiner's Theorem):
        I_CG = sum( I_i + m_i * ( (d_i . d_i) * Eye(3) - d_i (x) d_i^T ) )
        where d_i = r_i - r_CG.
        """
        items = self.get_all_bom_components()
        r_cg = self.compute_center_of_gravity()
        i_total = np.zeros((3, 3), dtype=float)
        eye3 = np.eye(3, dtype=float)

        for item in items:
            m_i = item.mass_kg
            d_i = item.cg_offset_m - r_cg
            # Parallel axis term: m * ( (d . d) * I - d d^T )
            d_sq = float(np.dot(d_i, d_i))
            d_outer = np.outer(d_i, d_i)
            i_parallel = m_i * (d_sq * eye3 - d_outer)
            i_total += item.inertia_tensor_kgm2 + i_parallel

        # Enforce exact symmetry
        i_total = 0.5 * (i_total + i_total.T)
        return i_total

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """
        Validate physical consistency of vehicle configuration.
        Checks:
        1. Mass > 0.
        2. Positive-definite 3x3 inertia tensor.
        3. All eigenvalues > 0 (physical rigid body).
        4. Triangle inequality for moments of inertia (Ixx + Iyy >= Izz, etc.).
        5. At least 3 rotors with valid positions and unit thrust vectors.
        """
        issues: List[str] = []
        mass = self.compute_total_mass()
        if mass <= 0.01:
            issues.append(f"Total mass {mass:.3f} kg is invalid (must be > 0.01 kg).")

        I = self.compute_total_inertia_tensor()
        if not np.all(np.isfinite(I)):
            issues.append("Inertia tensor contains non-finite elements.")
        else:
            eigvals = np.linalg.eigvalsh(I)
            if np.any(eigvals <= 1e-7):
                issues.append(f"Inertia tensor is not strictly positive definite (eigenvalues: {eigvals}).")
            
            # Check triangle inequalities
            ixx, iyy, izz = I[0, 0], I[1, 1], I[2, 2]
            eps = 1e-6
            if (ixx + iyy < izz - eps) or (ixx + izz < iyy - eps) or (iyy + izz < ixx - eps):
                issues.append(f"Inertia moments violate triangle inequality: Ixx={ixx:.4e}, Iyy={iyy:.4e}, Izz={izz:.4e}")

        if len(self.rotors) < 3:
            issues.append(f"Vehicle has {len(self.rotors)} rotors, minimum 3 required for multirotor.")

        for r in self.rotors:
            norm = np.linalg.norm(r.axis_b)
            if abs(norm - 1.0) > 1e-3:
                issues.append(f"Rotor {r.rotor_id} thrust axis is not unit norm: {r.axis_b}")

        return (len(issues) == 0, issues)

    def to_dict(self) -> Dict[str, Any]:
        """Export full vehicle configuration metadata to JSON-serializable dict."""
        cg = self.compute_center_of_gravity()
        I = self.compute_total_inertia_tensor()
        is_valid, issues = self.validate_configuration()
        return {
            "asset_id": self.asset_id,
            "vehicle_name": self.vehicle_name,
            "total_mass_kg": self.compute_total_mass(),
            "center_of_gravity_frd_m": cg.tolist(),
            "inertia_tensor_kgm2": I.tolist(),
            "rotor_count": len(self.rotors),
            "is_valid": is_valid,
            "validation_issues": issues,
            "battery_wh": self.battery.nominal_energy_wh,
        }
