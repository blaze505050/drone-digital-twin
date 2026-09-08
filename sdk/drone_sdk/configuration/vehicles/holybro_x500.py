"""
drone_sdk.configuration.vehicles.holybro_x500
=============================================
Pre-configured Holybro X500 V2 Reference Quadrotor configuration.
Conforms to PX4 Quad-X standard motor numbering and FRD body frame coordinates.
"""
from __future__ import annotations

import math
import numpy as np

from ..schema import (
    AirframeConfig,
    AvionicsConfig,
    BatteryPackConfig,
    ComponentBOM,
    MotorConfig,
    PropellerConfig,
    RotorPlacement,
    VehicleConfiguration,
)


def create_holybro_x500_v2() -> VehicleConfiguration:
    """
    Constructs the canonical Holybro X500 V2 reference asset.
    - Diagonal wheelbase: 500 mm (radius = 0.250 m)
    - Rotor angles: 45°, 135°, 225°, 315° relative to FRD +X nose
    - PX4 Standard Quad-X Motor Ordering:
      * Motor 1: Front-Right (+X, +Y), CCW (-1)
      * Motor 2: Rear-Left   (-X, -Y), CCW (-1)
      * Motor 3: Front-Left  (+X, -Y), CW  (+1)
      * Motor 4: Rear-Right  (-X, +Y), CW  (+1)
    """
    arm_radius = 0.250  # 250 mm
    cos45 = math.cos(math.radians(45))
    d_xy = arm_radius * cos45  # ~0.1768 m

    # 1. Airframe
    airframe = AirframeConfig(
        name="Holybro X500 V2 Carbon Fiber Frame",
        frame_type="quad_x",
        wheelbase_m=0.500,
        mass_kg=0.450,
        cg_offset_m=np.array([0.0, 0.0, 0.0]),
        inertia_tensor_kgm2=np.diag([0.0075, 0.0075, 0.0135]),
    )

    # 2. Battery (4S 5000mAh 60C)
    battery = BatteryPackConfig(
        name="Tattu 4S 5000mAh 60C LiPo",
        cells_in_series=4,
        cells_in_parallel=1,
        cell_capacity_ah=5.0,
        nominal_cell_voltage=3.7,
        max_cell_voltage=4.2,
        min_cell_voltage=3.2,
        cell_r0=0.012,
        cell_r1=0.008,
        cell_c1=1800.0,
        pack_mass_kg=0.490,
    )

    # 3. Avionics (Pixhawk 6C + PM02 Power Module + Telemetry + GPS)
    avionics = AvionicsConfig(
        name="Holybro Pixhawk 6C + GPS + Telemetry",
        mass_kg=0.180,
        cg_offset_m=np.array([0.0, 0.0, -0.015]),
        idle_power_w=8.5,
        inertia_tensor_kgm2=np.diag([0.0003, 0.0003, 0.0005]),
    )

    # 4. Motors (Holybro 2216 880kV)
    motors = [
        MotorConfig(
            name="Holybro 2216 880kV",
            kv=880.0,
            r_m=0.120,
            i_0=0.60,
            max_current_a=25.0,
            tau_m=0.032,
            mass_kg=0.075,
            i_rotor_kgm2=2.5e-5,
        )
    ]

    # 5. Propellers (APC 10x4.5 MR)
    propellers = [
        PropellerConfig(
            name="APC 10x4.5 MR",
            diameter_m=0.254,
            pitch_m=0.1143,
            blade_count=2,
            mass_kg=0.015,
            i_prop_kgm2=4.2e-5,
            polar_dataset="uiuc_apc_10x4.5",
            ct0=0.112,
            cp0=0.048,
        )
    ]

    # 6. Rotors placement (FRD Body Frame: +X forward, +Y right, +Z down)
    rotors = [
        # Rotor 1: Front-Right (+X, +Y), CCW (-1)
        RotorPlacement(
            rotor_id=1,
            position_b=np.array([+d_xy, +d_xy, -0.01]),
            axis_b=np.array([0.0, 0.0, -1.0]),
            spin_direction=-1,
            motor_index=0,
            propeller_index=0,
        ),
        # Rotor 2: Rear-Left (-X, -Y), CCW (-1)
        RotorPlacement(
            rotor_id=2,
            position_b=np.array([-d_xy, -d_xy, -0.01]),
            axis_b=np.array([0.0, 0.0, -1.0]),
            spin_direction=-1,
            motor_index=0,
            propeller_index=0,
        ),
        # Rotor 3: Front-Left (+X, -Y), CW (+1)
        RotorPlacement(
            rotor_id=3,
            position_b=np.array([+d_xy, -d_xy, -0.01]),
            axis_b=np.array([0.0, 0.0, -1.0]),
            spin_direction=+1,
            motor_index=0,
            propeller_index=0,
        ),
        # Rotor 4: Rear-Right (-X, +Y), CW (+1)
        RotorPlacement(
            rotor_id=4,
            position_b=np.array([-d_xy, +d_xy, -0.01]),
            axis_b=np.array([0.0, 0.0, -1.0]),
            spin_direction=+1,
            motor_index=0,
            propeller_index=0,
        ),
    ]

    # 7. Landing gear / wiring BOM additions
    components = [
        ComponentBOM(
            name="Carbon Fiber Landing Gear Legs",
            mass_kg=0.090,
            cg_offset_m=np.array([0.0, 0.0, 0.08]),  # Extending downward (+Z in FRD)
            inertia_tensor_kgm2=np.diag([0.0012, 0.0012, 0.0006]),
            description="High-strength carbon landing skids",
        ),
        ComponentBOM(
            name="Wiring Harness & XT60 Connector",
            mass_kg=0.045,
            cg_offset_m=np.array([0.0, 0.0, 0.01]),
            inertia_tensor_kgm2=np.diag([1e-5, 1e-5, 2e-5]),
            description="Power distribution wiring",
        ),
    ]

    return VehicleConfiguration(
        asset_id="holybro_x500_v2",
        vehicle_name="Holybro X500 V2 Reference Quadrotor",
        airframe=airframe,
        battery=battery,
        avionics=avionics,
        motors=motors,
        propellers=propellers,
        rotors=rotors,
        components=components,
    )
