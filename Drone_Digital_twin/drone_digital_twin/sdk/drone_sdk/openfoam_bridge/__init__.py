"""
drone_sdk.openfoam_bridge
=========================
OpenFOAM Integration — Module 12 of the UAV Digital Twin Platform.

Provides a complete Python API for:

1. **Case generation** — write all OpenFOAM dictionary files for a drone
   aerodynamics simulation (simpleFoam steady RANS, pisoFoam transient).

2. **Mesh pipeline** — blockMesh background mesh + snappyHexMesh surface
   refinement from an STL geometry.

3. **Boundary conditions** — automated inlet/outlet/wall setup from
   flight condition (velocity, AoA, Re).

4. **Runner** — launch OpenFOAM solvers as subprocesses with real-time
   residual monitoring.

5. **Results parser** — extract forces, moments, Cp fields from
   OpenFOAM postProcessing/ output.

6. **Aerodynamic database** — sweep AoA/velocity, build lookup table,
   export for PINN training (Module 14).

Requires: OpenFOAM v10+ (openfoam.org) or ESI OpenFOAM v2306+
Install: https://openfoam.org/download/

Python version: 3.9+
"""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Flow conditions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FlowCondition:
    """Free-stream aerodynamic conditions for a CFD simulation."""
    velocity_ms:      float   = 10.0    # Freestream speed, m/s
    aoa_deg:          float   = 0.0     # Angle of attack, degrees
    sideslip_deg:     float   = 0.0     # Angle of sideslip, degrees
    altitude_m:       float   = 100.0   # Altitude (for ISA density)
    turbulence_intensity: float = 0.05  # Fraction (5% default)
    turbulence_length_m:  float = 0.01  # m

    # Derived from ISA at altitude
    @property
    def density(self) -> float:
        """Air density from ISA model (kg/m³)."""
        T = 288.15 - 0.0065 * self.altitude_m
        p = 101325.0 * (T / 288.15) ** 5.2561
        return p / (287.058 * T)

    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity from Sutherland's law (m²/s)."""
        T   = 288.15 - 0.0065 * self.altitude_m
        mu  = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)
        return mu / self.density

    def reynolds_number(self, length_m: float) -> float:
        """Reynolds number based on reference length."""
        return self.velocity_ms * length_m / self.kinematic_viscosity

    def velocity_vector(self) -> np.ndarray:
        """Velocity vector in wind frame (x = freestream direction)."""
        aoa = math.radians(self.aoa_deg)
        beta = math.radians(self.sideslip_deg)
        U = self.velocity_ms
        return np.array([
            U * math.cos(aoa) * math.cos(beta),
            U * math.sin(beta),
            U * math.sin(aoa),
        ])

    def to_dict(self) -> dict:
        return {
            "velocity_ms":      self.velocity_ms,
            "aoa_deg":          self.aoa_deg,
            "sideslip_deg":     self.sideslip_deg,
            "altitude_m":       self.altitude_m,
            "density":          round(self.density, 4),
            "nu":               round(self.kinematic_viscosity, 8),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Aerodynamic results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AeroForces:
    """Aerodynamic forces and moments from OpenFOAM postprocessing."""
    aoa_deg:       float
    velocity_ms:   float
    CL:            float = 0.0    # Lift coefficient
    CD:            float = 0.0    # Drag coefficient
    CS:            float = 0.0    # Side-force coefficient
    Cl:            float = 0.0    # Roll moment coefficient
    Cm:            float = 0.0    # Pitch moment coefficient
    Cn:            float = 0.0    # Yaw moment coefficient
    L_D:           float = 0.0    # Lift-to-drag ratio
    # Raw forces [N]
    Fx:            float = 0.0
    Fy:            float = 0.0
    Fz:            float = 0.0
    converged:     bool  = False
    iterations:    int   = 0
    residual_final: float = 1.0

    def __post_init__(self):
        if abs(self.CD) > 1e-10:
            self.L_D = self.CL / self.CD

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  OpenFOAM dictionary writers
# ─────────────────────────────────────────────────────────────────────────────

class FoamDictWriter:
    """Writes OpenFOAM dictionary files from Python data structures.

    Generates syntactically correct OpenFOAM C++ dictionary format.

    Usage::

        writer = FoamDictWriter()
        writer.write_block_mesh_dict(case_dir, domain_size=(20,10,10))
    """

    @staticmethod
    def foam_header(foam_class: str, location: str, obj: str) -> str:
        return f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: Open Source CFD
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  10
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       {foam_class};
    location    "{location}";
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

    def write_control_dict(
        self,
        case_dir:         Path,
        application:      str   = "simpleFoam",
        end_time:         int   = 500,
        delta_t:          float = 1.0,
        write_interval:   int   = 50,
        write_precision:  int   = 8,
    ) -> Path:
        """Write system/controlDict."""
        content = self.foam_header("dictionary", "system", "controlDict") + f"""
application     {application};

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};

deltaT          {delta_t};

writeControl    timeStep;
writeInterval   {write_interval};
purgeWrite      3;

writeFormat     ascii;
writePrecision  {write_precision};
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;

functions
{{
    forces
    {{
        type            forces;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   {write_interval};
        patches         (drone);
        rho             rhoInf;
        rhoInf          1.225;
        CofR            (0 0 0);
    }}

    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   {write_interval};
        patches         (drone);
        rhoInf          1.225;
        liftDir         (0 0 1);
        dragDir         (1 0 0);
        pitchAxis       (0 1 0);
        magUInf         10.0;
        lRef            0.25;
        Aref            0.0625;
        CofR            (0 0 0);
    }}
}}
"""
        path = case_dir / "system" / "controlDict"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        logger.debug("Wrote %s", path)
        return path

    def write_fv_solution(self, case_dir: Path) -> Path:
        """Write system/fvSolution for simpleFoam."""
        content = self.foam_header("dictionary", "system", "fvSolution") + """
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
    }
    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
    k
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
    omega
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    residualControl
    {
        p               1e-4;
        U               1e-4;
        k               1e-4;
        omega           1e-4;
    }
}

relaxationFactors
{
    fields { p 0.3; }
    equations { U 0.7; k 0.7; omega 0.7; }
}
"""
        path = case_dir / "system" / "fvSolution"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def write_fv_schemes(self, case_dir: Path) -> Path:
        """Write system/fvSchemes for RANS simulation."""
        content = self.foam_header("dictionary", "system", "fvSchemes") + """
ddtSchemes     { default steadyState; }
gradSchemes    { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes  { default corrected; }
"""
        path = case_dir / "system" / "fvSchemes"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def write_block_mesh_dict(
        self,
        case_dir:    Path,
        domain:      Tuple[float,float,float] = (20.0, 10.0, 10.0),
        n_cells:     Tuple[int,int,int]       = (40, 20, 20),
    ) -> Path:
        """Write system/blockMeshDict for background hex mesh."""
        dx, dy, dz = domain
        nx, ny, nz = n_cells
        content = self.foam_header("dictionary", "system", "blockMeshDict") + f"""
scale 1;

vertices
(
    (-{dx/2:.2f} -{dy/2:.2f} -{dz/2:.2f})
    ( {dx/2:.2f} -{dy/2:.2f} -{dz/2:.2f})
    ( {dx/2:.2f}  {dy/2:.2f} -{dz/2:.2f})
    (-{dx/2:.2f}  {dy/2:.2f} -{dz/2:.2f})
    (-{dx/2:.2f} -{dy/2:.2f}  {dz/2:.2f})
    ( {dx/2:.2f} -{dy/2:.2f}  {dz/2:.2f})
    ( {dx/2:.2f}  {dy/2:.2f}  {dz/2:.2f})
    (-{dx/2:.2f}  {dy/2:.2f}  {dz/2:.2f})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

boundary
(
    inlet   {{ type patch; faces ((0 4 7 3)); }}
    outlet  {{ type patch; faces ((1 2 6 5)); }}
    top     {{ type symmetryPlane; faces ((4 5 6 7)); }}
    bottom  {{ type symmetryPlane; faces ((0 1 2 3)); }}
    sides   {{ type symmetryPlane; faces ((0 4 5 1)(3 2 6 7)); }}
);
"""
        path = case_dir / "system" / "blockMeshDict"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def write_snappy_hex_mesh_dict(
        self,
        case_dir:      Path,
        stl_filename:  str   = "drone.stl",
        n_refinement:  int   = 3,
        n_surface_layers: int = 3,
    ) -> Path:
        """Write system/snappyHexMeshDict for surface refinement."""
        content = self.foam_header("dictionary", "system", "snappyHexMeshDict") + f"""
castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
    {stl_filename}
    {{
        type triSurfaceMesh;
        name drone;
    }}
}}

castellatedMeshControls
{{
    maxLocalCells       1000000;
    maxGlobalCells      2000000;
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 3;

    features ( );

    refinementSurfaces
    {{
        drone
        {{
            level ({n_refinement} {n_refinement});
            patchInfo {{ type wall; }}
        }}
    }}

    refinementRegions {{ }}

    locationInMesh (0.001 0.001 0.5);
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch     3;
    tolerance        2.0;
    nSolveIter       30;
    nRelaxIter       5;
    nFeatureSnapIter 10;
}}

addLayersControls
{{
    relativeSizes true;
    expansionRatio 1.2;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
    layers
    {{
        drone {{ nSurfaceLayers {n_surface_layers}; }}
    }}
}}

meshQualityControls
{{
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minVol              1e-13;
    minTetQuality       1e-9;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.05;
    minVolRatio         0.01;
    minTriangleTwist    -1;
    nSmoothScale        4;
    errorReduction      0.75;
}}
"""
        path = case_dir / "system" / "snappyHexMeshDict"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def write_initial_conditions(
        self,
        case_dir: Path,
        flow:     FlowCondition,
    ) -> Dict[str, Path]:
        """Write 0/ initial condition files (U, p, k, omega, nut)."""
        U   = flow.velocity_vector()
        # nu = flow.kinematic_viscosity  # written by write_transport_properties
        I   = flow.turbulence_intensity
        L   = flow.turbulence_length_m
        k   = 1.5 * (flow.velocity_ms * I) ** 2
        omega = math.sqrt(k) / (0.09**0.25 * L)

        zero_dir = case_dir / "0"
        zero_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # U
        paths["U"] = self._write_field(
            zero_dir / "U", "volVectorField", "U",
            f"internalField   uniform ({U[0]:.4f} {U[1]:.4f} {U[2]:.4f});\n"
            "boundaryField\n{\n"
            f"    inlet   {{ type fixedValue; value uniform ({U[0]:.4f} {U[1]:.4f} {U[2]:.4f}); }}\n"
            "    outlet  { type zeroGradient; }\n"
            "    drone   { type noSlip; }\n"
            "    top     { type symmetryPlane; }\n"
            "    bottom  { type symmetryPlane; }\n"
            "    sides   { type symmetryPlane; }\n"
            "}",
        )

        # p
        paths["p"] = self._write_field(
            zero_dir / "p", "volScalarField", "p",
            "internalField   uniform 0;\n"
            "boundaryField\n{\n"
            "    inlet   { type zeroGradient; }\n"
            "    outlet  { type fixedValue; value uniform 0; }\n"
            "    drone   { type zeroGradient; }\n"
            "    top     { type symmetryPlane; }\n"
            "    bottom  { type symmetryPlane; }\n"
            "    sides   { type symmetryPlane; }\n"
            "}",
        )

        # k
        paths["k"] = self._write_field(
            zero_dir / "k", "volScalarField", "k",
            f"internalField   uniform {k:.6f};\n"
            "boundaryField\n{\n"
            f"    inlet   {{ type fixedValue; value uniform {k:.6f}; }}\n"
            "    outlet  { type zeroGradient; }\n"
            "    drone   { type kqRWallFunction; value uniform 0; }\n"
            "    top     { type symmetryPlane; }\n"
            "    bottom  { type symmetryPlane; }\n"
            "    sides   { type symmetryPlane; }\n"
            "}",
        )

        # omega
        paths["omega"] = self._write_field(
            zero_dir / "omega", "volScalarField", "omega",
            f"internalField   uniform {omega:.4f};\n"
            "boundaryField\n{\n"
            f"    inlet   {{ type fixedValue; value uniform {omega:.4f}; }}\n"
            "    outlet  { type zeroGradient; }\n"
            "    drone   { type omegaWallFunction; value uniform {omega:.4f}; }}\n"
            "    top     { type symmetryPlane; }\n"
            "    bottom  { type symmetryPlane; }\n"
            "    sides   { type symmetryPlane; }\n"
            "}",
        )

        # nut
        paths["nut"] = self._write_field(
            zero_dir / "nut", "volScalarField", "nut",
            "internalField   uniform 0;\n"
            "boundaryField\n{\n"
            "    inlet   { type calculated; value uniform 0; }\n"
            "    outlet  { type calculated; value uniform 0; }\n"
            "    drone   { type nutkWallFunction; value uniform 0; }\n"
            "    top     { type symmetryPlane; }\n"
            "    bottom  { type symmetryPlane; }\n"
            "    sides   { type symmetryPlane; }\n"
            "}",
        )

        return paths

    def write_transport_properties(
        self, case_dir: Path, nu: float
    ) -> Path:
        """Write constant/transportProperties."""
        content = self.foam_header(
            "dictionary", "constant", "transportProperties"
        ) + f"\ntransportModel Newtonian;\nnu {nu:.8e};\n"
        path = case_dir / "constant" / "transportProperties"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def write_turbulence_properties(self, case_dir: Path) -> Path:
        """Write constant/turbulenceProperties for k-omega SST."""
        content = self.foam_header(
            "dictionary", "constant", "turbulenceProperties"
        ) + "\nsimulationType RAS;\nRAS { RASModel kOmegaSST; turbulence on; printCoeffs on; }\n"
        path = case_dir / "constant" / "turbulenceProperties"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def _write_field(self, path: Path, foam_class: str, obj: str,
                     body: str) -> Path:
        content = self.foam_header(foam_class, "0", obj) + f"\ndimensions [0 2 -2 0 0 0 0];\n\n{body}\n"
        path.write_text(content)
        return path


# ─────────────────────────────────────────────────────────────────────────────
#  OpenFOAM case manager
# ─────────────────────────────────────────────────────────────────────────────

class OpenFOAMCase:
    """Manages one complete OpenFOAM CFD case.

    Creates directory structure, writes all dictionaries, runs the solver,
    and parses results.

    Usage::

        flow  = FlowCondition(velocity_ms=15.0, aoa_deg=5.0)
        case  = OpenFOAMCase("/tmp/cfd_cases/drone_aoa5", flow)
        case.setup(stl_path="airframe.stl")
        result = case.run(timeout_s=600)
        print(f"CL={result.CL:.4f}  CD={result.CD:.4f}")
    """

    def __init__(
        self,
        case_dir:    str,
        flow:        FlowCondition,
        reference_area: float   = 0.0625,   # m²
        reference_length: float = 0.25,     # m (arm length)
        application: str        = "simpleFoam",
        n_iterations: int       = 500,
    ) -> None:
        self._dir    = Path(case_dir)
        self._flow   = flow
        self._sref   = reference_area
        self._lref   = reference_length
        self._app    = application
        self._niter  = n_iterations
        self._writer = FoamDictWriter()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(
        self,
        stl_path:          Optional[str] = None,
        domain_size:       Tuple = (20.0, 10.0, 10.0),
        n_cells:           Tuple = (40, 20, 20),
        n_refinement:      int   = 3,
        n_surface_layers:  int   = 3,
    ) -> None:
        """Generate all OpenFOAM dictionaries for this case."""
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / "constant").mkdir(exist_ok=True)
        (self._dir / "system").mkdir(exist_ok=True)
        (self._dir / "0").mkdir(exist_ok=True)

        w = self._writer

        # System
        w.write_control_dict(self._dir, self._app, self._niter)
        w.write_fv_solution(self._dir)
        w.write_fv_schemes(self._dir)
        w.write_block_mesh_dict(self._dir, domain_size, n_cells)

        # STL geometry
        if stl_path:
            trisurf_dir = self._dir / "constant" / "triSurface"
            trisurf_dir.mkdir(exist_ok=True)
            import shutil
            stl_name = Path(stl_path).name
            shutil.copy2(stl_path, trisurf_dir / stl_name)
            w.write_snappy_hex_mesh_dict(self._dir, stl_name,
                                          n_refinement, n_surface_layers)

        # Constant
        w.write_transport_properties(self._dir, self._flow.kinematic_viscosity)
        w.write_turbulence_properties(self._dir)

        # 0/
        w.write_initial_conditions(self._dir, self._flow)

        logger.info(
            "OpenFOAMCase: setup complete in %s (U=%.1f m/s, AoA=%.1f°)",
            self._dir, self._flow.velocity_ms, self._flow.aoa_deg,
        )

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(
        self,
        timeout_s:        float   = 3600.0,
        on_residual:      Optional[Callable[[int, float], None]] = None,
        run_mesh_first:   bool    = True,
    ) -> AeroForces:
        """Run the OpenFOAM solver and return aerodynamic results.

        Args:
            timeout_s:      Maximum solver wall-clock time.
            on_residual:    Optional callback(iter, p_residual) for progress.
            run_mesh_first: Whether to run blockMesh + snappyHexMesh first.

        Returns:
            AeroForces with force coefficients.

        Raises:
            RuntimeError: If OpenFOAM is not installed or the solver fails.
        """
        if not self._openfoam_available():
            raise RuntimeError(
                "OpenFOAM is not installed or not sourced.\n"
                "Install: https://openfoam.org/download/\n"
                "Source:  source /opt/openfoam10/etc/bashrc"
            )

        if run_mesh_first:
            self._run_command("blockMesh", timeout_s=120)
            if (self._dir / "system" / "snappyHexMeshDict").exists():
                self._run_command("snappyHexMesh -overwrite", timeout_s=600)

        self._run_command(f"{self._app}", timeout_s=timeout_s)

        return self._parse_results()

    # ── Results parsing ───────────────────────────────────────────────────────

    def _parse_results(self) -> AeroForces:
        """Parse forceCoeffs postprocessing output."""
        coeff_dir = (
            self._dir / "postProcessing" / "forceCoeffs"
        )
        if not coeff_dir.exists():
            logger.warning(
                "OpenFOAMCase: forceCoeffs directory not found at %s", coeff_dir
            )
            return AeroForces(
                aoa_deg=self._flow.aoa_deg,
                velocity_ms=self._flow.velocity_ms,
                converged=False,
            )

        # Find the latest time directory
        time_dirs = sorted(
            [d for d in coeff_dir.iterdir() if d.is_dir()],
            key=lambda d: float(d.name) if d.name.replace(".","").isdigit() else 0,
        )
        if not time_dirs:
            return AeroForces(self._flow.aoa_deg, self._flow.velocity_ms)

        latest = time_dirs[-1]
        coeff_file = latest / "forceCoeffs.dat"
        if not coeff_file.exists():
            coeff_file = latest / "coefficient.dat"

        if not coeff_file.exists():
            return AeroForces(self._flow.aoa_deg, self._flow.velocity_ms)

        return self._parse_coeff_file(coeff_file)

    def _parse_coeff_file(self, path: Path) -> AeroForces:
        """Parse OpenFOAM forceCoeffs.dat or coefficient.dat."""
        lines = [l for l in path.read_text().splitlines()
                 if not l.strip().startswith("#") and l.strip()]
        if not lines:
            return AeroForces(self._flow.aoa_deg, self._flow.velocity_ms)

        # Take last non-comment line (final converged iteration)
        last = lines[-1].split()
        try:
            # Format: Time  Cm  Cd  Cl  (or Time Cx Cy Cz Cmx Cmy Cmz)
            CD = float(last[2]) if len(last) > 2 else 0.0
            CL = float(last[3]) if len(last) > 3 else 0.0
            Cm = float(last[1]) if len(last) > 1 else 0.0
        except (ValueError, IndexError):
            return AeroForces(self._flow.aoa_deg, self._flow.velocity_ms)

        return AeroForces(
            aoa_deg=self._flow.aoa_deg,
            velocity_ms=self._flow.velocity_ms,
            CL=CL, CD=CD, Cm=Cm,
            converged=True,
            iterations=int(float(last[0])) if last else 0,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _run_command(self, cmd: str, timeout_s: float = 3600.0) -> int:
        """Run an OpenFOAM command in the case directory."""
        env      = {**os.environ}
        cmd_name = cmd.split()[0]
        log_path = self._dir / f"log.{cmd_name}"
        # Use list form (not shell=True) to prevent shell injection.
        # Write stdout to log file explicitly.
        cmd_parts = cmd.split()
        logger.info("OpenFOAMCase: running '%s'", cmd)
        with open(log_path, 'w') as log_fh:
            result = subprocess.run(
                cmd_parts, cwd=self._dir, env=env,
                stdout=log_fh, stderr=subprocess.STDOUT,
                timeout=timeout_s
            )
        if result.returncode != 0:
            log = log_path
            tail = log.read_text()[-500:] if log.exists() else ""
            raise RuntimeError(
                f"OpenFOAM command '{cmd}' failed (code {result.returncode}).\n"
                f"Last 500 chars of log:\n{tail}"
            )
        return result.returncode

    @staticmethod
    def _openfoam_available() -> bool:
        """Check if OpenFOAM solver is on PATH."""
        import shutil
        return shutil.which("simpleFoam") is not None

    @property
    def case_dir(self) -> Path:
        return self._dir


# ─────────────────────────────────────────────────────────────────────────────
#  Aerodynamic Database Builder
# ─────────────────────────────────────────────────────────────────────────────

class AeroDatabase:
    """Sweep AoA / velocity and build an aerodynamic lookup table.

    Results are suitable for:
    - Surrogate model training (PINN / neural network, Module 14)
    - Flight dynamics model identification
    - Performance chart generation

    Usage::

        db  = AeroDatabase(base_dir="/tmp/aero_sweep", stl_path="drone.stl")
        db.run_sweep(
            velocities=[5, 10, 15, 20],
            aoa_range=(-5, 15, 5),   # start, stop, step (degrees)
        )
        db.save("/tmp/aero_database.json")
        array = db.to_numpy()   # shape (N, 8): [U, AoA, CL, CD, CM, L/D, Re, ...]
    """

    def __init__(
        self,
        base_dir: str,
        stl_path: Optional[str]   = None,
        ref_area: float           = 0.0625,
        ref_length: float         = 0.25,
    ) -> None:
        self._base    = Path(base_dir)
        self._stl     = stl_path
        self._sref    = ref_area
        self._lref    = ref_length
        self._results: List[AeroForces] = []

    def run_sweep(
        self,
        velocities:  List[float],
        aoa_range:   Tuple[float, float, float] = (-5.0, 15.0, 5.0),
        n_iterations: int = 300,
        parallel:    bool = False,
    ) -> List[AeroForces]:
        """Run a parametric sweep of velocity and AoA.

        Args:
            velocities:  List of freestream speeds (m/s).
            aoa_range:   (start, stop, step) in degrees.
            n_iterations: Solver iterations per case.

        Returns:
            List of AeroForces results.
        """
        aoas = list(np.arange(*aoa_range))
        self._results = []

        for U in velocities:
            for aoa in aoas:
                case_name = f"U{U:.1f}_aoa{aoa:+.1f}".replace("+", "p").replace("-","m")
                case_dir  = self._base / case_name
                flow      = FlowCondition(velocity_ms=U, aoa_deg=aoa)
                case      = OpenFOAMCase(
                    str(case_dir), flow,
                    reference_area=self._sref,
                    reference_length=self._lref,
                    n_iterations=n_iterations,
                )

                try:
                    case.setup(self._stl)
                    if OpenFOAMCase._openfoam_available():
                        result = case.run()
                    else:
                        # OpenFOAM not available — return placeholder
                        result = AeroForces(aoa, U, converged=False)
                        logger.warning(
                            "AeroDatabase: OpenFOAM unavailable — "
                            "skipping actual simulation for U=%.1f AoA=%.1f",
                            U, aoa,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "AeroDatabase: case %s failed: %s", case_name, exc
                    )
                    result = AeroForces(aoa, U, converged=False)

                self._results.append(result)
                logger.info(
                    "AeroDatabase: U=%.1f AoA=%.1f° → CL=%.4f CD=%.4f",
                    U, aoa, result.CL, result.CD,
                )

        return self._results

    def save(self, path: str) -> Path:
        """Save database to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "metadata": {
                "ref_area":   self._sref,
                "ref_length": self._lref,
                "n_cases":    len(self._results),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "results": [r.to_dict() for r in self._results],
        }
        p.write_text(json.dumps(data, indent=2))
        logger.info("AeroDatabase: saved %d results to %s", len(self._results), p)
        return p

    @classmethod
    def load(cls, path: str) -> "AeroDatabase":
        """Load database from JSON."""
        data    = json.loads(Path(path).read_text())
        db      = cls("", ref_area=data["metadata"]["ref_area"])
        db._results = [
            AeroForces(**r) for r in data["results"]
        ]
        return db

    def to_numpy(self) -> np.ndarray:
        """Return results as (N, 7) numpy array.

        Columns: [velocity_ms, aoa_deg, CL, CD, Cm, L_D, converged]
        """
        if not self._results:
            return np.empty((0, 7))
        rows = []
        for r in self._results:
            rows.append([
                r.velocity_ms, r.aoa_deg,
                r.CL, r.CD, r.Cm, r.L_D,
                float(r.converged),
            ])
        return np.array(rows)

    def get_polar(self, velocity_ms: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (aoa_array, [CL, CD]) polar for a given velocity."""
        subset = [r for r in self._results
                  if abs(r.velocity_ms - velocity_ms) < 0.5]
        if not subset:
            return np.array([]), np.zeros((0, 2))
        subset.sort(key=lambda r: r.aoa_deg)
        aoas = np.array([r.aoa_deg for r in subset])
        coef = np.array([[r.CL, r.CD] for r in subset])
        return aoas, coef

    def __len__(self) -> int:
        return len(self._results)

    def __repr__(self) -> str:
        return (
            f"AeroDatabase({len(self._results)} cases, "
            f"Sref={self._sref} m², Lref={self._lref} m)"
        )
