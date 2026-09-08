# UAV digital-twin references (accessed 2026-09-07 user-local)

Source facts are **Fact**; advice is **Implication**. No certification/compliance or novelty claims.

1. NASA/USAF, *The Digital Twin Paradigm for Future NASA and U.S. Air Force Vehicles* (2012). https://ntrs.nasa.gov/api/citations/20120008178/downloads/20120008178.pdf
**Fact:** Integrated multiphysics, multiscale, probabilistic simulation of an as-built vehicle using physical models, sensor updates, fleet history and data to mirror its flying twin; links to onboard health management. **Implication:** Demonstrate asset-specific data→update→prediction→residual; generic simulator should be called twin-inspired/demonstrator.

2. NASA-HDBK-7009B, *Model and Simulation Implementation Guide* (approved 2026-02-03). https://standards.nasa.gov/system/files/tmp/NASA-HDBK-7009B_Final%2002-03-2026.pdf
**Fact:** Implementation guidance for NASA-STD-7009B. **Implication:** Map purpose, assumptions, data pedigree, V&V, uncertainty and configuration management to selected practices; do not claim NASA compliance without the applicable process/review.

3. ASME V&V 10, *Standard for Verification and Validation in Computational Solid Mechanics*, 2019 (R2025). https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-solid-mechanics
**Fact:** CSM standard providing common language/framework/general guidance. **Implication:** For frame/arm models separate solution verification from physical strain/deflection/modal validation; no compliance claim from plots alone.

4. ASME V&V 20, *Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer*, 2009 (R2021). https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-fluid-dynamics-and-heat-transfer
**Fact:** Compares solution/data for specified variable at specified validation point, including uncertainty; scope is experiments with simulated conditions, not unvalidated extrapolation. **Implication:** Match propeller test conditions/variables; one point cannot validate a flight envelope.

5. UIUC Propeller Database Vol. 1. https://m-selig.ae.illinois.edu/props/volume-1/propDB-volume-1.html
**Fact:** Nearly 140 small-UAV/model propellers; mostly unmodified retail; wind-tunnel thrust/torque coefficients over advance ratio at specified RPM and static RPM sweeps; some geometry approximate. **Implication:** Calibrate/cross-check propeller model with matched conditions and limits; not whole-multirotor validation; bench-test actual motor/ESC/prop/installation.

6. ETHZ ASL, *EuRoC MAV Dataset* (2016). https://projects.asl.ethz.ch/datasets/euroc-mav/
**Fact:** Stereo monochrome 2×20 FPS, synchronized MEMS IMU 200 Hz, calibration, Vicon 6D pose, Leica 3-D position/structure; caveats include auto-exposure mismatch, dynamic tracker degradation, synchronization limits. **Implication:** VIO/state-estimation benchmark only—not propulsion, battery, loads, motor dynamics or full-airframe energy.

7. NASA Ames PCoE, *Li-ion Battery Aging Datasets*. https://data.nasa.gov/dataset/li-ion-battery-aging-datasets
**Fact:** Commercial 18650 cells, charge/discharge/EIS, temperatures/loads, ~10 Hz, run-to-failure at 30% fade (2→1.4 Ah); some discharges below OEM 2.7 V. **Implication:** Cell-level SOH/RUL features only; pack transfer requires pack-specific series/parallel, balancing, thermal, transient ESC-load and BMS tests.

8. PX4 User Guide, *Simulation*. https://docs.px4.io/main/en/simulation/
**Fact:** SITL/HITL; current Gazebo recommended for new projects; multirotors, sensors, custom worlds/plugins, multi-vehicle, lockstep/faster-than-real-time. **Implication:** Established baseline, not novelty; contribute calibrated/updateable model, uncertainty/residual analysis and SITL→HITL→bench/flight comparison.

9. JSBSim Team, *JSBSim Flight Dynamics Model*. https://jsbsim-team.github.io/jsbsim/
**Fact:** Open-source, multi-platform, object-oriented C++ FDM framework. **Implication:** Prior art/platform; differentiate with multirotor calibration/update method, credible evidence and reproducibility—not “world’s first.”

10. NIST IR 8298, *A Summary of Industrial Verification, Validation, and Uncertainty Quantification* (2020). https://nvlpubs.nist.gov/nistpubs/ir/2020/NIST.IR.8298.pdf
**Fact:** Distinguishes validation from verification and discusses input/experiment/model uncertainty, including epistemic uncertainty. **Implication:** Report uncertainty with metrics and sensitivity/identifiability before updating; good fits can be non-unique.

11. National Academies, *The Digital Twin Landscape*. https://www.nationalacademies.org/read/26894/chapter/4
**Fact:** Identifies V&V and UQ as digital-twin ecosystem elements. **Implication:** Organize around use case, observable states/parameters, update data and extrapolation limits.

## Recommendations
Pick one falsifiable use case and twin boundary. Layer rigid-body/actuator dynamics, measured propeller map, cell/pack energy and update/UQ. Use operating-point holdouts and hardware tests. Use EuRoC only for VIO and NASA battery data only as a cell-level prior unless pack data exist. Present PX4/Gazebo/JSBSim as existing infrastructure; claim contribution in calibration, updating, identifiability, uncertainty and physical evidence. Say “aligned with selected practices,” not compliant/certified.


## Review boundary

This is a targeted primary-source reference set, not an exhaustive prior-art or novelty review. The source audit provides commit-pinned repository links separately. Source descriptions do not imply project compliance, certification or achieved performance.
