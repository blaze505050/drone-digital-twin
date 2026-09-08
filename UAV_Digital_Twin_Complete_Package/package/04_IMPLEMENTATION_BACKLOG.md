# Implementation Backlog and Migration Plan

**Proposed, not executed • 7 September 2026**

This is a dependency-ordered plan, not a request to rebuild all 28 modules simultaneously. Existing source paths are from the inspected repository; **new** paths are proposed. Estimates are focused engineering person-days (roughly 6–8 productive hours), excluding procurement, access delays and experimental waiting time. They are planning ranges, not quotes. R1 is a selected subset; the entire backlog is a longer programme.

## 1. Build order

```text
Claims + data status + contracts
          |
Physics/estimator regressions and fixes
          |
Asset config + real-log ingestion + shared model
          |
Propulsion/pack calibration -> held-out prediction baseline
          |
Uncertainty + reports -> one measured design decision -> R1 release
          |
Bounded updates + real SITL transport + frame/CFD depth -> R2
          |
One preregistered research study -> external reproduction
```

Do not make CFD, RL, vision or F Prime prerequisites for the propulsion-energy vertical slice.

## 2. P0 — make the project trustworthy

### B01 — publish an honest capability/evidence table
- **Files:** `README.md`, `pyproject.toml`, **new** `docs/claims.md`; demo output strings.
- **Change:** replace production/zero-shot claims with scoped research status; show implemented/tested/calibrated/validated and synthetic/real status per feature. Generate test badges from CI. Preserve “inspired” attribution.
- **Done:** every numeric/compatibility claim links to an artifact or is explicitly unverified/planned; no invented test count. Audit A13; REP-01.
- **Depends/effort:** none; 0.5–1 day.

### B02 — fail-closed benchmark provenance
- **Files:** `validation_framework/{benchmarks,aerodynamic_benchmarks,flight_log_validator}.py`, `battery_twin/nasa_adapter.py`, OpenFOAM training export; **new** manifest/status contracts.
- **Change:** REAL/SYNTHETIC/INVALID/MISSING status cannot be overwritten by fallback; real-mode missing data cannot score. Remove truth-derived synthetic sensors from measured-data paths.
- **Done:** V00 negative tests cover missing files, parse failures, wrong checksum and synthetic inputs. A07/A08/A10/A12; DAT-01/02.
- **Depends/effort:** B01; 2–4 days.

### B03 — frames, clocks and stream contracts
- **Files:** `state_manager/schema.py`, `store.py`, telemetry source/recorder, core outputs; **new** `contracts/`.
- **Change:** document NED/FRD quaternion/specific-force semantics; preserve acquisition/receive clocks; separate estimate, prediction, truth, intent and applied actuation.
- **Done:** V01/V04 frame/time/schema tests; invalid/wrong-asset/nonfinite data rejected; no command target masquerading as measurement. A03/A04/A06; CFG-01/DAT-02/03.
- **Depends/effort:** B01; 3–5 days.

### B04 — correct rigid-body dynamics before tuning
- **Files:** `digital_twin_core/twin_model.py`; **new** `physics/{frames,rigid_body,rotors}.py` and verification tests.
- **Change:** NED force/gravity, quaternion rotation, full inertia Euler equation, rotor position/spin contract, mass-independent actuator limits, valid dt/input policy.
- **Done:** V01 free-fall/hover/tilt/torque-free/reference integration tests; no copied self-reference oracle. A01; PHY-01.
- **Depends/effort:** B03; 3–6 days.

### B05 — correct estimator measurement equations
- **Files:** `digital_twin_core/estimator.py`, estimator tests.
- **Change:** magnetic-field observation/Jacobian and disturbance gate; explicit specific-force; review error-state injection/reset covariance; quality/validity and GPS status handling.
- **Done:** known-heading counterexample now corrects; stationary/biased IMU and covariance regressions; invalid sensors do not produce healthy valid output. A03; EST-01.
- **Depends/effort:** B03; 3–6 days.

### B06 — fix battery voltage scoring and loading
- **Files:** `battery_twin/nasa_adapter.py`, model tests.
- **Change:** `v_terminal` access; test actual model dependence, not fallback constants; correct state reset/initial conditions and cell/pack scale; fixture-controlled parse failure status.
- **Done:** deliberate model-voltage perturbation changes score; invalid source cannot return REAL. A07; ENG-01/DAT-01.
- **Depends/effort:** B02; 1–2 days.

### B07 — disable unsafe/unjustified adaptation
- **Files:** `digital_twin_core/recalibrator.py`, orchestration.
- **Change:** offline/manual by default; freeze unidentifiable mass/thrust updates; correct frame conversion; log rejection reasons; prohibit silent in-place promotion.
- **Done:** insufficient excitation and stale/invalid data lead to no update. A04; ID-01/SAF-01.
- **Depends/effort:** B03–B05; 1–3 days.

### B08 — basic package/CI and deterministic regression gate
- **Files:** `pyproject.toml`, `setup.py`, `.github/workflows/`, tests.
- **Change:** select/test supported Python versions; install core and extras in clean environments; correct `all` semantics; dependency lock or reproducible constraints; warning policy; seeded tests.
- **Done:** core regression suite runs without optional hardware/ML; required integration skips are visible, not passes; README install verified. A13; REP-02.
- **Depends/effort:** B01; 2–3 days.

## 3. P1 — minimum defensible portfolio release

### B09 — reference asset and component mass model
- **Files:** `cad_engine`, **new** `configuration/`, `configs/vehicles/`, `hardware/BOM` and drawings.
- **Change:** measured asset identity, complete components/transforms, mass/CG/full inertia, invalid-mesh handling and explicit approximation labels.
- **Done:** V02 analytic orientation/translation tests and actual mass/CG check; no dropped products of inertia. A09; CFG-01/02.
- **Depends/effort:** B03/B04; 3–6 days plus measurement access.

### B10 — deterministic real-log replay
- **Files:** `validation_framework` loaders, telemetry recorder; **new** replay service and data manifests.
- **Change:** implement one real PX4 ULog or strict measured CSV path first; separate reference/sensors/outputs; clock alignment and reproducible replay. EuRoC native VIO task is a separate optional adapter.
- **Done:** V04/V05 real run with source trace; never generate GPS from truth; explicit missing signals. A12; DAT-02/03/04/INT-01.
- **Depends/effort:** B02/B03/B08; 4–7 days.

### B11 — installed propulsion baseline
- **Files:** **new** `physics/actuators.py`, propulsion calibration and experiment definitions; BEMT adapter.
- **Change:** authentic UIUC ingestion and measured interpolation baseline; installed RPM/command/voltage/thrust/power map; explicit motor lag and uncertainty.
- **Done:** V03 held-out curves with residuals; geometry/RPM/source matched. A10; PHY-02/03.
- **Depends/effort:** B02/B04/B09; 4–8 days plus safe bench campaign.

### B12 — pack-specific ECM/energy baseline
- **Files:** `battery_twin`, **new** calibration/profile loaders.
- **Change:** real pack topology and current/voltage/temperature calibration; identifiable ECM parameters, SOC initial condition and thermal assumptions; integrate electrical Wh.
- **Done:** V07 measured held-out profiles; voltage and energy separately scored with uncertainty; no NASA-to-pack validation shortcut. ENG-01.
- **Depends/effort:** B06/B10/B11; 4–8 days plus measurement access.

### B13 — shared plant and core orchestration
- **Files:** `digital_twin_core/__init__.py`, `twin_model.py`, wrapper interfaces.
- **Change:** same corrected kernel and configuration for offline replay/twin; explicit reseeding/horizons; synchronized estimate/plant publication; prediction input lineage.
- **Done:** V01/V04 identical kernel semantics across wrappers; no observation overwrite counted as forecast accuracy. PHY-01/INT-01.
- **Depends/effort:** B04/B05/B09–B12; 3–5 days.

### B14 — baseline benchmark and prospective-information boundary
- **Files:** `validation_framework`, **new** `experiments/energy_forecast/`.
- **Change:** nominal fixed vs offline-calibrated fixed; command-replay and prospective forecast are different tasks; run/configuration holdout.
- **Done:** V11 report with per-run error, bias, counts and failures; actual future commands not used in a claimed prospective forecast. ENG-02/ID-02.
- **Depends/effort:** B10–B13; 3–6 days.

### B15 — useful uncertainty and applicability
- **Files:** **new** `uncertainty/`, core/report schemas.
- **Change:** input/parameter sensitivity, practical bootstrap/ensemble or residual calibration; validity domain; shared-sensor correlation assumptions; explicit unknown state.
- **Done:** held-out coverage and width, OOD flag and frozen adaptation. UQ-01/02.
- **Depends/effort:** B14; 3–6 days.

### B16 — engineering cockpit and automated report
- **Files:** `dashboard`, **new** report templates/CLI.
- **Change:** plots for aligned physical estimate, prediction, residual, energy/voltage interval and evidence; config/model/data links; no UI-only scope explosion.
- **Done:** generated report from a fresh environment; every chart has units/source/conditions; unknown data not green. UI-01/02/INT-01.
- **Depends/effort:** B10/B14/B15; 3–5 days.

### B17 — one measured design decision and release
- **Files:** **new** `experiments/design_trade/`, technical report, release metadata, demo and portfolio assets.
- **Change:** prerecord a safe propeller/payload/frame choice prediction; measure result; publish limitations, requirement status and personal contribution.
- **Done:** V12 and Gate G3; tagged reproducible release and concise six-minute demo. ENG-03/REP-02.
- **Depends/effort:** B09–B16; 3–6 days plus physical test scheduling.

## 4. P2 — exceptional depth after R1

### B18 — bounded identification and rollback
- **Files:** **new** `identification/`; replacement recalibrator adapter.
- **Change:** designed excitation, rank/conditioning, constrained fit, candidate/active models, holdout promotion, bounded changes, rollback and fault freeze.
- **Done:** V06 identifiable and intentionally unidentifiable cases; updated model beats or honestly fails fixed baseline. ID-03.
- **Depends/effort:** B07/B14/B15; 6–12 days.

### B19 — real PX4/MAVLink transport and safety gateway
- **Files:** `mission_planner/sinks.py`, `mavlink_bridge`, **new** `safety/`, SITL integration tests.
- **Change:** actual transport, heartbeat/target/mode/freshness checks; proper command ACK vs streaming semantics; configured autopilot failsafes; no hardwired disarm-on-timeout.
- **Done:** V13 peer log, packet decode, link-loss/startup tests; default demos remain read-only. A05; INT-02/SAF-02.
- **Depends/effort:** B03/B08/B13; 5–10 days.

### B20 — true Gym wrapper, only then RL experiments
- **Files:** `rl_controller/__init__.py` split into wrappers/rewards/randomization.
- **Change:** actual Gym spaces/checker, shared dynamics, independent seeded RNGs, implemented latency/wind/motor/drag parameters, controller baseline.
- **Done:** API and deterministic seed tests; each randomized parameter measurably affects the intended mechanism. No sim-to-real claim from training reward. A02.
- **Depends/effort:** B13; 3–6 days, excluding policy research.

### B21 — BEMT model credibility
- **Files:** `cad_engine/bemt.py`, polar loaders and validation cases.
- **Change:** continuous/consistent hover-forward formulation, Reynolds-dependent polars as available, convergence diagnostics, no silent clipping-as-success.
- **Done:** V03 matched CT/CP/efficiency sweeps, error/uncertainty and failure regions. A10; PHY-03.
- **Depends/effort:** B11; 5–10 days.

### B22 — repair and prove one OpenFOAM pipeline
- **Files:** `openfoam_bridge/__init__.py` split into case writer/runner/parser; solver container and canonical cases.
- **Change:** correct dimensions/flow/reference values; declared distribution/version; checkMesh, patch/location validation, residual and force-convergence gates; prohibit bad training rows.
- **Done:** V08 real solver artifact and mesh study; no claim from generated dictionaries alone. A08; PHY-04.
- **Depends/effort:** B02/B08/B09; 6–12 days plus compute time.

### B23 — assembled frame model and test correlation
- **Files:** `structural_twin`, structural CAD/fixture/measurement files.
- **Change:** full frame/joints/installed masses, material assumptions, static/modal solutions and observable mode shapes; compare measured baseline, not only self-model reference.
- **Done:** V09 frequency/deflection evidence with repeatability and joint/material limitations. A11; PHY-05.
- **Depends/effort:** B09; 8–15 days plus lab access.

### B24 — health indicators without invented RUL
- **Files:** `residual_monitor.py`, `predictive_maintenance`, structural health interface.
- **Change:** domain/quality-aware thresholds, debounce, unknown class, hypothesis labels; residual and vibration/electrical evidence; labelled replay/bench events.
- **Done:** V10 false alarms/misses/delay; fatigue/RUL disabled or clearly research-only without material/lifetime data. HLT-01.
- **Depends/effort:** B15/B18/B23 as applicable; 4–8 days.

### B25 — own-aircraft flight correlation / HITL
- **Files:** hardware interface and operating procedures, data manifests and flight test reports.
- **Change:** first prove actual controller loop in supervised HITL; separately review physical shadow-flight test; suitable truth/telemetry reference and safety plan.
- **Done:** Gate G4 with external peer/hardware evidence; no inference from serialization or synthetic logs.
- **Depends/effort:** B19 and local authorization; 5–12 engineering days plus potentially substantial scheduling/training.

## 5. P3 — research and optional ecosystem

### B26 — preregister and execute one novelty study
- Select identification-gated forecasting or coupled design, not both by default. Include strong fixed and data-driven baselines, ablations, locked test set and uncertainty. **Done:** claim survives appropriate evaluation or negative result is transparently published. Depends B17/B18; 15–30+ days plus data acquisition.

### B27 — release an externally reproducible benchmark
- Versioned licensed data, configurations, calibration sheets, schemas, notebooks/CLI, container and contribution statement; archive release and request independent reproduction. **Done:** another person rebuilds central figures or reproducibility blockers are documented. Depends B17/B26; 4–8 days.

### B28 — evaluate surrogates only if useful
- Compare interpolation, polynomial regression, Gaussian process and neural/PINN candidates at equal data budgets; test runtime/accuracy/OOD. Use the term PINN only for an explicitly described physics-constrained objective. **Done:** measured benefit over simpler baseline or defer. Depends B21/B22; 5–15+ days.

### B29 — optional F Prime/ROS/vision integration
- Independent scoped examples with external peers/sensor files and clear compatibility claims. Fix command-to-telemetry contamination if adapter retained. No mandatory dependency for core energy result. **Done:** byte/semantic/peer test and documented bounds. Effort requires a separate mini-PRD before starting.

### B30 — optional material fatigue/lifetime study
- Validated rainflow, load spectra, material/layup/R-ratio data, environment/mean-stress assumptions and longitudinal validation; do not equate anomaly score to RUL. **Done:** defensible life-uncertainty study, not generic percentages. Multi-month research; not R1/R2 critical path.

## 6. First 14 days — concrete starting sequence

**Days 1–2:** tag/archive current baseline locally; record what actually runs; create claims table; rewrite README opening/status; select flagship quantity (electrical energy) and reference asset/dataset. No claim of test success until run.

**Days 3–4:** write failing NED free-fall/hover/tilt and magnetometer heading tests; write failed-data and wrong-voltage-field tests; fix the lowest-level defects. Save before/after test evidence.

**Days 5–6:** freeze frame/specific-force/clock contracts; separate measurement/estimate/prediction/truth/commands; disable automatic recalibration by default.

**Days 7–8:** build vehicle/component and dataset manifest templates; choose real propulsion/pack/log sources; confirm permissions and bench instrumentation access.

**Days 9–10:** ingest one genuine measured dataset; show real source status and deterministic replay; remove truth leakage and benchmark truncation shortcuts.

**Days 11–12:** create the first nominal-vs-calibrated baseline report—even if accuracy is poor. Plot residuals and define the first justified parameter fit.

**Days 13–14:** review evidence/limitations, freeze the initial evaluation protocol and safety plan, create a prioritized issue board for B09–B17. Do not spend these two weeks on animation, a copilot or swarm features.

These are sequenced working-day suggestions, not a promise that 14 calendar days at part-time availability completes all P0 tickets. Keep unfixed features disabled/labelled until their gates pass.

## 7. Pull-request discipline and issue definition of done

One change per PR with linked requirement/audit IDs; failing regression first; equations/units documented for physics changes; no silent default change; example/data/README updated; evidence attached; reviewer/checklist; rollback compatibility noted. Keep runtime fixes separate from formatting-only refactors.

Each issue records owner/role, priority, dependencies, acceptance tests, uncertainty/assumptions, required equipment, evidence output, estimate and status. A closed issue means its acceptance evidence exists—not merely that a new class was added.
