# Verification, Validation and Experimental Plan

**Proposed plan • 7 September 2026 • no numerical result in this document has been measured by this review.**

## 1. Credibility rules

**Verification:** are the equations and algorithms implemented/solved correctly? **Validation:** do they represent the measured physical system well enough for the stated decision and domain? **Uncertainty quantification:** how uncertain are inputs, parameters and predictions, and do reported intervals behave honestly?

Use NASA model-credibility guidance and selected ASME V&V practices as a framework. ASME V&V 10 concerns computational solid mechanics; V&V 20 concerns CFD/heat transfer. RMSE plots or unit-test counts do not establish compliance with either standard. See reference document.

A model can be correct as software, calibrated to one dataset, and still invalid for a different vehicle. Every validation statement must name the quantity, operating domain, physical configuration, reference and uncertainty.

## 2. Evidence pyramid

1. Unit/property and analytic tests: units, signs, frames, symmetries, physical bounds.
2. Numerical verification: integration/mesh/time-step convergence; independent reference solvers.
3. Synthetic identification and fault cases: known parameters/truth, seeded, explicitly synthetic.
4. Authentic public subsystem benchmarks: matched inputs and conditions, versioned manifests.
5. Own-component bench correlation: calibrated instruments, repeat runs, held-out conditions.
6. Complete asset shadow-operation correlation: real flight/operating logs, appropriate independent references.
7. Decision validation: predict a design/configuration change before physically testing it.

Passing a lower level never proves the higher one. Shared-kernel synthetic tests are regressions, not independent validation. Include independent analytic or differently implemented references to avoid the same bug appearing on both sides.

## 3. Data design and leakage prevention

- Split by flight/run, battery cycle group, propeller/configuration and test day as appropriate—not random adjacent rows from one time series.
- Maintain separate training, tuning/validation, and final locked test sets. Parameter identification and interval calibration cannot use final test outcomes.
- Record which conditions are interpolation vs extrapolation: RPM, inflow, voltage, SOC, temperature, mass, wind, manufacturing configuration and age.
- Never derive “measured GPS” by perturbing reference truth in a real-data benchmark. Such a generator belongs only in synthetic verification.
- Do not treat onboard fused navigation as independent truth for an estimator using the same sensors. Report agreement to onboard estimate separately.
- Do not infer whole-vehicle energy from EuRoC, whole-pack performance from NASA cell aging, or complete rotor performance from a sectional airfoil polar.
- Named external datasets must be downloaded legitimately, checksummed and parsed natively. Missing/invalid real data yields NOT_SCORED; synthetic fallback is allowed only in a separate demo command.
- Preserve failed tests, solver nonconvergence and out-of-domain records. Report exclusion counts and reasons fixed in advance.

## 4. Test matrix and exit criteria

Thresholds are initial proposals; tighten/relax only with documented pilot-test rationale before locking final data. A physical tolerance below instrument uncertainty is not meaningful.

| Test | Scope and protocol | Exit evidence |
|---|---|---|
| V00 | Provenance and scoring firewall | Missing file, wrong checksum, parse failure, synthetic status and truth-derived sensor fixture cannot produce REAL/PASS |
| V01 | Physics/estimator verification | Free-fall in positive NED z; level hover; single-rotor torque signs; tilted thrust rotation; full-tensor torque-free motion; magnetic known-heading correction; specific-force conversion; dt convergence and nonfinite-input rejection |
| V02 | CAD/assembly mass properties | Analytic translated/rotated/reversed box and tetrahedron; shell/overlap handling; complete assembly measured mass/CG; full inertia and uncertainty; no `abs()` masking invalid tensors |
| V03 | Propulsion system | Authentic UIUC matched propeller curves plus installed bench RPM/voltage sweep; thrust, reaction torque where instrumented, electrical power, delay and uncertainty; held-out operating points |
| V04 | Clock/alignment/replay | Reorder, duplicate, drift, delay, clock reset, dropped packet and gap scenarios; same replay results from same log/seed; no residual between mismatched epochs |
| V05 | Estimation/reference | Independent sensor and truth channels; native timestamps; appropriate SE(3) alignment declared; ATE/RPE/attitude/velocity and consistency; no arbitrary scale alignment for metric-state claims |
| V06 | Identification/update | Inject known changes in independent synthetic plant; design bench excitation; evaluate rank/conditioning; holdout fitting; freeze and rollback; indistinguishable parameter case must be refused |
| V07 | Battery/energy subsystem | Correct ECM field use; cell/pack scaling; pulse/drive profiles; train/test cycles/temperatures separated; voltage/SOC/energy uncertainty; invalid MAT never relabelled real |
| V08 | CFD numerical credibility | Field dimensions, boundary/force-reference correctness, `checkMesh`, independent geometry reference, at least three mesh levels when practical, residual/force stationarity and time-step study when unsteady |
| V09 | Structures | Analytic beam verification followed by assembled-frame static and modal test; fixture/joint/material assumptions, installed masses; observable modes and uncertainty |
| V10 | Fault/domain/safety states | Sensor bias/dropout, latency, unknown config, propulsion-model mismatch and unavailable data in replay/SITL; adaptation freeze; false-alarm/miss/delay reporting; no safety-critical physical faults injected |
| V11 | Flagship energy/prediction study | Identical held-out runs for nominal, calibrated fixed and adaptive models; explicit forecasting information boundary; accuracy, confidence interval, coverage and runtime |
| V12 | Design-decision experiment | Predict a safe payload/propeller/frame alternative before test; manufacture/configure, measure, compare uncertainty and benefit |
| V13 | Actual transport integration | PX4 peer decodes actual frames, heartbeat/mode state known, correct ACK semantics, setpoint streaming and link loss tested; no physical arming in CI |

### Suggested numerical regression tolerances

Double-precision transforms/rotations and algebraic identities should be near machine precision under well-conditioned inputs; choose explicit absolute/relative tolerances (e.g. 1e-9 to 1e-6 depending on the computation), not one global threshold. Integrated trajectories use error-vs-step-size studies and analytic/reference solutions. Near-zero forces and off-diagonal inertia need absolute tolerances. All tolerances belong beside their physical justification.

## 5. Experiment protocols

### E1 — as-built mass and geometry, required

Inventory every installed component and weighed subassembly. Document configuration photographs, drawing references and mounting transforms. Measure total mass at least three times with a calibrated/checkable scale. Estimate CG with a documented balance/suspension method and repeatability. For inertia, use credible CAD/component aggregation initially; bifilar/trifilar pendulum or other properly designed measurement is R2. Account for fixture inertia and measurement uncertainty. Do not quote millimetre CG or percent inertia precision unsupported by the method.

**Deliverables:** BOM; assembly and mass-budget table; full tensor; CG diagram; sensitivity to payload placement; test log; discrepancy explanation.

### E2 — propulsion map, required for own-hardware R1

Use a guarded, restrained, appropriately rated thrust stand operated with qualified supervision. Measure RPM, thrust, bus voltage/current, ambient conditions and temperature; torque only with a calibrated suitable sensor. Remain within motor, ESC, propeller and battery manufacturer limits. Do not test damaged propellers or intentional failures.

Pilot plan: 6–10 steady operating points over the intended envelope, at least 3 repeat sweeps, and 2–3 voltage/SOC conditions where safely controllable. Record thermal settling and hysteresis; randomize or balance sweep order when appropriate. Select a few safe step responses to identify actuator lag. Test-cell limits and actual counts require local safety review.

Fit nominal map on selected runs; hold out whole sweeps/conditions. Compare simple measured interpolation, BEMT and any advanced model. Static UIUC data can help validate propeller coefficients; it does not identify installed electrical efficiency. No wind tunnel means no claimed measured forward-flight map; label that limitation.

### E3 — pack transient energy model, required

Use the actual approved pack chemistry/topology and a suitable load/charger or safe representative propulsion duty cycle. Log calibrated pack voltage/current and temperatures. Collect training pulses and independent variable-load profiles over a safe SOC range; cell-level measurements only through suitable equipment. Use OEM voltage/current/temperature limits and qualified battery procedures. Do not reproduce abusive/deep-discharge NASA conditions on a flight pack.

Identify OCV/SOC relation and identifiable ECM parameters; preserve parameter dependence on SOC/temperature only where data support it. Validate a held-out profile and later a flight duty cycle. Measure uncertainty and current-sensor offset, which can dominate integrated energy/SOC drift.

### E4 — frame structural correlation, R2

Specify actual material/layup, section, joints, boundary conditions and installed masses. Perform calibrated low-risk static loading and an instrumented modal test with suitable excitation and accelerometers. Avoid unsafe propeller-on vibration tests. Match the model's boundary conditions to the fixture. Test repeatability and joint preload effects before attributing frequency change to damage.

Compare deflections and observable mode frequencies; compare mode shapes/MAC only if spatial measurements support them. Isotropic carbon-fibre tube approximations are labelled preliminary; no life/strength certification from a generic S–N curve.

### E5 — energy and model-updating experiment, flagship

Choose a safe reproducible mission/duty-cycle family (hover/low-speed segments or bench profiles). Use nominal configuration and a measured, safely approved payload/configuration change. Select measured inputs that distinguish mass, actuator efficiency and battery resistance. Keep environmental conditions recorded and controlled as feasible.

Pilot planning: 12–20 independent usable runs across 2–3 conditions can establish feasibility, not broad generality. Choose final sample size from pilot variability and desired confidence/power. Repeated time samples are not thousands of independent flights. More seeds on one synthetic plant do not replace physical replication.

Evaluate nominal fixed, offline-calibrated fixed and update-gated models. Do not let the adaptive method use future test inputs unavailable to its baselines. For mission forecasting, propagate a declared planned/controller policy from issue time; for command replay, disclose the recorded-actuation oracle and do not market it as mission foresight.

### E6 — decision validation, R1/R2

Pick one useful change: propeller selection for energy, payload placement for balance, or arm geometry for stiffness/mass. Predict benefit and uncertainty before measurement. Respect structural and flight safety margins. Publish the as-predicted and as-measured result even if the change fails to improve performance. Explaining a failed prediction with measured evidence is valuable engineering.

## 6. Metrics and statistical reporting

- **RMSE:** `sqrt(mean((prediction-reference)^2))`; also report bias and MAE in physical units.
- **Normalized RMSE:** divide by a preregistered physical scale/rated range. Do not change normalization to improve appearance.
- **Energy:** `E = integral(V(t)*I(t) dt)` with time alignment, units, offset and sampling uncertainty. Percentage errors apply only to nontrivial positive-energy runs; report absolute Wh error too.
- **Trajectory:** ATE/RPE after declared time/frame handling; attitude error from relative quaternion/geodesic angle. R² can be misleading near constant signals and is secondary.
- **Intervals:** nominal level, empirical coverage with confidence interval, width/sharpness and conditional coverage by operating region. Add proper probabilistic score if justified.
- **Comparison:** paired per-run differences and bootstrap confidence intervals clustered by independent run/day/configuration where appropriate. Display all runs, not just an average.
- **Health:** false alarms per hour/run, missed-event rate, detection delay and confidence; class labels validated only against labelled events. Healthy-only data cannot establish RUL accuracy.
- **Performance:** p50/p95/p99 core execution, acquisition-to-use age, dropped records and memory on stated hardware; end-to-end and compute latency are distinct.

If the result's confidence interval overlaps no improvement, say evidence is inconclusive. Do not convert a proposed 20% improvement target into an achieved resume bullet.

## 7. Minimum evidence bundle per result

Protocol/version and requirement IDs; source/licence/checksum; raw and processed lineage; asset/configuration and model/parameter IDs; instrument/calibration and uncertainty; training/tuning/test memberships; input/output units/frames/time basis; operating points and sample/run counts; exclusions and failures; scripts/environment; metrics and plots; uncertainty/sensitivity; pass/fail/NOT_SCORED with rationale; limitations and reproduction instructions.

## 8. CI and execution tiers

- **Every change:** syntax/types/style, unit/property tests, provenance firewall, deterministic synthetic physics regressions and package installation.
- **Integration tier:** real PX4 SITL peer tests and small pinned OpenFOAM case, explicitly provisioned; fail when claimed required infrastructure is broken.
- **Scheduled/manual expensive tier:** full benchmark matrices, solver convergence studies, Monte Carlo and dataset downloads with permitted licensing.
- **Physical test tier:** human-approved lab protocols; data ingested into immutable manifests after inspection. Never trigger flight/rotating equipment from ordinary CI.

A skipped test is visible and not counted as passed experimental evidence. Test count badges come from CI, while validation badges link to actual physical benchmark artifacts.

## 9. Traceability and release review

V00 covers DAT-01/02 and REP-01; V01 PHY-01/EST-01; V02 CFG-01/02; V03 PHY-02/03; V04 DAT-03; V05 EST-02; V06 ID-01/02/03; V07 ENG-01; V08 PHY-04; V09 PHY-05; V10 UQ-02/HLT-01/SAF-01; V11 ENG-02/UQ-01; V12 ENG-03; V13 INT-02/SAF-02. Reproducibility review also covers DAT-04, INT-01, UI-01/02 and REP-02.

At release, publish a requirement-status matrix containing evidence links and unresolved deviations. A human reviewer signs off on the intended use and limitations. This is an engineering credibility review, not regulatory approval.
