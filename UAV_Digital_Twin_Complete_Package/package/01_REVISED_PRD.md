# UAV Digital Twin — Revised Product Requirements Document

**Working title:** Evidence-Driven UAV Digital Twin  
**Version:** proposed PRD 3.0 • 7 September 2026  
**Status:** engineering proposal, not an implemented release  
**Repository:** https://github.com/blaze505050/drone-digital-twin  
**Source baseline reviewed:** `4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4` (public HEAD observed during review; commit dated 3 September 2026).

## 1. Executive decision

**Build one exceptionally well-evidenced UAV twin, not a larger collection of demonstrations.**

The repository already contains meaningful engineering code: an error-state estimator, forward dynamics, CAD mass integration, a BEMT propeller solver, battery and structural models, telemetry adapters, and validation scaffolding. Preserve that investment. The next release should trade breadth for mathematical correctness, physical correlation, reproducibility, and a clear decision the twin improves.

The flagship product is an **asset-specific, uncertainty-aware energy and condition twin for one electric quadrotor**. It connects as-built geometry and mass properties to propulsion, battery, rigid-body dynamics, measured telemetry, and a conservative mission-energy forecast. It detects when its predictions cease to be trustworthy and proposes bounded model updates, with human review.

The research ambition is not “the best digital twin ever made.” That cannot be established from a feature list, and no responsible PRD can guarantee historical impact. A defensible ambition is: **publish an unusually reproducible aero-electro-mechanical UAV benchmark and demonstrate one method that materially improves a practical prediction over strong baselines.** An independently reproduced result can become influential; unsupported superlatives damage credibility.

## 2. Problem, users, and value

### Problem

A simulator often represents a nominal vehicle, while a real aircraft changes with payload, battery condition, propeller selection, manufacturing tolerance, repairs, and environment. A trajectory-matching dashboard does not tell an engineer whether a model can predict energy, whether an apparent fault is actually a sensor error, or whether a design change is worth building.

### Primary users and jobs

| User | Job to be done | Evidence of value |
|---|---|---|
| UAV/mechanical engineer | Compare propeller, payload and frame alternatives | Predicted trade-off followed by a measured design iteration |
| Flight-test engineer | Replay a flight and explain model/measurement disagreement | Time-aligned residuals, uncertainty, configuration and reproducible report |
| Battery/propulsion engineer | Forecast electrical demand under a specified duty cycle | Held-out pack-voltage and mission-energy accuracy |
| Controls researcher | Use a consistent plant model with PX4 and controlled system identification | Verified frames, actuator mapping, transport and baseline comparisons |
| Recruiter/technical interviewer | Determine what the author personally designed and proved | Short demo, requirements traceability, engineering drawings, test data and honest limitations |

### Primary workflow

Select a physical vehicle configuration → load traceable measurements → calibrate only identifiable parameters → replay unseen runs → issue energy predictions and validity flags → evaluate a proposed design/configuration change → manufacture or configure one safe change → measure whether the prediction was useful.

### Secondary workflow

Run the same model through a real PX4 SITL interface for integration and safety tests. SITL is not proof of hardware accuracy. Optional later HITL and physical telemetry add evidence at separate maturity levels.

## 3. Assumptions and decisions still open

Planning assumes one contributor working approximately 15–20 hours/week, access to a laptop, and eventual access to a small quadrotor or propulsion bench. A nominal 1–2 kg, 8–10 inch electric quadrotor is a planning example, not a required purchase or a validated design. Reuse an available safe platform where possible.

Before procurement or flight, record: existing hardware; measured mass and payload; propeller/motor/ESC; pack chemistry and series/parallel configuration; sensor/ground-truth access; lab supervision; operating location and permissions; budget and deadline. Freeze those choices at Gate G0. Unknown values remain `null`/unmeasured, never plausible-looking constants.

No hardware access is not a blocker to a strong first release. Deliver a **data-linked research demonstrator** using genuine licensed datasets and clearly separated synthetic verification. Do not claim that it twins the author's own physical aircraft until that link exists.

## 4. Product boundary and maturity

### Twin boundary

One identified airframe, its installed propulsion system, battery pack, payload/configuration, environment estimates, and measurement history. The twin keeps versioned state estimates and parameters tied to that particular asset. A generic simulator is a reusable component, not itself evidence of this asset linkage.

“Digital twin” terminology varies. This project will publish its operational definition: an asset-linked model that updates from physical observations, predicts decision-relevant quantities, records uncertainty and configuration, and supports a documented engineering feedback process. Direct flight-command authority is **not required** to achieve the first release's purpose.

### Maturity labels shown in every report

- **M0 — synthetic demonstration:** generated inputs; code/physics verification only.
- **M1 — external-data correlation:** named, authentic independent data; only the measured subsystem is validated.
- **M2 — physical-asset bench twin:** actual component identities, calibration and held-out bench measurements.
- **M3 — flight-correlated shadow operation:** real vehicle logs/live telemetry, independent reference where available, no experimental control authority.
- **M4 — supervised closed-loop integration:** separately approved SITL/HITL/flight integration, with explicit safety evidence. This is not an airworthiness certification.

A module's maturity does not automatically promote the entire platform. A NASA cell fit is not M3 validation of the aircraft. EuRoC state-estimation performance is not propulsion or endurance validation.

## 5. Scope and release tiers

### R0: credibility repair and reproducible core

Repair coordinate and physics defects, synthetic-data provenance, real-vs-simulated transport claims, packaging status and benchmark scoring. Define configuration, timestamps and sensor semantics. Deliver deterministic analytic tests and a generated capabilities/evidence table.

### R1: resume-grade vertical slice — required

One vehicle configuration; measured or documented mass properties; propulsion map correlated with authentic propeller data and preferably the installed assembly; pack-voltage/energy model; real-log replay; fixed-vs-calibrated prediction comparison; uncertainty and operating-domain flags; reproducible reports; one engineering trade study. Flight control remains with the established autopilot.

### R2: exceptional multiphysics project — subsequent

Full assembly mass model; measured static/modal frame correlation; installed propeller/motor/ESC mapping; real PX4 SITL transport; bounded identification with rollback; own-vehicle flight correlation; one manufactured/tested design iteration. CFD is an offline model-development activity with independent solver and mesh verification, not a required runtime dependency.

### R3: research contribution — conditional

An uncertainty/identifiability-gated update method, or a validated coupled aero-electro-structural design method, evaluated against appropriate baselines and ablations. Select **one** primary thesis. Add experimental replication, an open benchmark release, and an external reproduction. A paper is conditional on novelty and results, not a release promise.

### Explicit non-goals for R1/R2

No weapons, target tracking or electronic-warfare mission features; no swarm autonomy; no experimental controller with direct unrestricted motor authority; no “certified airworthiness,” “NASA-approved,” or “zero-shot sim-to-real” claims; no generative copilot controlling engineering calculations; no full deforming-airframe CFD/FEM at 100 Hz; no trustworthy fatigue RUL without material and lifetime data; no mandatory Kubernetes, microservices, blockchain or photorealistic metaverse.

## 6. North-star outcome and success metrics

**North-star experiment:** on held-out, physically measured operating conditions, predict electrical energy for a specified mission/duty cycle more accurately than both a nominal fixed model and a strong offline-calibrated fixed model, while reporting calibrated uncertainty and refusing unsupported extrapolation.

For a mission forecast, only information available at forecast time may be used. A model driven by actual future recorded motor commands is a **retrospective command-replay model**, not a prospective mission forecast. Report these as different tasks.

All values below are proposed starting targets, not observed repository results or universal aerospace standards. Freeze justified thresholds after pilot experiments, before scoring the final test set. Publish any revision and retain the original threshold.

| Outcome | Initial release target | Evidence and conditions |
|---|---|---|
| Provenance | 100% of scored records have data source, status and config hash | Missing/invalid/synthetic data cannot enter real-data scores |
| Mathematical correctness | All frame, free-fall, hover, torque, units and integration regressions pass | Analytic reference and declared numerical tolerances |
| Mass model | Total mass within 2% of calibrated scale; CG within 5 mm where measurement supports it | Actual asset, uncertainty stated; inertia separately measured/estimated |
| Propulsion | Normalized RMSE ≤10% for thrust and electrical power within declared bench envelope | Define normalization from fixed rated/measured scale, exclude neither failures nor near-zero points silently |
| Pack voltage | RMSE ≤50 mV/cell equivalent on held-out dynamic profiles | Pack RMSE also reported; temperature, current and chemistry matched |
| Mission/duty-cycle energy | Median absolute percentage error ≤10% on held-out runs | Report all runs, bias, 90th percentile, sample counts and uncertainty |
| Research improvement, R3 | ≥20% relative reduction in chosen error versus offline-calibrated fixed baseline | Paired held-out runs; confidence interval must support improvement, not just one favourable run |
| Forecast uncertainty | Nominal 90% intervals approximately calibrated | Show coverage with binomial interval and interval width; no requirement to make intervals arbitrarily wide |
| Short-horizon plant prediction | Position/velocity/attitude errors at 0.1, 0.5, 1 and 2 s | Compare fixed and updated models on identical initialization; thresholds set by truth sensor and use case |
| Structural correlation, R2 | First three observable mode frequencies within 10%; static deflection within 10% | MAC >0.9 only where spatial mode-shape measurements support it; 5% frequency error is stretch |
| Timing, R2 | 100 Hz light-model target; core step p95 ≤5 ms and p99 ≤10 ms on named hardware | End-to-end telemetry age and worst-case overruns reported separately; not hard-real-time certification |
| Repeatability | Fresh environment reproduces published tables from manifests | Data availability and solver/container versions pinned; no secret local files |

The most important target is not a single percentage: **results must be independent, reproducible, attributable to a specific configuration, and useful for an engineering decision.**

## 7. Functional requirements

Priority meanings: P0 blocks credible scoring/safe use; P1 defines R1; P2 adds R2; P3 is optional research. Verification identifiers refer to the companion V&V plan.

### A. Configuration and evidence

| ID / priority | Requirement | Acceptance |
|---|---|---|
| CFG-01 / P0 | Versioned `VehicleConfig` defines asset ID, component IDs, body frame, rotor positions/directions, SI units, measured/assumed values and uncertainties | Schema rejects missing safety-critical geometry, invalid units and nonphysical mass/inertia; V01/V02 |
| CFG-02 / P1 | Assembly mass pipeline includes frame, motors, props, ESCs, battery, payload, fasteners and wiring | Full CG and inertia retained through dynamics; mesh validity exposed; V02 |
| DAT-01 / P0 | Dataset manifest records source, permission/license, checksum, acquisition, calibration, splits and status | Real-data run fails or returns NOT_SCORED if data are missing; V00 |
| DAT-02 / P0 | Separate raw sensors, onboard estimates, twin predictions, reference truth, command intent and applied actuation | Test detects any truth-derived synthetic sensor entering a real score; V00/V05 |
| DAT-03 / P0 | Preserve acquisition/receive/simulation time and clock mapping uncertainty | Reordering, lag and reset tests have deterministic outcomes; V04 |
| DAT-04 / P1 | Immutable raw logs; derived data records transformation version and parent hashes | Same manifest and code reproduce the same processed dataset within declared platform tolerances |

### B. Mechanics, propulsion and energy

| ID / priority | Requirement | Acceptance |
|---|---|---|
| PHY-01 / P0 | One tested rigid-body kernel serves twin prediction and RL/SITL wrappers | Correct NED/FRD gravity and thrust; full inertia Euler equation; V01 |
| PHY-02 / P1 | Explicit actuator model maps applied motor command and voltage to RPM, thrust and reaction torque | Measured delay/lag/saturation and rotor mapping; payload mass does not magically scale motor capability; V03 |
| PHY-03 / P1 | Propeller model offers measured-map baseline and BEMT with validity/convergence reporting | Genuine matched UIUC/bench comparison; unsupported conditions fail visibly; V03 |
| PHY-04 / P2 | CFD produces traceable offline aerodynamic data | Correct dimensions/reference quantities, mesh/time/iteration studies and independent benchmark; V08 |
| PHY-05 / P2 | Assembled frame model includes joints and installed masses with stated material limitations | Static and modal correlation to manufactured article; V09 |
| ENG-01 / P1 | ECM/SOC/thermal pack model uses actual chemistry, topology and measured parameters | Correct voltage field and pack scaling; independent pulse/flight profiles; V07 |
| ENG-02 / P1 | Energy forecast integrates electrical power over a declared future duty cycle or mission policy | Separate retrospective and prospective evaluations; V11 |
| ENG-03 / P2 | Configuration trade tool compares payload, propeller, battery and arm/frame alternatives | Uncertainty-aware feasible design comparison plus at least one physical confirmation; V12 |

### C. Estimation, calibration and validity

| ID / priority | Requirement | Acceptance |
|---|---|---|
| EST-01 / P0 | Estimator uses documented specific-force, gravity, quaternion and magnetic-field equations | Known-heading correction, IMU bias, innovation-gate and covariance tests; V01/V05 |
| EST-02 / P1 | Reference truth is independent where claimed; correlated estimates are labelled as agreement, not truth | Report schema lists reference instrument and uncertainty; V05 |
| ID-01 / P0 | Only identifiable parameter subsets may update | Rank/conditioning or excitation gate blocks unobservable mass/thrust separation; V06 |
| ID-02 / P1 | Offline fitting precedes online adaptation; calibration/test split is by run/configuration | No row-wise leakage from the same flight; fixed baseline retained; V06/V11 |
| ID-03 / P2 | Candidate parameter versions are bounded, validated and promoted atomically | Stale/faulted/OOD data freeze updates; rollback on failed holdout and divergence; V06/V10 |
| UQ-01 / P1 | Predictions carry uncertainty, method, interval level and applicability | Calibration/coverage and interval width reported on holdout, with sample limitations |
| UQ-02 / P1 | Unknown, stale, outside-domain and insufficient-excitation are first-class states | Dashboard/report cannot render these as healthy green scores; V10 |
| HLT-01 / P2 | Residual-based condition flags are tested as hypotheses, not definitive fault diagnoses | False alarms, missed detections and delay on labelled scenarios; no RUL without longitudinal evidence; V10 |

### D. Integration, safety and user experience

| ID / priority | Requirement | Acceptance |
|---|---|---|
| INT-01 / P1 | Deterministic CLI supports import, calibrate, replay, validate and report | Clean-environment example returns artifact paths and meaningful exit codes |
| INT-02 / P2 | One real PX4/MAVLink integration replaces serialization-only dispatch claims | Peer decodes messages; heartbeat/mode/setpoint handling proven in SITL; V13 |
| SAF-01 / P0 | Read-only/offline operation is default; experimental actuation is explicit opt-in | No armed hardware command from default examples/tests; V10/V13 |
| SAF-02 / P2 | Safety gateway checks vehicle ID, finite/ranged values, frame, freshness and allowed state transitions | Link loss/stale setpoint uses configured autopilot behaviour, never blindly disarms in flight; V13 |
| UI-01 / P1 | Engineering cockpit shows physical estimate, prediction, aligned residual, forecast interval, domain and evidence level | Every plot exposes units, time basis, source, asset and model version |
| UI-02 / P1 | Single command exports a professional benchmark/design-review report | Figures include conditions, uncertainty, counts, failures and reproducibility command |
| REP-01 / P0 | README and metadata distinguish implemented, tested, calibrated and validated capabilities | Remove static production claims and unsupported performance/compatibility labels |
| REP-02 / P1 | Tagged release includes technical report, data manifest, environment, licensing and contribution record | External reader can inspect and reproduce the central claim |

## 8. Non-functional requirements

**Reproducibility:** pin a supported Python baseline, tested dependencies and solver versions. Maintain fast deterministic tests and separately gated optional integration/experimental suites. Do not suppress all warnings indiscriminately. Record seeds, CPU/GPU, OS, compiler/solver versions and floating-point tolerances.

**Performance:** decouple high-rate ingestion/model steps from 10–20 Hz display and asynchronous recording. Apply bounded queues with counted drops/backpressure; no silent sample loss. Offline CFD, FEM and fitting do not block telemetry. Optimize compiled kernels only after profiling.

**Reliability:** reject NaN/Inf, invalid covariance, nonpositive mass, inconsistent clocks and corrupt packets. Reconnection must not replay stale commands or overwrite asset identity. Models can return invalid, unavailable or not converged instead of inventing plausible output.

**Security/privacy:** no telemetry credentials or home coordinates in public fixtures. Explicit network binding, access control for any command endpoint, dependency and secret scanning, documented MAVLink signing/trust policy. Keep logs sanitized while preserving engineering provenance.

**Maintainability:** internal modules may maintain private numerical states; what must be singular is the physics definition and external contract, not one mutable state object shared between estimate and prediction. Replace large monolithic implementation files incrementally, with compatibility exports and regression tests.

## 9. Module disposition — every current capability has a role

This is a scope decision, not an assertion that every listed package was source-audited in full.

| Current capability | Decision | Required evolution |
|---|---|---|
| `state_manager` | Keep/refactor | Separate typed streams, immutable snapshots and asset-scoped contracts |
| `telemetry_engine` | Keep | Preserve clocks/lineage, record drops, deterministic replay |
| `mavlink_bridge` | Keep | Reuse actual receive/transport capabilities rather than parallel fake-connected sinks |
| `ros2_bridge` | Optional adapter | Test NED/ENU and FRD/FLU transforms; do not require ROS for core |
| `dashboard` | Keep lean | Engineering plots and evidence state before photorealistic visuals |
| `sitl` | Keep | Pin one working PX4/Gazebo configuration and integration test |
| `gazebo_bridge` | Keep selective | Simulator adapter; not the independent validation authority |
| `mission_planner` and sinks | Narrow | Supervised setpoint workflow, real transport and safety gateway |
| `sensor_models` | Keep | Noise calibration, correlations and clock faults; synthetic label mandatory |
| `cad_engine` | Strengthen | As-built component assembly, mesh validation, CG/full tensor, manufacturing outputs |
| `cad_engine.bemt` | Strengthen | Matched polars/data, convergence and installed-system correlation |
| `openfoam_bridge` | Repair then defer expensive scope | Real dimensional/mesh/convergence gates; offline drag/propulsion support |
| `validation_framework` | Make central | Evidence manifests, native loaders, independent truth, holdouts and uncertainty |
| `pinn_engine` | Defer | First beat interpolation/least-squares/GP baselines on measured data; describe actual physics constraints |
| `battery_twin` | Core | Fix validation bugs; pack-specific transient/thermal calibration |
| `battery_twin.nasa_adapter` | Keep narrowly | Authentic cell-level study; do not imply pack validation |
| `structural_twin` | R2 depth track | Full-frame assembly, joint effects and physical static/modal correlation |
| `predictive_maintenance` | Narrow/defer RUL | Condition indicators first; replace unsupported health/RUL certainty |
| `hil_interface` | Verify before claim | Source retrieval did not establish implementation; require actual hardware loop evidence |
| `rl_controller` | Preserve but defer training | Real implementation exists; share physics, fix Gym API/seeding and unused randomization |
| `ai_copilot` | Defer | Optional documentation/search assistant, no flight or engineering authority |
| `swarm_engine` | Archive from main roadmap | Not needed for the central mechanical/aerospace claim |
| `digital_twin_core` | Core rewrite by increments | Correct dynamics, estimator semantics, alignment, identification and uncertainty |
| `perception_bridge` / Glob3R-inspired | Optional separate experiment | Real sensor/depth evidence required; landing safety not inferred from generated point clouds |
| `fprime_bridge` | Optional protocol experiment | Keep “inspired/custom” wording; independent F Prime interoperability is a separate deliverable |

## 10. Flagship demonstration

A six-minute engineering demonstration, using saved data so it is reproducible:

1. Show the physical aircraft/bench and its CAD assembly, mass budget and sensor calibration.
2. Load an unseen measured run; display telemetry, independent reference where available, twin prediction, and uncertainty.
3. Compare nominal, offline-calibrated fixed and safely updated models on the same condition. Explain any remaining error.
4. Show a **safe, known** payload/configuration change. Hold mass fixed to the measured value when identifying propulsion efficiency; do not let two indistinguishable parameters “explain” the same residual.
5. Demonstrate an OOD or sensor-quality condition in replay/SITL: uncertainty rises and adaptation freezes rather than pretending confidence.
6. Open a generated report and show a predicted design trade-off that was physically measured. End with limitations and the exact reproduction command.

The showcase is a measured prediction and design decision, not a spinning 3D drone.

## 11. Delivery plan and resources

### Planning ranges, not promises

- **First 2 weeks:** credibility repair, failing physics tests, data contracts, measured-asset plan.
- **Weeks 3–6:** native real-data ingestion, correct shared physics, propulsion/energy baseline and repeatable reports.
- **Weeks 7–12:** bench correlation or well-bounded external datasets, held-out evaluation and uncertainty, first trade study.
- **Weeks 13–16:** earliest narrowly scoped R1 packaging/review window when existing code, datasets and facilities are reusable; otherwise extend the following phases rather than skip evidence.
- **Following 3–6 months:** R2 structures/CFD/own-aircraft flight and supervised integration, selected by available facilities.
- **Research programme:** typically 6–12+ further months for new datasets, rigorous studies and independent reproduction. Full simultaneous mastery of all original modules is a multi-person programme, not a realistic quick solo upgrade.

For the repair and R1 scope represented by backlog B01–B17, budget approximately 300–700 focused engineering hours, roughly 4–11 months at 15–20 hours/week, plus procurement/lab delays. A 16-week release is only a lower-end fast path with substantial reuse and ready measurements, not the default commitment. R1 excludes full CFD, full-frame FEA, battery aging campaigns, RL training and flight certification. The remaining backlog adds substantial work; do not sum the entire programme into a 16-week promise.

### Budget planning bands

Illustrative INR allowances, not current supplier quotations: software/existing datasets ₹0–10k; useful propulsion/current/voltage/temperature bench access or instrumentation roughly ₹15k–50k; a new reference aircraft and spares roughly ₹25k–70k; advanced thrust/torque/DAQ/RTK/motion-capture access may exceed ₹1 lakh or require a university lab. Existing lab equipment can reduce cost substantially. Reserve 20–30% contingency, and price local parts before committing. Do not buy a GPU before obtaining reliable measurements.

### Engineering allocation

Prioritize physical reasoning, measurement and correlation over UI/AI breadth: approximately half the effort on physics/experiments, a third on data/software/V&V, and the remainder on communication and integration. Exact allocation depends on hardware access.

## 12. Risks and mitigations

| Risk | Consequence | Mitigation / decision gate |
|---|---|---|
| Insufficient hardware access | Cannot prove own-aircraft twin accuracy | Release bounded M1 study first; seek lab access before purchasing |
| Mass/thrust/wind confounding | Attractive but meaningless parameter estimates | Independent measurements, excitation/rank gate and frozen parameters |
| Synthetic/reference leakage | Overstated benchmark accuracy | Immutable source-separated logs; negative provenance tests |
| Over-scoping | Many incomplete claims, no defensible result | Ship R1 before CFD/structures/RL; one thesis |
| Weak ground truth | GPS/estimator agreement mistaken for validation | State reference quality; use suitable RTK/mocap/bench instruments for claims |
| Unsafe testing | Injury, battery fire, vehicle loss | Supervised enclosed/guarded bench, OEM limits, formal pre-test review, approved operations |
| Model extrapolation | False confidence in flight/energy decisions | Domain flag, conservative uncertainty, no autonomous safety authority |
| Research non-novelty or negative results | No publication claim | Strong baselines and prior-art review; publish reproducible negative/benchmark findings |
| Unsupported RUL/composites claims | Misleading engineering conclusions | Condition monitoring first; material-specific evidence and long-term data required |
| Reproducibility/dependency drift | Others cannot verify results | Pinned manifests, containers, licensed data and tagged artifacts |

## 13. Release gates

**G0 — Definition:** one use case, asset/configuration, measurement plan, safety envelope and frozen evaluation protocol.

**G1 — Correctness:** physics/frame/provenance regressions pass; P0 audit issues addressed or explicitly disabled; no real-data scores from fallback inputs.

**G2 — Correlation:** subsystem parameters fitted only on training data; held-out measured propulsion/energy results with uncertainty; failures published.

**G3 — Portfolio release:** clean reproduction, report/demo, one justified design decision, contribution record and precise README. A failed ambitious accuracy target may still yield a valuable honest release, but the target is not silently marked passed.

**G4 — Integration/research:** additional safety approval for command/HITL/flight scope; successful preregistered comparison and ablations before claiming the proposed method improves performance.

## 14. Definition of done

A release is done when its central claim can be reconstructed from requirements → model assumptions → code/configuration → calibrated instruments/data → held-out experiments → uncertainty → decision and limitations. All claimed external integration has an actual peer test; all numerical performance is linked to an artifact; all unsupported capabilities are labelled experimental/planned.

**Resume-worthy:** “I designed, measured, calibrated and verified this specific engineering system, and here is where it works and fails.”

**Potentially influential:** other engineers can use your benchmark or method, reproduce the result, and make a better decision because of it.

## 15. Companion documents

- `02_ARCHITECTURE_AND_ICD.md` — model equations, interfaces, timing, data/parameter contracts and safety.
- `03_VERIFICATION_VALIDATION_PLAN.md` — experiments, baselines, metrics, uncertainty and release evidence.
- `04_IMPLEMENTATION_BACKLOG.md` — concrete repository changes, dependencies, estimates and first 14 days.
- `05_SOURCE_BACKED_REPO_AUDIT.md` — observed strengths, defects, source links and review limitations.
- `06_RESEARCH_AND_PORTFOLIO.md` — research thesis, studies, mechanical/aerospace deliverables and honest resume wording.
- `07_REFERENCES.md` — primary references and limits of their applicability.
- `templates/` — proposed vehicle/data/test/risk templates, not executable finished integrations.
