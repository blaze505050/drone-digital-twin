# UAV Digital Twin — Engineering Upgrade Package

**Prepared for the owner of `blaze505050/drone-digital-twin` • 7 September 2026**

## The recommendation in one minute

Your repository has a worthwhile technical foundation. Its next leap should be **correctness and physical evidence, not another dozen modules**.

Make the flagship an asset-specific UAV energy/condition twin: connect as-built mass/CAD, measured propulsion and pack behaviour, real telemetry, bounded calibration and honest uncertainty. Show one prediction that improves a real design decision. That is much stronger for mechanical/aerospace roles than an unsupported “production-ready AI platform” claim.

A revolutionary result cannot be promised. A reproducible benchmark, a genuinely better method, or a well-measured engineering design can become an influential contribution.

## What this package contains

1. **Revised PRD:** purpose, users, scope, release tiers, 30 functional requirements, metrics, risks, resources, maturity and release gates.
2. **Architecture/ICD:** governing equations, frames, clocks, typed data, coupling, safety and proposed package layout.
3. **V&V plan:** 14 test families, six experimental protocols, baselines, holdouts, uncertainty and evidence requirements.
4. **Implementation backlog:** 30 dependency-ordered changes, existing/proposed paths, acceptance tests, effort ranges and first-14-day sequence.
5. **Source-backed audit:** 13 findings with commit-pinned source links and explicit review limits.
6. **Research/portfolio strategy:** falsifiable thesis, ablations, mechanical/aerospace artifacts, demo and truthful resume templates.
7. **Primary references** and six reusable planning templates. Structured requirement and backlog CSVs are generated from the documents.

The numbered Markdown files are the editable source of truth. The PDF is the compiled reading edition; the DOCX is an editable convenience export. The ZIP bundles the documents, templates and structured trackers.

## Fix these before adding features

- Correct NED gravity/thrust and full rigid-body rotational dynamics; share the plant with the RL wrapper.
- Correct magnetic-heading observation, acceleration semantics and time/source alignment.
- Fix battery validation reading the wrong voltage field; enforce real/synthetic/invalid dataset status.
- Replace serialization-only command “dispatch” with real transport before claiming interoperability; keep hardware authority disabled by default.
- Validate one physical propulsion-energy chain and compare against a strong fixed calibrated model on held-out runs.

The audit also identifies OpenFOAM dimensional/reference issues, incomplete frame-FEM correlation, CAD assembly/validity gaps and benchmark truth leakage. Those are actionable findings, not a claim that the entire repository is unusable.

## Scope and honesty

Reviewed baseline: `4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4`. This was a **static source review**, not an executed build/test suite, CFD run, dataset validation or hardware test. No repository files were changed. Proposed commands, schemas and features are labelled as proposals. All numerical targets and effort/budget ranges are planning assumptions, not achieved results or vendor quotations.

## Reading order

- **10 minutes:** this page → PRD executive decision/scope/metrics → audit bottom line.
- **Start implementation:** audit → backlog B01–B08 → architecture conventions → V&V V00/V01.
- **Build the standout result:** PRD R1 → experiments E1/E2/E3/E5/E6 → portfolio strategy.
- **Long-term research:** only after R1, choose one thesis and work through R2/R3 gates.

## Definition of success

A reviewer can follow **requirement → physics → configuration → calibration → independent measurement → uncertainty → useful decision**, and another engineer can reproduce your central result.

That is the standard to build toward.


# Source-Backed Repository Audit

**Reviewed:** public repository `blaze505050/drone-digital-twin` at commit `4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4`. **Review date:** 7 September 2026, user-local.

## 1. Scope and limits

This was a read-only, static source review through public GitHub/raw-file pages. It inspected the README, packaging metadata, selected core/estimator/dynamics/transport code, CAD/BEMT/OpenFOAM/structural/battery/benchmark implementations and representative tests. It was **not** an exhaustive security audit, a repository clone/build, test-suite execution, numerical solver run, dataset experiment, or hardware test. Advertised test counts, runtime/performance and external compatibility were not independently reproduced. No GitHub files were modified.

Findings below distinguish directly visible source behaviour from engineering risks that need a regression test. Only source-reviewed capabilities are described as observed. Failure to retrieve a file is not proof that a capability is absent.

## 2. What is already valuable

- The project is more than a dashboard: there is a forward rigid-body model, error-state MEKF, motor lag, residual monitoring and parameter-update structure.
- CAD contains genuine signed tetrahedral/polyhedral mass integration, including products of inertia. Do **not** discard this and propose “add inertia calculation” as if nothing exists.
- BEMT contains actual radial/induction calculations and Prandtl corrections; battery and structural packages contain useful low-order engineering models.
- State schemas, thread-safe storage, source tags and optional MAVLink receive paths are useful foundations.
- Synthetic-fallback flags already exist in some loaders. Improve their consistency and enforcement instead of claiming the project hides all simulation.
- The RL module is a substantial implementation in `__init__.py`, not an empty module. It contains a Gym-like environment, randomization, reward and curriculum code.
- README already says F Prime/Glob3R **inspired**. Preserve that distinction. It does not prove official interoperability or use of those external implementations.

The current weakness is the gap between useful prototype code and the README's production/validation claims.

## 3. High-priority source findings

### A01 — NED thrust/gravity inconsistency and incomplete rotational dynamics

**Observed:** `DynamicTwinModel.step` labels state as NED but adds gravity `[0,0,-g]`; at identity attitude its thrust vector has positive z. With zero thrust, a free-falling model accelerates towards negative NED z, opposite the documented convention. The RL `_integrate` repeats the same pattern. Angular acceleration divides moments by diagonal inertia without the rigid-body `omega x (I omega)` term. Products of inertia from CAD are not carried through this path.

**Change:** implement the frame-defined shared equations in the ICD; use full tensor or a rigorously justified principal-axis transform; configure rotor positions/axes/spin signs rather than hardcoded mixer patterns. Add free-fall, hover, tilt, one-rotor torque and torque-free-body tests before any controller tuning.

**Important accuracy note:** the inspected core source refreshes quaternion locals inside its substep loop. A stale-substep-quaternion defect is **not** an established finding.

Sources: [twin_model.py](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/digital_twin_core/twin_model.py), [RL implementation](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/rl_controller/__init__.py).

### A02 — RL API and randomization do not match the advertised contract

**Observed:** `DroneGymEnv` does not inherit `gymnasium.Env` and the inspected class has no actual `action_space`/`observation_space` objects despite examples calling `env.action_space.sample()`. `reset(seed)` seeds global NumPy but not the randomizer's separate generator. The integrator samples but does not apply several advertised parameters (per-motor scales, latency, wind, drag scale). Maximum thrust is recomputed from randomized vehicle mass, masking a real payload/thrust-margin effect.

**Change:** retain the interface idea, wrap the shared plant, implement proper spaces/reset seeding and a real Gym environment checker test. Apply each randomized parameter to a measured physical mechanism or remove it from claims. Defer expensive policy training until physics and baseline control are verified. Source: RL implementation above.

### A03 — Magnetometer correction is not a valid independent heading observation

**Observed:** `update_magnetometer` rotates the measured body vector using the current estimated attitude, then subtracts estimated yaw. For a level true-north measurement and an erroneous yaw estimate, the resulting residual is zero rather than correcting that error. This is a concrete testable counterexample, not an assertion that every magnetic input always gives zero residual. The estimator emits specific-force components as `ax/ay/az`; the plant emits inertial acceleration in similarly named fields.

**Change:** use a calibrated navigation magnetic-field reference with the correct body observation model/Jacobian; gate disturbances. Name acceleration quantities explicitly, propagate quality and test covariance/error-state reset conventions.

Source: [estimator.py](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/digital_twin_core/estimator.py).

### A04 — Time/source alignment and online identification are not yet trustworthy

**Observed:** core outputs assign current wall/monotonic time with simulation time zero; residuals compare latest states without causal alignment. Recalibration records current time rather than a complete measurement/control interval. It fits vertical acceleration and imposes a mass/thrust relationship without independent identifiability. Body specific force from the estimator is treated as though it were NED acceleration. Broad exception handling and in-place changes provide little promotion/rollback evidence.

**Change:** typed streams and preserved clocks; distinguish command intent from applied actuation; measured mass prior; fit only identifiable parameters; offline fitting first; bounded candidate/active versions and holdout/rollback. Separate internal estimator/plant states are legitimate—the issue is their external semantics and evidence, not merely having two state objects.

Sources: [recalibrator.py](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/digital_twin_core/recalibrator.py), [residual_monitor.py](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/digital_twin_core/residual_monitor.py), [store.py](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/state_manager/store.py).

### A05 — Command serialization is not actual MAVLink dispatch

**Observed:** `MAVLinkCommandSink.send` stores an encoded payload and reports success; `is_connected()` returns true. That does not send a complete MAVLink message to PX4. A separate `MAVLinkSource` has genuine optional pymavlink I/O, so transport work can reuse that foundation.

**Change:** transport injection, actual peer integration and a safety gateway. Use acknowledgements only for message classes that define them; streamed setpoints need rate/mode/state-response validation. Default examples remain read-only. Hardware-in-the-loop implementation was not established by this review; source retrieval of the claimed path was unsuccessful, so label it **unverified**, not “definitely absent.”

Sources: [sinks.py](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/mission_planner/sinks.py), [mavlink_source.py](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/mavlink_bridge/mavlink_source.py).

### A06 — F Prime-inspired adapter mixes commands with measured state

**Observed in inspected path:** custom sync/CRC/channel framing and loopback are implemented; an external F Prime peer was not demonstrated. `FPrimeTelemetrySource.poll` turns received command targets into state updates, conflating intent and telemetry. The inspected bridge's UDP path does not establish a complete autonomous bidirectional receive/send integration.

**Change:** route commands to a command bus, specify custom protocol/version, demonstrate a real external endpoint before compatibility claims. Keep this optional, not central to mechanical/aerospace validation.

Sources: [F Prime source](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/fprime_bridge/source.py), [bridge.py](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/fprime_bridge/bridge.py).

### A07 — NASA battery voltage validation can score a constant fallback

**Observed:** adapter validation reads `state.terminal_voltage`, while the battery state field is `v_terminal`; the fallback is 3.7 V. That makes the stated RMSE unrelated to the intended ECM output in this path. Loader exception/fallback flow can also lose the true synthetic status. Authentic NASA cell aging is not proof of pack performance under UAV loads.

**Change:** fix field access and add a test that fails if model voltage is ignored; preserve REAL/SYNTHETIC/INVALID status through every parse path; calibrate actual pack topology, transient behaviour and temperatures.

Sources: [nasa_adapter.py](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/battery_twin/nasa_adapter.py), [battery model](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/battery_twin/__init__.py).

### A08 — OpenFOAM case generation has dimensional/reference defects

**Observed:** `_write_field` uses the same dimensions for incompatible fields; `U`, `omega` and `nut` cannot share pressure dimensions. Force-coefficient references such as speed/area/density are hardcoded rather than consistently derived from the requested case. Presence of a force output file is treated as convergence without sufficient solver/stationarity evidence. Empty database training has a synthetic fallback.

**Change:** field-specific dimension table; configured reference quantities; version-pinned solver; geometry/patch/checkMesh gates; convergence and averaging criteria; mesh/time-step studies; data-quality filtering before surrogate training. Dictionary/file-existence tests alone are inadequate.

Source: [OpenFOAM implementation](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/openfoam_bridge/__init__.py).

### A09 — CAD integration needs geometry validity and complete assembly handling

**Observed:** genuine polyhedral integration exists, but shell/degenerate paths use approximations, absolute-valued inertia can hide invalid geometry, and CAD-to-twin drops CG/products. The generated frame geometry is not the entire installed aircraft.

**Inferred risk requiring regression:** signed volume/centroid treatment can mishandle reversed mesh orientation. Test translated, rotated and reversed solids before calling results exact for arbitrary meshes.

**Change:** normalize/check orientation/closure, distinguish solids vs shells, expose invalid cases rather than minimum invented mass, aggregate all components, retain full tensor/reference point. Source: [CAD implementation](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/cad_engine/__init__.py).

### A10 — BEMT is a baseline needing matched data and solver diagnostics

**Observed:** sectional polars are generic; induction/negative result clipping and limited convergence diagnostics can conceal unsupported operating points. UIUC comparison contains eight in-code points with a fallback flag rather than a complete traceable external-data campaign. The documented NACA benchmark is not implemented in that inspected benchmark file; this is not a whole-repo absence claim.

**Change:** authentic propeller files and condition metadata; measured interpolation baseline; Re-dependent sectional polars where available; convergence/failure status; correct hover/forward-flight formulations; thrust, power, efficiency and uncertainty comparisons. Do not claim rotor validation from an airfoil-only comparison.

Sources: [bemt.py](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/cad_engine/bemt.py), [aerodynamic_benchmarks.py](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/validation_framework/aerodynamic_benchmarks.py).

### A11 — Structural frame and fatigue claims exceed the implemented analysis

**Observed:** element stiffness/mass formulations exist, but frame results use one-arm cantilever expressions and hardcoded higher-mode ratios rather than assembling the complete frame. Fatigue uses simplified cycle counting and generic material-life assumptions; mode shapes/real-life validation are not demonstrated.

**Change:** preserve cantilever model as a labelled low-order baseline; assemble joints/body/arms/motor masses; correlate actual static/modal tests. Use verified rainflow and material-specific data before damage/RUL claims. Carbon composites require appropriate material/layup and failure assumptions, not an unexplained isotropic constant.

Source: [structural implementation](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/structural_twin/__init__.py).

### A12 — Named trajectory/flight benchmarks need native sensors and independent truth

**Observed:** benchmark fallbacks generate trajectories; Zurich's inspected loader ignores the supplied path. Evaluation truncates arrays rather than establishing complete temporal alignment. The inspected flight CSV path uses position as reference and generates noisy GPS from it rather than parsing an independently measured GPS channel. Synthetic tests can be useful, but cannot establish measured flight accuracy.

**Change:** native loaders, actual sensor/truth separation, documented transforms/clocks, holdouts, appropriate alignment and uncertainty. Missing files are NOT_SCORED. EuRoC is a visual-inertial benchmark; do not fabricate GPS/barometer and report standard EuRoC performance without disclosing the altered task.

Sources: [benchmarks.py](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/validation_framework/benchmarks.py), [flight_log_validator.py](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/sdk/drone_sdk/validation_framework/flight_log_validator.py).

### A13 — Production/readiness metadata is ahead of demonstrated evidence

**Observed:** README includes “PRODUCTION READY,” static test/module counts and strong transfer/validation language. `pyproject.toml` declares Production/Stable, while the inspected paths above need substantial repair. The `all` extra is not a literal union of all optional groups. These are fixable presentation/install-contract issues, not proof that all code is unusable.

**Change:** Research Prototype/Beta as justified; generated CI badges; implemented/tested/calibrated/validated matrix; tested Python/dependency matrix; verified install commands; optional-dependency contract. Remove unsupported zero-shot transfer and airworthiness implications. Do not claim tests failed or passed without running them.

Sources: [README](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/README.md), [pyproject.toml](https://github.com/blaze505050/drone-digital-twin/blob/4d5b8d3fbe84c0bb2d97fb93bde45cf7f1abf9c4/pyproject.toml).

## 4. Overall verdict

**Promising broad research prototype; not yet demonstrated production-grade or experimentally validated as a complete aircraft twin.** The best investment is to correct the plant/measurement contracts, make benchmark provenance strict, validate one actual propulsion-energy chain, and publish one design prediction confirmed by measurement. The breadth becomes an advantage only after that narrow chain is trustworthy.


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


# Architecture and Interface Control Document

**Proposed design • 7 September 2026 • supports PRD 3.0**

This is a target architecture, not a claim that the repository currently implements it. Keep the package name `drone_sdk` and compatibility imports during migration. Prefer a modular Python library and a small CLI over distributed services until measurements demonstrate a need.

## 1. System architecture

```text
Physical asset / authentic dataset / PX4 SITL
    |
    v
Read-only adapters ---> immutable raw log + data manifest
    |
Clock mapping + calibration + frame conversion + quality checks
    |
    +--> SensorMeasurement --> estimator --> StateEstimate
    |
    +--> AppliedActuation -----------------------+
    |                                           |
    +--> ReferenceTruth --> evaluation ONLY      v
                                     shared physics kernel
VehicleConfig + active ParameterSet ----------> TwinPrediction
                                                   |
Aligned estimate/prediction ---> residual + validity + uncertainty
                                                   |
                                     engineering cockpit/report
                                                   |
                             candidate parameter fit (offline first)
                                                   |
                         excitation / holdout / safety / bounds gates
                                                   |
                               versioned promotion or rejection

Optional and separate:
Human-approved CommandIntent --> safety gateway --> PX4 transport
```

Truth is inaccessible to the estimator and fitting process except within explicitly declared training/validation jobs. A prediction must identify its initialization time and all inputs used. Raw telemetry and command intents cannot be written into the same state fields without type separation.

Internal estimator and plant states remain separate. “Single source of truth” means a single versioned contract and one authoritative stream for each meaning, not overwriting the plant prediction with the latest measurement every frame.

## 2. Model hierarchy and coupling

### Fidelity levels

- **L0:** analytic equations and intentionally simple synthetic systems for numerical verification.
- **L1:** real-time rigid body, measured actuator map, low-order drag, ECM/thermal model; main operational twin.
- **L2:** BEMT, full-frame beam/FE model, offline CFD and system identification; supply validated parameters/maps to L1.
- **L3:** expensive high-fidelity analyses and specialised research; only when their additional accuracy changes a decision.

Do not call a low-order model “high fidelity” merely because its coefficients came from CFD. Record which quantities are correlated and where.

### Coupling contracts

1. CAD/component assembly supplies mass, CG, full inertia tensor, rotor locations/axes and geometry hashes.
2. Motor/ESC/propeller model receives applied command, bus voltage, density and local inflow; returns RPM, thrust, reaction torque, electrical current/power and confidence.
3. Battery ECM uses current and temperature; returns terminal voltage, SOC, thermal state and limits. Avoid an instantaneous algebraic loop: use a documented predictor/corrector or converged fixed-point coupling with iteration limits.
4. Rigid-body dynamics uses force/moment sums, air-relative velocity and inertia; returns position, orientation and rates.
5. Structural model uses time-dependent rotor/body loads. Its outputs initially support offline deflection, vibration-risk and design decisions. Feed deformation back into aerodynamics only after separately validating the coupling.
6. Condition monitoring observes residuals and vibration/electrical features. It does not silently change flight-control settings or invent component life.

Each exchanged quantity carries SI units, coordinate frame, reference point and validity domain. Never silently substitute an unconverged CFD coefficient or synthetic surrogate training row.

## 3. Coordinate and physical conventions

Freeze the following convention in tests and documentation:

- Local navigation frame **NED:** x north, y east, z down; gravity `g_n = [0, 0, +g]`.
- Body frame **FRD:** x forward, y right, z down. Nominal rotor thrust acts along negative body z; actual rotor axes are configuration vectors.
- Quaternion `q_nb = [w,x,y,z]`, Hamilton product, active rotation from body to navigation; `v_n = R_nb(q) v_b`.
- Body angular velocity `omega_b` in rad/s. Positions in metres; RPM converted explicitly to rad/s or revolutions/s before equations.
- Specific force `f_b` is accelerometer output after bias removal, not inertial acceleration: `a_n = R_nb f_b + g_n`.
- `reference_origin` and geodetic datum are explicit. Local NED z is not automatically height above terrain. Compute AGL from a defined terrain/range reference, not merely `-z` outside a flat home-origin test.
- Magnetic reference is a calibrated navigation-frame field vector (including inclination/declination convention); predict the body observation with `m_b_pred = R_nb^T m_n`.

### Rigid-body equations

```text
p_dot_n = v_n
m v_dot_n = m g_n + R_nb sum(F_b) + F_aero_n
I_b omega_dot_b = tau_b - omega_b x (I_b omega_b)
q_dot_nb = 0.5 q_nb ⊗ [0, omega_b]
tau_b = sum((r_rotor_b - r_CG_b) x F_rotor_b + Q_reaction_b) + other moments
```

Rotor gyroscopic moments require a consistent rotor angular-momentum model and reaction-torque signs. Do not double count them. If inertia varies with payload changes, update it through a versioned configuration; use variable-inertia terms only if a physically moving-mass model is actually implemented.

Analytic invariants: zero thrust at level gives positive NED vertical acceleration; steady hover has total upward thrust `mg`; a positive full-body force must rotate consistently; torque-free non-spherical bodies follow Euler dynamics; quaternions remain normalized within tolerance; power and energy have consistent signs/units.

### CAD aggregation

For each component with mass `m_i`, local CG and inertia `I_i`, transform into the body frame. Let displacement from assembly CG be `d_i`:

```text
m = sum(m_i)
r_CG = sum(m_i r_i) / m
I_CG = sum(R_i I_i R_i^T + m_i ((d_i·d_i) Identity - d_i d_i^T))
```

Require a symmetric positive-definite tensor for a nondegenerate 3-D assembly, physical principal moments, orientation checks, and explicitly handled thin-shell/overlap assumptions. Near-zero off-diagonal terms use absolute rather than relative error in tests.

## 4. Message contracts

### Common envelope

Every record contains `schema_version`, `asset_id`, `configuration_hash`, `run_id`, `source_id`, `sequence`, `acquired_time`, `received_time`, `clock_id`, optional `sim_time`, quality flags and a lineage reference. Clock mapping stores offset/drift estimates and uncertainty. A device monotonic timestamp is meaningful only within its clock domain.

### Typed payloads

| Contract | Required payload | Must not contain |
|---|---|---|
| `SensorMeasurement` | sensor ID, raw/calibrated values, units, frame, covariance/noise model, calibration ID | Invented truth-derived GPS labelled as measured |
| `StateEstimate` | pose/velocity/biases as available, covariance and convention, estimator ID, update horizon | Claim of independent ground truth |
| `ReferenceTruth` | instrument/reference pose or quantity, calibration and uncertainty, covered interval | Automatic routing into live estimation |
| `CommandIntent` | target vehicle, mode, frame, target values, creation/expiry, requesting authority | Fake connected/success flag |
| `AppliedActuation` | actual output/PWM/RPM as available, scaling/calibration, application interval and command linkage | Assumption that a position setpoint equals rotor thrust |
| `TwinPrediction` | forecast issue time, target time/horizon, parameter/model IDs, inputs, mean/interval/domain | Unlabelled use of future measured inputs |
| `ResidualReport` | aligned record IDs, quantities/units, covariance assumptions, validity and threshold | Definitive cause or airworthiness statement without evidence |
| `ParameterCandidate` | previous/new values, units/bounds/covariance, dataset, method, excitation, holdout result | Direct in-place mutation of active flight parameters |

Unknown covariance is `unknown`, not zero. “Not observed” differs from a physically zero sensor value. A missing rotor RPM should never become 0 RPM unless an actual stopped-rotor measurement exists.

## 5. Timing and replay

- Acquire at native sensor rate; do not manufacture a 100 Hz stream from slow measurements and pretend it contains more information.
- Preserve source timestamps; never replace acquisition time with receive time. Save both.
- Deterministic replay advances a simulation clock from event timestamps. Wall time is only for display/profiling.
- Explicitly handle duplicated, out-of-order, stale, dropped and reset timestamps. Bound buffers and log every dropped/rejected item.
- Define interpolation separately for positions, orientations, and measurements. Use valid quaternion interpolation; do not interpolate across a reset or unobserved long gap.
- Residuals compare matching asset/configuration, time and quantity. Interpolation uncertainty or alignment rejection is recorded.
- Use a declared prediction horizon and initialization/reseeding schedule. Repeatedly resetting to measurements can make tracking look perfect without proving forecasting.

Suggested R2 rates: IMU native ~100–400 Hz depending on hardware; lightweight twin 100 Hz target; energy/health 10–20 Hz; UI 10–20 Hz; fit/report asynchronous. These are design targets, not measured performance. Latency acceptance is based on end-to-end acquisition-to-use age, not only computation time.

## 6. Parameter lifecycle and identifiability

Start with measured constants: mass from scale, geometry from as-built/CAD, battery topology, calibrated voltage/current. Fit motor maps and battery parameters on designed bench profiles. Avoid simultaneous mass/thrust fitting from one hover trace: vertical acceleration often identifies their ratio, not both separately.

Parameter lifecycle:

```text
UNINITIALIZED -> PRIOR -> OFFLINE_FIT -> CANDIDATE
CANDIDATE -> ACCEPTED (all gates pass) -> ACTIVE
CANDIDATE -> REJECTED (reason recorded)
ACTIVE -> FROZEN (sensor/domain/excitation failure)
ACTIVE -> ROLLED_BACK (degradation or invalid promotion)
```

Promotion gates: physical bounds; finite values; regressor rank/conditioning or equivalent information criterion; sufficient independent excitation; good sensor/time quality; training window definition; disjoint validation window; no unacceptable degradation in held-out prediction or interval calibration; bounded change rate. Active and candidate models run separately. Real-asset R1 uses human-reviewed offline promotion; autonomous parameter updating first runs in replay/SITL/shadow mode.

Uncertainty combines measured-input uncertainty, parameter uncertainty and model discrepancy. Explain correlations: the estimator and twin may share sensors or initialization, so summing their covariances as independent is generally unjustified. Use a stated joint approximation, empirical residual calibration, ensemble method or conservative bound and test coverage. NIS assesses innovation consistency under its assumptions; NEES requires suitable state truth and matching error-state definitions.

## 7. Safety and external interfaces

**Default:** telemetry/read-only and offline analysis. No script in quickstart arms an aircraft.

For optional control, inject an actual MAVLink transport and a safety gateway. A serializer is not a sender. PX4 setpoint streaming (`SET_POSITION_TARGET_LOCAL_NED`, etc.) is not generally acknowledged like a `COMMAND_LONG`; validate expected message semantics. Use command ACKs where the protocol defines them, and mode/heartbeat/position response and timeout evidence for streamed setpoints. Document supported PX4 version, dialect, target IDs, message rates, frames, timestamps, type masks and offboard-entry conditions.

Gateway checks: explicit operator authorization; correct target and mode; valid bounded finite values; monotonic freshness; rate limits; geofence/envelope; connected peer; safe startup and link-loss behaviour. Flight controller failsafes remain authoritative. **Never implement generic “disarm on timeout” in flight.** Bench energy cutoffs and in-flight recovery are different safety cases.

F Prime-inspired framing remains a project-specific protocol unless an external F Prime application, dictionary and topology demonstrate compatibility. ROS 2 encoders alone do not prove running-node interoperability. HITL requires real controller evidence, wiring/interface specification, rates, closed sensor/actuator loop and failure handling.

## 8. Proposed repository layout

```text
sdk/drone_sdk/
  contracts/         # schemas, units, frames, clocks and evidence status
  physics/           # shared rigid body, actuator and environmental kernels
  configuration/     # assembly mass, asset and parameter manifests
  digital_twin_core/ # orchestration, estimate/predict separation, validity
  identification/    # offline fits, excitation tests, candidate promotion
  uncertainty/       # sensitivity, sampling, interval calibration
  validation_framework/ # scoring, native loaders, report generation
  adapters/          # compatibility wrappers around MAVLink/ROS/PX4 sources
  safety/            # command eligibility and audit trail; optional control
  ... existing domain packages, split incrementally ...
configs/vehicles/    # reviewed asset configurations, no secrets
data/manifests/     # checksums/licenses/splits; large data elsewhere
experiments/        # versioned protocols and reproducible run definitions
reports/            # small generated summaries; release data artifacts external
hardware/           # CAD, drawings, BOM, calibration and test-fixture docs
tests/verification/
tests/integration/
tests/validation/
docs/claims.md      # each claim maps to evidence and limitations
```

Do not move everything at once. Introduce contracts and test adapters, migrate one vertical slice, then maintain deprecated re-exports for an announced compatibility period.

## 9. Operational commands and artifact contract

Proposed CLI interface, **not currently available commands**:

```text
drone-twin data validate --manifest <manifest>
drone-twin calibrate --experiment <training-spec>
drone-twin replay --run <run-id> --parameters <version>
drone-twin validate --suite <suite> --split test
drone-twin report --results <artifact-directory>
```

Each run emits a run manifest, structured metrics, figures, warnings/failures, parameter snapshot, model/configuration hashes and human-readable report. Validation commands return a nonzero exit code for a required failed gate; missing real data is NOT_SCORED, not PASS. Publication never depends on a hidden interactive notebook state.


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


# Research Strategy and Mechanical/Aerospace Portfolio

**Proposed strategy • 7 September 2026**

## 1. What would actually make this exceptional?

A famous project is not a product requirement. Influence comes when others can trust, reproduce and use something you made. The most promising route here is an **open, experimentally grounded benchmark plus one well-tested method**, not a claim to replace every aerospace simulation platform.

PX4/Gazebo already provide mature simulation/integration infrastructure; JSBSim is established flight-dynamics prior art. Digital-twin research already spans multiphysics, model updating, uncertainty and health management. Using those ideas is valuable engineering, but not automatically original research.

### Recommended central question

**Can observability- and uncertainty-gated model updating improve small-UAV electrical-energy prediction after a known configuration/condition change, without falsely adapting to sensor error or unsupported operating conditions?**

This connects aerodynamics, propulsion, dynamics, battery physics, experimental design and statistical reasoning. The novelty would be in a specific demonstrably better method or unusually useful dataset/protocol—not the words “AI digital twin.” A focused prior-art review is required before novelty or publication claims.

### Proposed product differentiator

An engineer can trace every prediction from **as-built component configuration → measured propulsion/pack parameters → aligned physical run → bounded update → uncertainty-aware energy decision**. The system shows when it does not know, and distinguishes a weak sensor from a changing vehicle instead of forcing all errors into one “health score.”

## 2. Research hypotheses and rejection criteria

### H1 — useful update, not just better curve fitting

The gated updated model reduces held-out energy/power error compared with a strong offline-calibrated fixed model after a safely measured configuration change.

- Baselines: nominal physics; offline-calibrated fixed physics; conventional least-squares/RLS update; proposed gated update; simple data-driven regression if the available dataset supports a fair comparison.
- Same training/tuning/test access, initialization, mission information and computational reporting for all methods.
- Target: ≥20% relative reduction in preregistered error with a paired confidence interval supporting improvement. This is a proposed goal, not a known achievable result.
- Reject/limit claim if gains disappear against the calibrated baseline, only occur on training trajectories, rely on future inputs, or are too uncertain to distinguish from no improvement.

### H2 — trustworthy refusal

The proposed method refuses updates when mass/thrust are not separately identifiable, data are stale/invalid, or the operating point is out of domain, and therefore suffers less degradation than ungated adaptation.

- Bench/replay/synthetic conditions: insufficient excitation, clock delay, sensor bias/dropout and known parameter mismatch. Physical faults are not intentionally induced in flight.
- Outcomes: rejection quality, forecast degradation, false alarms, detection delay and interval calibration.
- Ablations: remove excitation gate; remove temporal alignment; remove holdout/promotion check; remove domain/uncertainty gate.
- Reject/limit claim if refusal merely stops all updates or makes predictions uninformatively broad.

### H3 — design value

A prediction-driven change in propeller/payload/frame configuration produces a measurable improvement in a specified objective without violating constraints.

- Baseline: original configuration at matched conditions.
- Objective: electrical energy/endurance, mass-normalized stiffness, or resonance separation—choose one primary quantity.
- Constraints: actual thrust margin, current/temperature limits, structural limits and safe approved operating envelope.
- Publish the prediction before the test. A negative result with an explained model limitation is a valid engineering result.

## 3. Alternative thesis if structural facilities are stronger

If you have good CAD/FEA/manufacturing/modal-test access but weak flight/energy instrumentation, choose **an experimentally correlated frame digital thread and uncertainty-aware lightweight redesign** instead.

Model the complete as-built frame including motor masses and joint compliance; correlate static deflection and mode frequencies; examine rotor excitation frequencies and mode separation; optimize arm section/length/joint design under mass, stiffness and manufacturing constraints; manufacture one variant; verify prediction. Keep flight dynamics and battery as supporting context rather than weakly implemented centrepieces.

A deeper, measured mechanical thesis is better than a shallow energy thesis chosen because it sounds fashionable. Do not attempt both thesis tracks at full research depth in the first solo release.

## 4. Mechanical engineering evidence package

Required or highly valuable artifacts:

- Native CAD and neutral STEP assembly with version and material/component provenance.
- Engineering drawings for one designed component or fixture: dimensions, tolerances, datums and sensible GD&T only where function requires it.
- BOM, mass budget, measured CG and full inertia calculation with uncertainty.
- Load cases, free-body diagrams, hand calculations and comparison to numerical model.
- Material/layup and joint assumptions; mesh/element/constraint sensitivity where FEA is used.
- Static and modal test setup, calibration, repeated measurements and correlation plots.
- DFM/DFA and cost discussion: machining/printing/composite process, fasteners, access, repairability, tolerances and mass trade-offs.
- One manufactured or physically configured revision, predicted vs measured result, and reasons for discrepancies.

Do not substitute colourful stress contours for load-path reasoning or validation. A well-designed, measured arm/fixture can demonstrate more engineering skill than an unverified full-aircraft rendering.

## 5. Aerospace engineering evidence package

- Frame/sign-convention derivation, 6-DOF equations, rotor mapping and aerodynamic assumptions.
- Mass/inertia measurement or credible estimation; thrust-to-weight, disk loading, power loading and endurance calculations with conditions.
- Propulsion coefficient definitions, RPM/advance-ratio/Reynolds applicability and matched data.
- Flight/control envelope and handling of actuator lag, saturation, voltage sag, wind and sensor error.
- PX4 baseline integration, estimator residual/consistency reasoning and appropriate trajectory reference.
- System-identification experiment design with observable parameters and delay compensation.
- Test cards, instrumentation, uncertainty budgets, pass/fail criteria, risk review and configuration management.
- Mission/duty-cycle energy forecasts with physical assumptions and withheld-test results.

Avoid claiming certification, airworthiness, damage tolerance, aeroelastic clearance or operational safety from a student/independent test campaign. Explain what additional evidence would be required.

## 6. Resume and interview positioning

### Recommended project title

**Experimentally Validated UAV Energy Digital Twin — CAD, Propulsion, Battery and Flight-Log Correlation**

Use “experimentally validated” only after the stated experiments exist. Until then:

**UAV Multiphysics Digital-Twin Research Prototype — Verification and Calibration Framework**

### Resume bullet templates — fill only after measurement

- “Developed an asset-specific UAV dynamics and energy twin integrating measured mass/inertia, propulsion maps and a pack ECM; achieved **[measured error]** on **[N] independent held-out [profiles/flights]** across **[conditions]**.”
- “Designed and instrumented **[test rig/component]**, calibrated **[sensors/models]**, and correlated **[thrust/power/deflection/modes]** within **[measured tolerance]**, reporting uncertainty and reproducible test data.”
- “Implemented excitation-gated parameter updating and fault-aware rollback; reduced **[metric]** by **[measured improvement with interval]** versus an offline-calibrated fixed model on **[held-out conditions]**.”
- “Redesigned **[component/configuration]** using CAD, analytical/FE models and measured constraints; verified **[mass/energy/stiffness result]** on a physical revision.”

Do not put proposed PRD targets, synthetic-only results or unsupported “NASA integration” claims into achieved resume bullets. If no physical tests were done, use “verified in simulation” or “correlated to [named external dataset]” precisely.

### Interview questions you should be able to answer

1. What exactly is the physical twin, and what is only simulated?
2. Which quantity does your model predict that an ordinary logger cannot?
3. Derive the thrust/gravity signs and explain your quaternion convention.
4. How did you obtain the inertia and propulsion map?
5. What is identifiable from your experiment, and what is not?
6. What ground truth is genuinely independent?
7. Where is uncertainty from measurement vs model inadequacy?
8. Which baseline almost beat your method, and why?
9. What failed, what changed, and which decision improved?
10. Which work is your original contribution, which is adapted open source, and which tools assisted implementation?

Keep a contribution log and credit upstream code/data, collaborators and assistance accurately. Deep understanding is more persuasive than claiming every component was invented from scratch.

## 7. README and portfolio layout

Recommended first screen:

1. One-sentence purpose and explicit maturity.
2. Photograph/diagram of the actual asset or clearly labelled demonstrator.
3. One central measured result with uncertainty and link to full report.
4. Three engineering contributions, not 28 badges.
5. Reproduction quickstart and hardware/data requirements.
6. Architecture/physics summary.
7. Evidence table: implemented, verified, correlated, unverified/planned.
8. Limitations, safety, licensing, citation and personal contribution.

Suggested opening, valid before experiments:

> A research platform for asset-linked UAV dynamics and energy modelling. It combines a verified rigid-body kernel, configuration-aware propulsion/battery models and traceable real-data evaluation. Current capability and validation status are reported per subsystem; the project is not flight-certified or production-qualified.

Once measurements exist, replace generic promise with the actual bounded result.

### Reviewer-friendly portfolio assets

A 90-second overview; six-minute technical demo; one-page engineering brief; full V&V report; architecture diagram; CAD/drawings; raw-data manifest; two or three clean notebooks or CLI experiments; tagged release; concise failure/limitations section. The report should be understandable without watching the video.

## 8. What to publish, and in what order

1. **Engineering release:** corrected code, analytic verification, honest limitations and reproduction.
2. **Measured benchmark release:** asset/configuration, sensor calibration, matched raw/processed data and baseline results. Check dataset permissions, location privacy and upstream licences.
3. **Technical report/preprint:** falsifiable question, literature/prior art, method, experimental design, baselines, results, ablations and limitations. Publication venue only after matching actual contribution and scope.
4. **External reproduction:** ask a lab/student to recreate one figure or evaluate another asset. Record replication success and failure.
5. **Useful adoption:** docs/tutorials, issues and contributions that help others run real experiments—not vanity star targets.

Archive a stable release with persistent identifiers where appropriate; do not invent a DOI or imply peer review. A careful benchmark can be a lasting contribution even if the proposed advanced method does not win.

## 9. The standard to aim for

Not: “I built the greatest digital twin ever.”

Instead: **“I can show exactly what my model predicts, why the equations are correct, what the physical tests demonstrate, where it fails, and how another engineer can reproduce the result.”**

That is the foundation of an outstanding mechanical/aerospace project—and the credible route toward something genuinely influential.


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
