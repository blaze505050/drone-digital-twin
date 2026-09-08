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
