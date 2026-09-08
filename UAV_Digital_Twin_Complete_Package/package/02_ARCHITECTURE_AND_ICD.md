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
