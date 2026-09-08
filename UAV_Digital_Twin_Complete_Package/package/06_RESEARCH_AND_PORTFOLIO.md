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
