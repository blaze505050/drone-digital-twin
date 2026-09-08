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
