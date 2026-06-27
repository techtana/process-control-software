# Run-to-Run Process Control Software — Design Requirements Specification

**Status:** Draft for review

**Scope:** Identification, experiment planning, MHE/MPC simulation, controller optimization, diagnostics, and broad predictive regression for a closed-loop run-to-run (R2R) process-control system.

**Audience:** Process/control engineers and scientists implementing and validating the toolchain.

---

## 1. Purpose and Philosophy

This document specifies a cohesive software system for building, simulating, tuning, and diagnosing run-to-run process controllers from imperfect industrial data. It binds six functional components onto a single conceptual core so that they share data contracts, assumptions, and diagnostics rather than re-deriving them independently.

### 1.1 The unifying idea: variance accounting against a noise floor

Every estimation problem in this system — identifying the feedback (FB) process model, identifying the feedforward (FF) model, fitting the broad all-data predictor, tuning the estimator and controller weights, and deciding whether the controller is "done" — is the same question in different clothing:

> *How many directions of variation in the data can be trusted as signal, given the excitation actually present, and where is the floor below which variation is indistinguishable from noise?*

This is the control-theory restatement of the random-matrix-theory result that motivated the project. The Marchenko–Pastur edge is the variance level below which an eigen-direction of a sample covariance is indistinguishable from sampling noise; the minimum-variance / Harris benchmark is the variance level below which a controller cannot push the output regardless of tuning. Both are *noise floors*. The system computes them analytically where the assumptions hold and empirically (permutation / cross-validation) where they do not, and uses them consistently to (a) gate which model terms to trust, (b) decide whether more data or excitation is needed, and (c) declare when residual variance is irreducible rather than a controller defect.

### 1.2 The central obstacle: closed-loop confounding and excitation deficiency

Passive inline operating data reflects the controller's corrections, not the open-loop process. Because feedback makes the control knobs `u` a function of estimated state, the knob and disturbance regressors become collinear and the data covariance is rank-deficient in exactly the directions one needs to identify. This is structural, not incidental: no quantity of passive closed-loop data resolves it. The system must therefore (a) *quantify* the confounding (so the limitation is visible and stakeholder-justifiable), (b) *exploit* whatever accidental excitation exists (manual overrides, control-off episodes, DOE), and (c) *request* deliberate excitation when the gap cannot be closed otherwise. Components 1, 2, and 6 are bound together by this obstacle.

### 1.3 Architectural relationship between the two estimators (Components 1 and 6)

The FB-model identifier (Component 1) and the all-data regression machine (Component 6) are **two configurations of one estimation engine**, not two unrelated tools. They share the data model, the data-quality validation layer, the excitation/confounding analysis, the noise-floor estimator, the counterfactual-reconstruction service, and the validation framework. Component 1 is the structured, control-oriented MIMO gain-plus-dynamics estimate consumed by the controller; Component 6 is the broad predictor across all available sensors whose purpose includes discovering feedforward opportunities. Keeping them unified is a hard requirement (§REG-01) because the confounding and noise-floor logic is identical for both.

---

## 2. System Context and Glossary

| Term | Meaning in this system |

|------|------------------------|

| Knob `u` | Adjustable control input (≈10 knobs). Two columns exist per knob: **recommended** (what the existing controller asked for) and **used** (what was physically applied). |

| Post-processing measurement `y` | The controlled output, measured after processing (≈10 outputs). Sparse, asynchronous metrology. |

| FB model `M` | Process gain (and dynamics) mapping knob deltas to output deltas: the dynamic model identified for MPC. Shape contract `n_out × n_knob`. |

| FF model | Mapping from pre-processing measurements / tool sensors / process configuration to output, used for proactive compensation of measurable disturbances. |

| `u0` / `initial_knob` | The open-loop baseline knob setting; the control contribution is `M @ (u_used − u0)`. The counterfactual is sensitive to this being the true open-loop baseline. |

| MHE | Moving-horizon estimator producing the state/disturbance (tool-state) estimate the controller acts on. |

| MPC | Model-predictive controller computing knob moves over a horizon. |

| Estimator weights (State QR) | MHE process-noise vs measurement-noise weighting; balances trust in the model versus trust in the measurement. |

| Controller weights (Controller QR) | MPC tracking-error vs move-suppression weighting; balances hitting target versus knob stability. |

| DOE data | Designed-experiment data (typically fractional/partial factorial); wide operating window, sparse, possibly stale, with a known aliasing structure. |

| Inline data | Closed-loop operating data; narrow operating window, noisy, contains manual adjustments and existing FB/FF control with unknown errors. |

| Innovation | One-step MHE prediction error; the part of the output the estimator's model did not anticipate. |

| Noise floor | The variance level below which variation is indistinguishable from sampling noise (estimation) or unachievable by any controller (control). |

| Cpk / %OOS | Process-capability / out-of-spec metrics; require per-output targets and spec limits. |

| Harris index | Ratio of achieved output variance to the minimum-variance (minimum-achievable) benchmark. |

---

## 3. Architectural Overview

```

                         ┌───────────────────────────────────────────────┐

                         │            CROSS-CUTTING FOUNDATIONS           │

                         │  §4.1 Data model & schema contract             │

                         │  §4.2 Data-quality & assumption validation     │

                         │  §4.3 Excitation & confounding analysis (VIF,  │

                         │        spectrum, effective rank)               │

                         │  §4.4 Noise-floor estimation (MP analytic +    │

                         │        permutation/CV empirical)               │

                         │  §4.5 Counterfactual reconstruction service    │

                         │  §4.6 Validation framework (forward-chaining,  │

                         │        purge/embargo, block bootstrap)         │

                         └───────────────────────────────────────────────┘

                                          ▲      ▲

        ┌──────────────────────┬──────────┘      └──────────┬──────────────────────┐

        │                      │                            │                      │

 ┌──────┴───────┐   ┌──────────┴────────┐        ┌──────────┴───────┐   ┌──────────┴───────┐

 │ 1. FB / proc │   │ 6. All-data       │        │ 3. MHE-MPC       │   │ 5. Diagnostics & │

 │   dynamic    │◄─►│   regression      │───────►│   simulator      │◄─►│   visualization  │

 │   model ID   │   │   machine         │  FF    │ (w/ & w/o ctrl)  │   │                  │

 └──────┬───────┘   └──────────┬────────┘ leads  └──────────┬───────┘   └──────────────────┘

        │                      │                            │

        │  gaps                │  gaps                       │ wrapped by

        ▼                      ▼                            ▼

 ┌──────────────────────────────────┐            ┌──────────────────┐

 │ 2. Experiment planner            │            │ 4. Controller    │

 │    (gap study, prioritized       │            │    optimizer     │

 │     excitation proposals)        │            │ (outer loop)     │

 └──────────────────────────────────┘            └──────────────────┘

```

---

## 4. Cross-Cutting Foundations

### 4.1 Data Model and Schema Contract

- **DM-01** The system SHALL ingest a unified event-indexed table where each processing event carries: timestamp; recommended knob vector; used knob vector; FF inputs (pre-processing measurements, tool sensors, process configuration including discrete configs such as product count in tool); and, when available, the post-processing measurement(s) with their own measurement timestamp.

- **DM-02** The system SHALL treat metrology as **sparse and asynchronous**: a measurement for one event may arrive after several subsequent events have already processed. Joins between knobs/FF and `y` SHALL be by measurement provenance, not by row position.

- **DM-03** The system SHALL support **array-valued cells** (e.g., per-site or per-trace values within one event). Finiteness MUST be checked per cell, because standard null-checking (`pandas .notna()`) returns `True` for an array containing only NaNs. A dedicated `_is_measured` predicate SHALL gate every measurement-presence decision.

- **DM-04** The system SHALL enforce and verify the FB-model **shape contract** (`M` is `n_out × n_knob`) at ingestion and at every interface, failing loudly on mismatch.

- **DM-05** The system SHALL retain the **recommended-vs-used distinction** end to end; their divergence is a first-class signal (manual override or existing-controller error), not a cleanup artifact (see §EX-04).

### 4.2 Data-Quality and Assumption Validation Layer

This layer is the system's gatekeeper. For every data source, it makes the source's assumptions explicit, detects critical violations, and emits a *use / down-weight / reject* decision with a recorded rationale. No downstream component may consume data that has not passed through it.

**DOE data**

- **DQ-01** *Assumptions.* Factor levels span the intended operating range; the design's **alias/confounding structure (resolution)** is known; tool state is approximately stationary across the DOE campaign; measurements are valid.

- **DQ-02** *Critical-violation detection.* The layer SHALL (a) parse the design and report its resolution and the alias chains (which main effects are confounded with which interactions); (b) detect **covariate shift** between DOE operating points and the current inline operating point (e.g., distributional distance on knobs/FF); (c) detect **staleness** via the time gap between DOE collection and the current epoch, flagging when tool aging likely invalidates the gain.

- **DQ-03** *Decision rule.* DOE SHALL be used for **wide-range gain structure and curvature** that inline data cannot reveal. Terms that are irrecoverably aliased at the design's resolution SHALL be flagged and excluded from any claim of identifiability (surfaced to the experiment planner as a resolvable gap, §EP). Stale DOE SHALL be down-weighted or rejected for gain, with the staleness recorded.

**Inline data**

- **DQ-04** *Assumptions (and why most are violated).* Inline closed-loop data violates the conditions for direct gain identification: knobs are correlated with estimated state (**closed-loop confounding**), excitation is confined to a **narrow operating window**, manual adjustments are **unmodeled exogenous inputs**, the existing FB/FF models carry **unknown errors** (so recommended ≠ used), and metrology is **asynchronous and sparse**.

- **DQ-05** *Critical-violation detection.* The layer SHALL compute, on the inline knob+FF regressors: variance-inflation factors (VIF) and the condition number / effective rank of the regressor covariance (§EX); the recommended-vs-used divergence per knob and per epoch (§EX-04); metrology timestamp alignment and effective measured-sample count after `_is_measured` filtering; and stationarity diagnostics on the raw output (ADF/KPSS, variance-ratio) to confirm the presence of drift.

- **DQ-06** *Decision rule.* Inline data alone SHALL NOT be used for gain identification when the effective excitation rank is deficient in the knob directions of interest; this condition SHALL be reported, not silently worked around. Inline data SHALL instead be used for: noise-floor estimation, drift tracking, validation (including control-off gold episodes, §VAL-04), and innovation-target regression (§REG-04). Where DOE supplies gain, inline supplies the operating-point-local correction and the noise characterization.

**Joint-use contract**

- **DQ-07** When DOE and inline data are combined, the layer SHALL record the **provenance and weight** of every contribution and SHALL prevent a stale or covariate-shifted DOE point from dominating an estimate at the current operating point. Combination SHALL be explicit (e.g., weighted / hierarchical), never an unlabeled concatenation.

- **DQ-08** All decisions (use/down-weight/reject) and their triggering diagnostics SHALL be persisted with the resulting model artifact so any downstream consumer can audit why a datum was or was not used.

### 4.3 Excitation and Confounding Analysis

- **EX-01** The system SHALL quantify closed-loop confounding via VIF on the knob+FF regressors and report it in stakeholder-facing terms ("this much of the apparent knob–output relationship is an artifact of feedback"), to justify experimental investment.

- **EX-02** The system SHALL compute the **spectrum (eigenvalues/singular values) of the regressor covariance** and locate the noise floor on it (§4.4), reporting the **effective rank** (number of directions above the floor) as the count of independently identifiable directions — the direct transplant of the signal-vs-noise eigenvalue split into identification.

- **EX-03** The system SHALL map effective-rank deficiency to **specific unidentifiable knob/FF directions** (not just a scalar), so the experiment planner can target them.

- **EX-04** The system SHALL mine **accidental excitation**: episodes where used knobs diverge from recommended (manual overrides) or where control was off inject variance into directions the controller normally suppresses. These episodes SHALL be detected, catalogued, and offered as identification data and validation gold, because they are "free" excitation in otherwise excitation-deficient records.

### 4.4 Noise-Floor Estimation

- **NF-01** The system SHALL provide an **analytic** floor where assumptions hold: the Marchenko–Pastur edge `(1 + √q)²` with `q = N/T` for balanced-panel covariance spectra, and the **minimum-variance / Harris** benchmark for achievable control residual.

- **NF-02** The analytic floor SHALL declare its assumptions and SHALL refuse to apply silently when violated. In particular, `q = N/T` assumes a balanced panel (every series observed over all `T`); with the asynchronous/sparse metrology of §DM-02 this is generally false.

- **NF-03** The system SHALL provide an **empirical** floor for assumption-violating data: a permutation/shuffle procedure that destroys cross-series correlation while preserving each series' marginal distribution and its exact missingness pattern, repeated to build the null distribution of the top eigenvalue (or residual statistic). The chosen high quantile (e.g., 99th percentile) is the floor. Real eigen-directions exceeding it are signal.

- **NF-04** When the analytic floor's assumptions are violated, the empirical floor SHALL take precedence, and the system SHALL record which floor was used and why.

- **NF-05** Optional eigenvalue **cleaning** (clipping and Ledoit–Péché-style nonlinear shrinkage) SHALL be available for stabilizing identification covariances; the system SHALL note that nonlinear shrinkage is itself derived under the balanced-panel assumption and SHALL prefer cross-validated cleaning strength (§VAL) over the analytic edge when panels are unbalanced.

### 4.5 Counterfactual Reconstruction Service

- **CF-01** The system SHALL reconstruct what output variation would have looked like **without R2R control active** by *subtraction*, not simulation: because FF and FB both act through the observed `u_used`, the control contribution is `M @ (u_used − u0)`, and process sensitivity and per-event disturbance realizations cancel algebraically (validated to machine precision on synthetic MIMO data).

- **CF-02** The service SHALL surface its sensitivity to `u0` / `initial_knob` being the true open-loop baseline; the entire counterfactual depends on it, so the baseline source and any uncertainty SHALL be recorded and propagated.

- **CF-03** The service SHALL accept a relative-uncertainty estimate on `M` and produce **Monte-Carlo uncertainty bands** on the counterfactual.

- **CF-04** This service is shared: Component 3 uses it for the no-control simulation arm, Component 6 uses the same subtraction to decouple the controller's contribution from the regression target, and Component 5 uses it for instability/gain diagnostics.

### 4.6 Validation Framework

- **VAL-01** Because inline data carries temporal structure (drift, autocorrelation, trends), the system SHALL **NOT** use naive (random) k-fold cross-validation for any temporally ordered data; doing so leaks future-adjacent information and yields optimistic, non-deployable error estimates.

- **VAL-02** The default temporal validator SHALL be **forward-chaining / expanding-window** (train on the past, test on the future, expand, repeat). Reported error SHALL be the forward error even though it is larger (more honest) than a shuffled estimate.

- **VAL-03** When autocorrelation has a known timescale, the validator SHALL apply **purge/embargo** (a gap between train and test folds) to prevent autocorrelated neighbors leaking across the boundary.

- **VAL-04** The system SHALL identify and **set aside control-off episodes** (outages, maintenance windows) as validation gold for the counterfactual reconstruction and for clean open-loop checks; these SHALL NOT be used for both fitting and validating the same artifact.

- **VAL-05** Resampling-based selection (e.g., stability selection in Component 6) SHALL use **block bootstrap** (contiguous blocks) rather than i.i.d. bootstrap, to preserve local temporal correlation.

- **VAL-06** The system SHALL test for **concept drift in the relationship itself** (not just in the output) by checking whether forward-validation error degrades systematically with horizon; systematic degradation is the signature of a mapping that is itself moving and SHALL trigger per-campaign modeling or inclusion of tool-age/state as a feature.

---

## 5. Component 1 — FB / Process-Dynamic Model Identification

**Purpose.** Identify the MIMO process gain `M` and the dynamics needed for the state-space MPC, from a small known I/O set (≈10 knobs × ≈10 outputs), partial-factorial DOE, and noisy inline data, while honestly representing what the data can and cannot identify.

**Inputs.** DOE table, inline table, schema/shape contract, prior `M` (if any), the existing FB/FF models (to recover recommended-vs-used), control-off episode catalogue.

**Outputs.** Identified state-space model (A, B/`M`, C as applicable); a **parameter-uncertainty covariance**; the **identifiable-direction report** (effective rank and which directions are unresolved); the data-usage decision log; residual/innovation diagnostics.

- **FB-01** The identifier SHALL fuse DOE (wide-range gain/curvature) and inline (operating-point-local correction, noise) per the joint-use contract (§DQ-07), with provenance-weighted combination.

- **FB-02** The identifier SHALL solve the identification as a regularized estimation whose conditioning is governed by the regressor-covariance spectrum, and SHALL set **model order / retained directions by where the singular-value spectrum drops into the noise floor** (§NF) — the 8-vs-392 decision applied to system ID. Subspace-style order selection (model order = singular values above the noise gap) SHALL be supported.

- **FB-03** The identifier SHALL apply regularization analogous to eigenvalue cleaning (ridge/Tikhonov on the normal equations; SVD truncation as clipping; shrinkage on the regressor covariance) to prevent noisy small-eigenvalue directions inflating parameter variance and corrupting the MPC model.

- **FB-04** The identifier SHALL report **closed-loop identifiability limits explicitly** (§EX): where inline excitation is deficient, the affected entries of `M`/dynamics SHALL be marked low-confidence and routed to the experiment planner, never reported as confidently identified.

- **FB-05** The identifier SHALL incorporate **accidental excitation** (manual-override and control-off episodes, §EX-04) as additional identification data where it improves the identifiable-direction report.

- **FB-06** The identifier SHALL produce **innovation/residual diagnostics**: one-step residual variance (the unexplained-variance term), whiteness (Ljung-Box), and a check that the residual is consistent with the estimated noise floor. Colored residuals SHALL be reported as model mis-specification, distinct from mere noise.

- **FB-07** The identifier SHALL emit the parameter-uncertainty covariance in a form the simulator (§SIM) and counterfactual service (§CF-03) can consume directly for uncertainty propagation.

- **FB-08** *Identifiability boundary.* The identifier SHALL document that passive closed-loop data cannot separate **gain error** from **estimator tuning** — the loop reveals only their product — and SHALL NOT emit attributions that require resolving this without either active excitation or the estimator specification. The limit is demonstrated, not papered over.

---

## 6. Component 2 — Experiment Planner

**Purpose.** Study the gap in current data, propose experiments that add the most model-relevant information, prioritize them, and estimate each one's value — so that high-cost experiments are justified before a human approves them.

**Inputs.** Identifiable-direction reports from Components 1 and 6; current regressor-covariance spectrum and VIF; the set of *adjustable* inputs (knobs and a few configs such as product count) versus *non-adjustable* sensors; material/operating constraints.

**Outputs.** A ranked list of proposed experiments, each with: the directions it excites, its estimated information gain, its predicted effect on model quality, and its feasibility/cost.

- **EP-01** The planner SHALL use the closed-loop confounding measures (VIF, effective-rank deficiency, §EX) on passive data to **quantify and localize the gap** — which specific knob/FF directions are unidentifiable — and to justify experimental investment to stakeholders.

- **EP-02** The planner SHALL only propose experiments on **adjustable inputs**. Most sensors are not adjustable; the planner SHALL NOT propose experiments for them and SHALL treat them as observed-only in both Components 1 and 6.

- **EP-03** The planner SHALL support a **seed design** stage and a **sequential infill** stage. Seed designs SHALL support D-optimality (parameter-precision-oriented) and I-optimality (prediction-variance-oriented), with the primary objective configurable.

- **EP-04** Sequential infill SHALL use a surrogate (Gaussian-process) with an acquisition criterion selectable among integrated-MSE / active-learning-Cohn (ALC) and expected information gain, choosing the next experiment that most reduces model uncertainty in the gap directions.

- **EP-05** Each proposed experiment SHALL carry a **priority metric** and an **estimated value on the model** (e.g., predicted reduction in parameter-covariance volume for a D objective, or in integrated prediction variance for an I objective), so experiments can be ranked and the marginal value of the next experiment compared against its cost.

- **EP-06** The planner SHALL operate under **manual approval**: because experiment cost is high, proposals SHALL be presented for a scientist to approve rather than executed automatically (the reinforcement-learning analogy with a human-in-the-loop gate). The priority metric exists precisely to make this approval decision well-founded.

- **EP-07** The planner SHALL expose a **gate-based workflow**: framing → passive-data diagnosis → seed design → model fit / re-diagnosis → sequential infill → stopping / display / handoff, with explicit stopping criteria (e.g., marginal information gain below a threshold, or budget exhausted).

- **EP-08** The planner SHALL declare its **own open design decisions** as configuration, not hidden defaults: model form, primary objective (D vs I), the input-distribution weighting `p(x)` used by I-optimality, loop status during experiments (open vs closed), and budget granularity.

---

## 7. Component 3 — MHE / MPC Controller Simulator

**Purpose.** Simulate process outcomes **with and without** control in place, reporting capability and control metrics, while accepting and propagating the errors / unexplained variance in both the FB and FF models. Fully configurable.

**Inputs.** FB model + its uncertainty (§FB-07); FF model + its uncertainty; per-output targets and spec limits; disturbance/noise model; full controller and estimator configuration.

**Outputs.** Time series of simulated `y`, knob moves, and estimated state; metrics **Cpk, standard deviation, mean-off-target, Harris index** (and %OOS) for each arm; uncertainty bands on all metrics.

- **SIM-01** The simulator SHALL produce two arms — **control-on** and **control-off** — using the shared counterfactual service (§CF) for consistency, so the no-control arm is the algebraic counterfactual rather than an independently coded simulation.

- **SIM-02** The simulator SHALL accept **model-error inputs** for both FB and FF (e.g., a relative-uncertainty estimate on `M`, FF residual variance, or full covariances) and SHALL propagate them via Monte-Carlo to produce metric uncertainty bands, not point metrics.

- **SIM-03** The simulator SHALL implement an **MHE** whose estimator weights (State QR: process-noise vs measurement-noise) are configurable, and an **MPC** whose controller weights (Controller QR: tracking-error vs move-suppression) are configurable. The two weight sets SHALL be independently addressable.

- **SIM-04** The simulator SHALL expose, at minimum, these configuration parameters: measurement **sampling rate** (and asynchronous/delayed-metrology behavior); **measurement-tool uncertainty**; **min/max control adjustment per event** (move limits / rate limits); MHE **Q and R**; MPC **Q and R**; horizon lengths; and the disturbance/drift model driving the plant.

- **SIM-05** The simulator's plant SHALL support drift, autocorrelated disturbance, and measurable disturbances routed through the FF path, so that FF-model error and FB-model error can be studied independently.

- **SIM-06** Metric computation SHALL require per-output **targets and spec limits** (for Cpk/%OOS and mean-off-target) and SHALL compute the **Harris index** against the minimum-variance benchmark (§NF-01), so the simulator's output ties directly to the achievability floor used by the diagnostics.

- **SIM-07** The simulator SHALL be deterministic given a seed (§NFR) so that the optimizer (§OPT) and diagnostics (§DIAG) can compare configurations reproducibly.

---

## 8. Component 4 — Controller Optimizer

**Purpose.** Iteratively search controller/estimator configurations to find the best-performing one, using a selectable optimization algorithm and the simulator as the objective.

**Inputs.** The simulator (§SIM) as a black-box objective; a configurable search space (e.g., MHE Q/R, MPC Q/R, move limits); an objective metric; constraints.

**Outputs.** The best configuration found; the search trace; sensitivity of the objective to each configuration dimension.

- **OPT-01** The optimizer SHALL wrap the simulator in an **outer optimization loop**, evaluating candidate configurations by simulation and returning the configuration that optimizes the chosen metric (e.g., minimize standard deviation or mean-off-target, maximize Cpk, drive Harris toward 1).

- **OPT-02** The optimizer SHALL offer a **selection of algorithms** appropriate to a noisy, possibly non-convex, simulation-based objective (e.g., Bayesian optimization / GP-based, pattern/Nelder–Mead search, evolutionary strategies, and grid/random for baselines).

- **OPT-03** The optimizer SHALL optimize **against uncertainty**: because the simulator returns metric distributions (§SIM-02), the objective SHALL be robust (e.g., a risk-adjusted statistic), so the chosen configuration is not tuned to one favorable noise realization.

- **OPT-04** The optimizer SHALL report **objective sensitivity** to each configuration dimension, distinguishing dimensions the performance is genuinely sensitive to from those it is flat in — itself a noise-floor statement about which tuning choices matter.

- **OPT-05** The optimizer SHALL respect feasibility constraints (move limits, stability) and SHALL reject configurations the diagnostics (§DIAG) flag as unstable (e.g., gain-mismatch over-correction), rather than reporting a numerically optimal but physically unstable setting.

---

## 9. Component 5 — Diagnostics and Visualization

**Purpose.** Diagnose defective control models and tell the engineer *where* a performance problem lives, including the decisive judgement — borrowed from the Marchenko–Pastur / Ledoit–Péché noise-floor idea — of **whether the controller is already as good as it can be and the remaining variance is irreducible.**

**Inputs.** Identified FB/FF models and uncertainties; estimated state/innovation sequences; controller and estimator configurations; simulated and/or live metrics; the counterfactual service; the noise-floor estimator.

**Outputs.** A diagnosis report attributing residual variance to causes, with supporting visualizations and an explicit achievability verdict.

- **DIAG-01** *FF-leakage diagnostic.* The system SHALL test whether **changes in the controller's estimated state correlate with FF inputs**. A significant correlation means the FF model is failing to account for a measurable disturbance, so feedback is reactively absorbing what feedforward should pre-compensate. Such FF inputs SHALL be reported as **feedforward opportunities** (the same logic by which sensors explaining the innovation are feedforward candidates).

- **DIAG-02** *Gain-mismatch instability diagnostic.* The system SHALL detect the failure mode where the identified process gain is too small, causing the controller to over-correct and oscillate. Detection SHALL compare the **realized gain** (observed Δoutput / Δknob, via the counterfactual service) against the model gain, and SHALL flag oscillation signatures in the innovation/output. The diagnosis SHALL distinguish "moves too large because gain is underestimated" from genuine disturbance growth.

- **DIAG-03** *Estimator-QR vs Controller-QR attribution.* The system SHALL diagnose whether a performance problem lies in the **Controller QR** (MPC: knob-stability vs tracking weight) or the **State QR** (MHE: measurement-trust vs process-noise weight). The two have distinct signatures — controller-weight problems show in the move/tracking trade-off; estimator-weight problems show in the responsiveness and whiteness of the innovation — and the diagnostic SHALL report which, while honoring the §FB-08 limit that passive data alone cannot fully separate gain error from estimator tuning (resolution requires excitation or the estimator spec).

- **DIAG-04** *Variance decomposition.* The system SHALL decompose residual output variance into the recognized sources — incorrect FF model, incorrect process gain, sub-optimal estimator tuning, and irreducible noise — using innovation-sequence analysis, the Harris index, whiteness testing (Ljung-Box), and joint attribution regressions, presenting the share attributable to each (with the identifiability caveat surfaced, not hidden).

- **DIAG-05** *Achievability verdict (the MP/Ledoit–Péché analog).* The system SHALL compute the **achievability floor** — the minimum-variance / Harris benchmark, the control-theoretic analog of the Marchenko–Pastur noise edge — and SHALL state plainly when the achieved residual variance has reached that floor: *the controller is working as well as it could, and the remainder is unexplained / irreducible variance.* Conversely, when achieved variance sits above the floor, the system SHALL report the recoverable gap and route it to the responsible cause (FF, gain, or QR) above.

- **DIAG-06** *Visualization.* The system SHALL provide visualizations for: the regressor/identification spectrum with the noise floor overlaid (signal vs noise directions); innovation time series and its whiteness; realized-vs-model gain; state-vs-FF correlation; the variance-decomposition shares; and before/after metric comparison (control-on vs control-off, and pre/post cleaning). Visualizations SHALL make the noise floor and the achievability verdict legible at a glance.

---

## 10. Component 6 — All-Data Regression Machine

**Purpose.** Build a predictor of post-processing measurements from **all** available data (knobs, tool sensors, pre-processing measurements, process configuration), whether the data was collected with or without control, decoupling the controller's contribution when control is present, and handling the low-variation-knob problem by routing gaps to the experiment planner.

**Inputs.** The unified table; the FB/FF models and the counterfactual service (when control is present); the validation framework; the experiment planner interface.

**Outputs.** A predictive model with honest forward-validated error; a **defensible feature-importance ranking**; identified low-variation/unidentifiable inputs; feedforward-candidate sensors.

- **REG-01** The regression machine and the FB identifier (§5) SHALL be **one estimation engine in two configurations**, sharing the data model, data-quality layer, excitation analysis, noise-floor estimator, counterfactual service, and validation framework. This unification is mandatory.

- **REG-02** *Controller decoupling.* When a controller is in place, the regression SHALL **decouple the control contribution** using the counterfactual subtraction (`M @ (u_used − u0)`, §CF) so that relationships are estimated against the open-loop-equivalent signal rather than the controller's corrections. When control is absent, the raw data is used directly. When the control model's error is estimable it SHALL be propagated; when it is purely unknown, the regression SHALL fall back to innovation-target modeling (§REG-04) and flag the limitation.

- **REG-03** *High-dimensional, tiny-data regime.* The machine SHALL assume `N_features ≫ T` (e.g., thousands of sensors, hundreds of samples), where the sample covariance is singular and ordinary regression is undefined. It SHALL therefore use regularized / latent estimators whose retained dimensionality is set by the noise floor (§NF): **partial least squares** (supervised latent components maximizing covariance with the target — the directly target-relevant analog of the eigenvalue decomposition), **elastic-net** (sparse-with-correlated-groups), and **ridge** (dense-but-shrunk), with the choice encoding the engineer's structural belief and selected by validation.

- **REG-04** *Innovation-target option.* The machine SHALL support predicting the **post-control innovation/residual** rather than raw output. This separates what the control loop already tracks (drift, slow tool state) from what sensors can additionally explain; the target is closer to stationary, softening the temporal-validation problem, and sensors that explain the innovation are **feedforward candidates** — turning feature selection into a control-improvement search (the same object as §DIAG-01).

- **REG-05** *Honest validation.* The machine SHALL validate with the temporal framework (§VAL): forward-chaining, purge/embargo, never naive k-fold; and SHALL report the forward error.

- **REG-06** *Defensible selection.* Feature importance SHALL be produced by **stability selection** (elastic-net over **block-bootstrap** resamples, §VAL-05), reporting selection frequency ("selected in X% of resamples") rather than a single fit, to give a stakeholder-justifiable ranking robust to the split.

- **REG-07** *Low-variation handling and planner hand-off.* When variation in adjustable inputs (knobs, and configs such as product count) is too low — because they are tightly controlled or fixed — the machine SHALL **detect the resulting unidentifiability** (via the excitation analysis, §EX) and route those directions to the experiment planner (§EP). For **non-adjustable sensors** with low variation, no experiment is possible; the machine SHALL treat them as observed-only and report them as unidentifiable-by-data rather than proposing infeasible experiments.

- **REG-08** *Method discipline.* High-capacity nonlinear models (transformer / attention) SHALL NOT be the default at this data scale: with hundreds of samples they overfit and their importance maps are unstable. They MAY be used only as a **representation** stage (e.g., a regularized per-sensor temporal encoder) feeding a regularized linear/PLS head, and only when the regularized baseline leaves structured residuals indicating nonlinear temporal signal, validated to beat that baseline under the same temporal CV.

- **REG-09** *Representation choice for traces.* When inputs are full traces of differing temporal length (the ragged-panel problem), the machine SHALL default to **physics-aware aggregation** (per-substep / tool-state-aware statistics) to inject prior and reduce dimension before fitting, and SHALL retain the raw trace only when aggregation is shown to destroy discriminative signal (a transient or ramp-rate that the mean hides).

---

## 11. Component Interfaces and Data Flow

- **IF-01** Components 1 and 6 SHALL publish: the model artifact, its uncertainty covariance, the identifiable-direction report, and the data-usage log, in a shared schema consumable by Components 2, 3, and 5.

- **IF-02** Component 2 SHALL consume the identifiable-direction reports from 1 and 6 and SHALL publish approved-experiment specifications that, once executed, re-enter the data model (§4.1) and trigger re-identification.

- **IF-03** Component 3 SHALL consume the FB model + uncertainty (from 1), the FF model + uncertainty (from 6 or external), and the configuration; and SHALL publish metric time series and distributions to Components 4 and 5.

- **IF-04** Component 4 SHALL consume Component 3 as an objective and SHALL publish the chosen configuration back to Component 3's defaults and to Component 5 for verification.

- **IF-05** Component 5 SHALL consume artifacts from 1, 3, 4, and 6 plus the shared counterfactual and noise-floor services, and SHALL publish the diagnosis report and achievability verdict.

- **IF-06** The shared services (§4.4 noise floor, §4.5 counterfactual, §4.6 validation) SHALL be single implementations called by all components, never re-implemented per component, so that "the noise floor" and "the counterfactual" mean exactly one thing system-wide.

---

## 12. Non-Functional Requirements

- **NFR-01** *Stack.* Python with numpy / pandas / scipy is the preferred implementation stack; statistical tests (ADF, KPSS, variance-ratio, Ljung-Box) SHALL be available, implemented from scratch where a dependency is to be avoided. Gaussian-process surrogates support the experiment planner.

- **NFR-02** *Reproducibility.* All stochastic procedures (Monte-Carlo, bootstrap, optimization, simulation) SHALL be seedable and deterministic given a seed, so results are auditable and configurations comparable.

- **NFR-03** *Provenance and auditability.* Every model artifact SHALL carry its data-usage decisions, the noise floor used (analytic vs empirical) and why, the validation scheme and its error, and the assumptions checked — sufficient for a future session or reviewer to reconstruct the reasoning without re-establishing context.

- **NFR-04** *Validation by construction.* Each diagnostic / estimation module SHALL ship with synthetic smoke tests carrying **deliberately injected faults**, including intentional failure cases that confirm correct behavior at known identifiability limits (e.g., that the system *refuses* to separate gain error from estimator tuning on passive data).

- **NFR-05** *Self-contained handoff.* Each major component SHALL produce a reference document plus its module(s) so future work continues without re-deriving context, consistent with the established iterative pattern (theory → workflow → implementation → validation → reference).

- **NFR-06** *Honesty over convenience.* Across all components, hard identifiability and achievability limits SHALL be surfaced explicitly rather than producing misleading attributions; a correct "cannot be determined from this data" is a required output, not a failure.

---

## 13. Open Design Decisions and Risks

These require a human decision and SHALL be tracked as configuration rather than silently defaulted.

1. **Estimator unification boundary (resolved as default):** Components 1 and 6 unified into one engine (§REG-01). Revisit only if an organizational reason demands separate deliverables.

2. **Counterfactual baseline (`u0`):** confirm `initial_knob` is the true open-loop baseline; the entire counterfactual and the regression decoupling depend on it (§CF-02).

3. **Spec inputs:** obtain per-output targets and spec limits required for Cpk/%OOS and mean-off-target (§SIM-06).

4. **`M` uncertainty:** obtain a relative-uncertainty estimate on `M` to drive Monte-Carlo bands (§CF-03, §SIM-02).

5. **Model-consistency diagnostic:** the current consistency check conflates inaccurate `M` with FB tracking lag; the `y_pred` column is a cleaner isolation signal and SHALL be used to disambiguate (§DIAG-02/03).

6. **Experiment-planner choices:** model form, primary objective (D vs I), `p(x)` weighting, loop status during experiments, budget granularity (§EP-08).

7. **Regression target:** raw output vs post-control innovation (§REG-04) — depends on whether the control loop's drift estimate / innovation sequence is available as a column; this determines which path is open and SHALL be confirmed.

8. **Noise-floor default per dataset:** which floor (analytic MP vs empirical permutation) applies, given the asynchronous/sparse metrology that generally breaks the balanced-panel assumption (§NF-02/04).

---

*End of specification.*
