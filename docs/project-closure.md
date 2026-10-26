# Project Closure Report — `r2r_control` 0.3.0

**Status:** closed on 2026-10-24.
**History:**
- 0.2.0 was merged in PR #1 (merge commit `1fc041f`).
- The closure review of 0.2.0 found nine known issues (KI-1 to KI-9).
- 0.3.0 (PR #2) fixes KI-1 to KI-8 and part of KI-9. Section 8 gives each
  issue's evidence before and after the fix, and section 8.10 lists what
  remains or is new.

Read this before using any output on real process data. It states what the
toolchain assumes, where it applies, how it works, what it cannot do, the
defects found at closure and how they were resolved, and what to do next.

---

## 1. Summary

`r2r_control` is an offline Python toolchain for run-to-run (R2R) process
control. It:

- identifies the feedback gain;
- plans experiments;
- simulates and tunes an estimator/controller;
- diagnoses residual variance;
- screens feedforward (FF) candidates.

Every decision is framed as *signal versus a noise floor*. Estimation uses the
Marchenko–Pastur edge, or a structure-preserving surrogate floor. Control uses
the Harris minimum-variance benchmark.

At closure all 90 tests pass and the end-to-end walkthrough runs (§3). The
architecture, refusal behavior and assumption guards work as specified **on
synthetic data**.

The three issues that mattered most at the 0.2.0 closure are fixed:

1. **Closed-loop bias in the gain estimate (KI-1).**
   - Before: fusing closed-loop inline data with the DOE gave 10–31% gain error,
     against 2–5% from the DOE alone.
   - Now: closed-loop rows are detected by a feedback test and excluded from gain
     identification, so the fused error equals the DOE-only error (1.6–4.8%).
2. **Over-confident gain uncertainty (KI-2).**
   - Before: the actual error was 2.4–4.8× the reported uncertainty.
   - Now: the ratio is 0.6–1.4 (median 1.2), and the identified uncertainty
     reaches the simulator, counterfactual bands and phantom-FF envelope.
3. **No measurement delay in the simulator (KI-3).** Measurements now arrive
   `metrology_delay` steps late, and control degrades with delay as it should.

What remains before real-data use (§8.10):

- validation on synthetic data only;
- a VAL-04 trade-off that lowers FF-selection power when control-off episodes
  are the main FF-independent excitation;
- a gain-versus-FF ambiguity that the new `y_pred` diagnostic reports but
  cannot resolve;
- the revised specification is still missing from the repository.

## 2. Deliverables

| Item | Location |
|---|---|
| Python package (six components plus shared services) | `src/r2r_control/` (≈6,300 lines) |
| Test suite (90 tests: unit, injected-fault, refusal, architecture, integration) | `tests/` |
| End-to-end walkthrough | `examples/walkthrough.py` |
| Architecture and theory reference | `docs/architecture.md` |
| Requirements traceability matrix | `docs/requirements-traceability.md` |
| This closure report | `docs/project-closure.md` |
| Original specification | `design-requirements-specification.md` (see KI-9) |
| Change history | `CHANGELOG.md` |

## 3. Verification at closure

| Check | Result |
|---|---|
| Environment | Python 3.11.17, numpy 2.4.6, pandas 3.1.0, scipy 1.17.1, pytest 9.1.1, Pillow 13.0 |
| `pytest` | **90 passed** in ≈19 s (0.2.0: 64 tests in ≈8 s; the constrained MPC accounts for most of the increase) |
| `pytest -o filterwarnings=default -rw` | 90 passed, **no warnings** |
| `python examples/walkthrough.py` | completes in ≈18 s; every guard fires as intended |
| Closure probes (§11.3) | 0.2.0 reproduced KI-1 to KI-4. On 0.3.0 the same probes show the fixes. |

The package declares `requires-python >=3.9`. Only 3.11 was re-checked at
closure.

**What the tests establish:**
- the layering and decoupling rules hold;
- each guard and refusal fires on the fault it targets;
- each KI fix has a dedicated test (§8);
- the shared services behave correctly on constructed cases;
- the pipeline is reproducible and serializable.

The FB gain tolerances were tightened from 30–35% to 10% relative error.

**What the tests do not establish:** real-world accuracy. Every dataset comes
from `synthetic.py`, which shares the methods' structural assumptions (L1).

## 4. Methodology

| Stage | Method | Module |
|---|---|---|
| Data model | Event-indexed table with recommended and used knobs, FF inputs, sparse array-valued metrology, optional per-output controller prediction `y_pred` (§13.5) and optional trace columns. One presence predicate, `_is_measured`, handles all-NaN arrays. | `contracts/` |
| Data quality | **DOE:** alias/resolution parsing, staleness decay (180 d horizon), covariate-shift distance (down-weight above 1, reject above 3). **Inline:** VIF, condition number, entropy rank, ADF/KPSS/variance-ratio drift tests, effective measured count. Plus a **feedback test** (Granger-style F-test: does the history of measured outputs predict `u_t` beyond `u_{t−1}`?). It sets `gain_usable = False` for closed-loop data. Each check emits use / down-weight (weight 0.5) / reject with its rationale. | `core/data_quality/`, `core/excitation/feedback.py` |
| Noise floor | **Balanced panel:** MP edge `(1+√(N/T))²` on the correlation matrix. **Any missing cells:** the 99th percentile of the top eigenvalue under block-permutation (default) or phase-randomization surrogates. A plain shuffle is refused. **NF-05:** optional covariance cleaning for the ridge solve: clip, Ledoit–Péché, or linear shrinkage chosen by forward validation. | `core/noise_floor/`, `estimation/sensitivity/` |
| Excitation | Numerical rank (singular values above 1% of the largest) for identifiability, and MP signal rank for low-rank structure. Unidentifiable directions are expressed in knob/FF terms. Overrides and control-off episodes are mined as accidental excitation. | `core/excitation/` |
| Counterfactual | `y_nocontrol = y − M(u_used − u0)`. `u0` comes from control-off episodes, with a drift check and an optional interpolated time-varying `u0`. Monte-Carlo bands use the **identified** relative uncertainty on `M` and on `u0`. | `core/counterfactual/` |
| Validation | Forward chaining with purge and embargo (naive k-fold is refused), block bootstrap, control-off gold holdout with a representativeness check, concept-drift slope test. | `core/validation/` |
| C1 FB identification | **Gain rows (DQ-06):** DOE rows, plus inline rows only if the feedback test finds them open-loop. Closed-loop inline rows are excluded. Without a DOE, an endogenous fallback is used, flagged with a 100% bias floor and every knob low-confidence. **Fit:** SVD-truncated least squares at the numerical rank (ridge as an alternative). **Uncertainty:** `√(E‖ΔM‖²_F)/‖M‖` from the parameter covariance, combined in quadrature with a bias term from the DOE-vs-inline discrepancy. **Dynamics and FF:** first-order disturbance `A` (stability-clipped at \|λ\| ≥ 1); FF gains by ridge on the counterfactual. Ljung–Box innovation check. FB-08 refusal recorded. | `components/fb_identification/`, `estimation/` |
| C2 Experiment planner | **Coded space:** coded [−1, 1] with mapping to engineering units, integer and categorical snapping (E11), and p(x) weighting (uniform/central). **Gap:** diagnosis gives signed directions restricted to adjustable knobs, plus knobs flagged by C1. **Design:** D/I-optimal seed by coordinate exchange. **Infill:** GP (ALC/IMSE/EIG) with hyperparameters fitted by marginal likelihood when responses exist; acquisition multiplied by `1 + gap_weight·alignment` with the gap. Information discounted by `1 − rejection_fraction` when the loop is closed (A6); the fraction can be calibrated from measured residual excitation. Value per unit cost. Manual approval required. | `components/experiment_planner/`, `core/gp.py` |
| C3 Simulator | **Plant:** `y = d + FF·g + M(u−u0) + e`, with `d` = AR (identified `A`) + random-walk drift. **Measurements:** released `metrology_delay` steps late. **Estimator:** Kalman disturbance estimator (State QR). A late measurement within `mhe_horizon` is folded in by replaying the time updates. An older one goes to a slow-bias estimator. **Controller:** receding-horizon MPC on the static gain over the predicted disturbance trajectory `A^k d̂` (Controller QR), with move limits enforced exactly (bounded least squares) and optional absolute bounds (SLSQP). **Outputs:** records `y_pred`; Monte-Carlo over multiplicative `M` and FF perturbation; metrics with and without control. | `components/simulator/`, `controller/`, `metrics/` |
| C4 Optimizer | Robust objective = mean output std + λ·(p95 − p05 Monte-Carlo band). A configuration is rejected if its controlled std exceeds 5× the uncontrolled std. Search: random, grid, Nelder–Mead, evolutionary or GP-EI Bayesian, over a simulator factory. Finite-difference sensitivity per tuning dimension. | `components/controller_optimizer/` |
| C5 Diagnostics | **Decision tree:** excitation gate (RMS Δu ÷ knob spread above 0.75) → gain mismatch (realized/model ratio above 1.3 plus lag-1 ringing) → **2b. `y_pred` model consistency** (§13.5, when `y_pred` is logged) → FF leakage (Δstate vs ΔFF correlation; Šidák across outputs, Holm across inputs) → QR attribution (innovation whiteness, then move/track ratio) → variance-share allocation → Harris verdict (mean index ≤ 1.25 means at floor). Blocked nodes return "unidentifiable — excitation required". | `components/diagnostics/` |
| C6 Regression machine | **Decoupling:** removes the controller via the counterfactual. **Phantom-FF quarantine:** a sensor is quarantined if `\|t\| > 2` and its slope lies within `2·rel_unc·‖m_row‖·‖a_j‖`. **Model choice:** ridge, PLS or elastic net by forward chaining (purge 1, configurable embargo). **Holdout:** control-off episodes held out as gold (VAL-04). **Selection:** elastic-net stability selection over block-bootstrap resamples, per feature and per correlated group. **Traces:** aggregated per substep, with the raw resampled trace kept only when a held-out check shows aggregation loses signal (REG-09). Low-variation inputs are routed to the planner. | `components/regression_machine/`, `estimation/regularized/` |
| Metrics | Cpk, std, %OOS, mean off target, Harris (delay = metrology delay + 1). **Optional finite-sample intervals:** Cpk with `n_eff = n_measured · ESS/len`, which discounts autocorrelation; Harris by block bootstrap. Minimum-sample gate (default 8). ADF uses MacKinnon finite-sample critical values and p-values. | `metrics/`, `core/stats.py` |
| Orchestration | `GatePipeline` runs identify → plan → simulate (→ optimize) → diagnose → regress from one `RunConfig`. It passes C1's `A`, FF gains, uncertainty and low-confidence knobs downstream, with full provenance. | `pipeline/` |

## 5. Assumptions

### 5.1 Load-bearing assumptions with guards (§1.4)

Each guard detects the stated violation. The right-hand column says what it
does **not** catch.

| ID | Assumption | Guard | What the guard does not catch |
|---|---|---|---|
| A1 | Gain `M` is time-invariant (drift is additive) | Realized-gain trend monitor | Fast or step changes inside one 60-event window. Drift masked by tight control (realized gain undefined). |
| A2 | `u0` is valid over the analysis horizon | Control-off baseline trend, validity horizon, time-varying `u0` | Needs at least 2 control-off episodes. With none it falls back to the global knob mean and only warns. |
| A3 | `M` is accurate enough to decouple | Monte-Carlo propagation and phantom-FF quarantine, now sized by the **identified** uncertainty | The uncertainty is only as good as its bias term. With DOE-only identification the bias term is zero, so DOE-to-inline drift is not represented. |
| A4 | Metrology delay is shorter than the MHE horizon | Delayed measurements are folded into the estimator if within the horizon, otherwise routed to a slow-bias estimator | The estimator is an unconstrained Kalman filter with replay, not a constrained MHE. |
| A5 | Control-off episodes are representative | Tool-state distance above 3σ excludes an episode from the C6 gold holdout | Needs a tool-state signal (`tool_state_cols`). Without one, every episode is used. |
| A6 | Injected excitation reaches the process | Closed-loop discount on information gain; `calibrate_rejection` measures it | Calibration needs a pilot experiment. Otherwise the configured 0.6 rejection is used. |

### 5.2 Modelling assumptions

| ID | Assumption | Consequence if violated | Guard (0.3.0) |
|---|---|---|---|
| M1 | **Linear** gain: `Δy = M Δu` over the operating range. No curvature or interactions in the FB model. | Gain varies with operating point. DOE "curvature" information is unused. | none |
| M2 | **Superposition:** output = disturbance + FF + control + noise, with the disturbance independent of knob settings | The counterfactual and the decoupling are wrong. | none |
| M3 | Rows used for gain identification are **exogenous** | Biased gain | **Feedback test** excludes closed-loop inline rows; recorded as an assumption on the artifact. It can miss feedback through signals that are not in the table (§8.10). |
| M4 | The disturbance is roughly first-order AR plus random-walk drift | Harris benchmark and simulated performance mis-stated for richer disturbances | none |
| M5 | **Run-to-run:** one event is one run, the process settles within a run, and there are no within-run dynamics | Unsuitable for continuous or dynamic processes | none |
| M6 | Outputs are roughly normal (for Cpk intervals) | Intervals mis-stated for heavy tails | Autocorrelation is now discounted (ESS), normality is not checked |
| M7 | Simulation treats the identified `M`, `A` and FF gains as the plant. Model error is i.i.d. multiplicative noise of the identified size. | Structured or biased plant–model mismatch is not represented. | none |
| M8 | Missing metrology is ignorable (not caused by the process state) | Surrogate floors and metrics biased if, say, only bad lots are measured | none |
| M9 | DOE factors are coded around `u0` | Wrong coding shifts the planner's design | `knob_bounds` maps proposals to engineering units |

### 5.3 Decisions that belong to a person (§13)

These are configuration, not hidden defaults. Each needs confirming per process.

| Decision | Where | Default at closure |
|---|---|---|
| `u0` is the true open-loop baseline | `FBIdentifier.identify(u0=…)`, `RunConfig.allow_time_varying_u0` | Control-off mean; static |
| Targets and spec limits | `RunConfig.targets/lsl/usl` | Required, no default |
| Minimum sample count for metrics | `RunConfig.min_samples` | 8 (low; 30 or more is advisable) |
| Relative uncertainty on `M` | `RunConfig.rel_uncertainty_M` | `None` = use C1's identified value. An override below it is logged with a warning. |
| Planner objective, loop status, budget, integer dimensions | `RunConfig` → `PlannerConfig` | D-optimal, open loop, 8 runs |
| Planner categorical levels, engineering bounds, cost model | `PlannerConfig` only (not yet in `RunConfig`) | none, coded units, unit cost |
| Regression target (raw or innovation) | `RunConfig.regression_target` | raw |
| Hold out control-off episodes as gold (VAL-04) | `RegressionMachine(gold_holdout=…)` | on (see §8.10, R-1) |
| Consistency diagnostic using `y_pred` (§13.5) | `EventTable.y_pred_cols`; `diagnose(y_pred=…)` | runs when the controller's prediction is logged |

## 6. Applicability

**Intended use:** offline engineering analysis and decision support for an
existing or planned R2R loop. Every output is a recommendation for a process or
control engineer. Experiment proposals require manual approval (EP-06). Nothing
in the package actuates a tool.

**Good fit:**
- Discrete-run processes (semiconductor-style R2R, batch steps) where a run's
  response settles within the run.
- A few to roughly ten knobs and outputs, up to tens of FF sensors, and hundreds
  to low thousands of events. The tests use 400 events, 4 knobs, 5 FF sensors
  and 3 outputs.
- Behavior close to linear around the operating point, with gain stable over the
  analysis window.
- Some exogenous excitation available: a DOE campaign, manual overrides, or
  control-off episodes.

**Poor fit or out of scope:**
- Continuous processes, or within-run dynamics that need transfer-function
  models (M5).
- Strongly nonlinear, regime-switching, or fast-drifting-gain processes (M1, A1).
- Hundreds of sensors or raw traces at scale (scale is untested; L5).
- Real-time control, automatic actuation, or safety-critical use (L4).

**Data prerequisites:**
- An event table with recommended and used knobs, FF inputs, and timestamped
  outputs.
- Targets and spec limits.
- At least 2 control-off episodes, for the `u0` drift check and the C6 gold
  holdout.
- **A DOE, or open-loop inline excitation, for the gain.** Closed-loop inline
  data is now detected and excluded (DQ-06). Without a DOE the result is
  flagged as unusable (100% bias floor).
- Recommended: the controller's own prediction `y_pred` per output, for the
  §13.5 consistency diagnostic.
- Enough measured events per output for metrics. The gate is 8; meaningful Cpk
  needs far more.

## 7. Limitations (by design or scope)

| ID | Limitation |
|---|---|
| L1 | **Validated on synthetic data only.** The generator encodes M1, M2, M4 and M7, so the tests check self-consistency, not real-world validity. No back-test on historical process data has been done. |
| L2 | Linear static-gain FB model with no curvature, interactions or operating-point dependence (M1). |
| L3 | The MPC is multi-step but its plant model is the static gain. Only the disturbance prediction evolves over the horizon, so the horizon matters little unless moves are heavily penalized. The disturbance model is first order. The estimator is an unconstrained Kalman filter with replay of late measurements. |
| L4 | A library only: no data connectors, CLI, UI, or service. `ModelArtifact` serializes (`to_dict`/`to_json`) but cannot be loaded back. |
| L5 | Scale is untested beyond the test sizes. These parts will be slow on large data: cell-by-cell Python loops (`y_matrix`, `_is_measured`), an empirical floor costing O(n_perm·N·T), coordinate exchange costing O(runs·pool·passes), and SLSQP when absolute bounds bind. |
| L6 | Ridge λ = 1, elastic-net λ = 0.1 and PLS (at most 5 components) are fixed, not tuned inside forward validation. GP hyperparameters are now fitted by marginal likelihood over a small grid. |
| L7 | The regression machine fits one output per call, and `GatePipeline` regresses only the first output. |
| L8 | Variance decomposition is an **allocation rule**: the recoverable share is split equally among the causes the tree flagged. It does not estimate each cause's contribution. |
| L9 | The FB-08 refusal is structural: it is always emitted. Separating gain error from estimator tuning always needs active excitation or the estimator specification, as the spec intends. The §13.5 diagnostic narrows this when `y_pred` is logged. |
| L10 | Decision thresholds are heuristics calibrated on synthetic loops (§9). |
| L11 | Only Python 3.11 was verified at closure; the declared minimum is 3.9. |

## 8. Known issues found at the 0.2.0 closure, and their resolution

The 0.2.0 closure probes (§11.3) reproduced KI-1 to KI-4 directly. KI-5 to
KI-9 were found by code inspection.

| KI | Severity | Status in 0.3.0 |
|---|---|---|
| KI-1 Closed-loop bias in the fused gain | High | **Fixed** |
| KI-2 Gain uncertainty too small, not propagated | High | **Fixed** |
| KI-3 No simulated metrology delay | Medium | **Fixed** |
| KI-4 Inert configuration, clipped move limits | Medium | **Fixed** (see L3 for what remains simplified) |
| KI-5 Identified dynamics not used by the simulator | Medium | **Fixed** |
| KI-6 Planner not targeted at the gap | Medium | **Fixed** |
| KI-7 Unwired or untested code | Low | **Fixed**, except `subspace_order` (still test-only) |
| KI-8 Statistical approximations | Low | **Fixed** |
| KI-9 Documentation gaps | Low | **Partly fixed**: §13.5 implemented; revised specification still missing |

### KI-1 (High, fixed): closed-loop inline data biased the fused gain

**Before (0.2.0).** Relative gain error `‖M̂ − M‖/‖M‖` on the synthetic
benchmark (probe A, seeds 0–4, `ff_leak` 0 and 0.4):

| | Fused (shipped) | Inline only | DOE only |
|---|---|---|---|
| Error range | 0.099–0.310 | 0.395–1.365 | 0.016–0.048 |

Adding inline data made the gain 3–8× worse than the DOE alone. There were three
causes:
1. The inline down-weight verdict carried no weight.
2. The quality checks detected collinearity, not endogeneity.
3. The test accepted 30% error.

**Fix.**
- `feedback_test` (`core/excitation/feedback.py`) asks whether the history of
  measured outputs predicts each knob's setting beyond its previous value. This
  is the signature of a controller reacting to the process. If it does
  (Bonferroni over knobs, α = 0.01), the inline report sets
  `gain_usable = False`.
- `FBIdentifier` builds the gain from DOE rows plus inline rows only when they
  are usable. It records the exclusion in provenance and the M3 assumption on
  the artifact.
- Without a DOE, a closed-loop fallback is fitted but flagged: 100% bias floor,
  all knobs low-confidence.
- A `down_weight` verdict now carries weight 0.5 (configurable).

**After (0.3.0).** The same probe gives `gain_source = "DOE only"` in every run.
The fused error equals the DOE-only error (0.016–0.048). FB tests are tightened
to 10%.

Tests:
- `test_closed_loop_inline_rows_do_not_bias_the_gain`
- `test_endogenous_fallback_is_flagged_with_bias_floor`
- `test_open_loop_inline_is_used_and_checked_against_doe`
- `test_inline_down_weight_verdict_carries_a_weight`

### KI-2 (High, fixed): reported gain uncertainty was far smaller than the error

**Before.** The actual error divided by `FBModel.relative_uncertainty()` was
2.4–4.8 (median 3.7). `GatePipeline` ignored the identified uncertainty and
used the configured 0.1.

**Fix.**
- Relative uncertainty is now
  `hypot(variance term, bias term)`. The variance term is
  `√(trace(cov)·Σσ²_j)/‖M‖`, the expected Frobenius error summed over
  outputs (`contracts/artifact.py`). The bias term is the excess discrepancy
  between separate DOE and inline fits when both enter the gain, or the 100%
  floor for the endogenous fallback.
- `RunConfig.rel_uncertainty_M` defaults to `None`, meaning "use the identified
  value". An override below it is logged with a warning.
- The simulator, counterfactual bands, phantom-FF envelope and regression
  machine receive the propagated value.

**After.** On probe A the ratio is 0.60–1.39 (median 1.19).

Tests: `test_reported_gain_uncertainty_is_calibrated`,
`test_pipeline_propagates_identified_dynamics_and_uncertainty`.

### KI-3 (Medium, fixed): the simulator did not model metrology delay

**Before.** Delay only selected the route. Delay 0 and delay 1 gave identical
results.

**Fix.**
- Measurements enter a queue and are released `metrology_delay` steps late.
- A released measurement within `mhe_horizon` is folded in by propagating the
  filter to its sample time, updating, and predicting forward to the present
  (`KalmanMHE.predict_ahead`). An older one feeds the slow-bias estimator.
- The controller acts on the predicted current disturbance.
- The Harris benchmark uses delay = metrology delay + 1.

**After.** Probe B, mean controlled std per output:

| Delay | Output 1 | Output 2 | Output 3 |
|---:|---:|---:|---:|
| 0 | 0.390 | 0.368 | 0.391 |
| 1 | 0.427 | 0.427 | 0.455 |
| 2 | 0.479 | 0.496 | 0.546 |

In the walkthrough, delay 0 vs 2 within the horizon gives
0.419 vs 0.525 on output 1.

Test: `test_metrology_delay_is_simulated`.

### KI-4 (Medium, fixed): inert configuration fields and a clipped move limit

**Before.**
- `mpc_horizon` and `px_weighting` were never read.
- The move limit was applied by clipping the unconstrained move.
- There were no absolute bounds.

**Fix.**
- `MPCController` optimizes `Δu_0 … Δu_{H−1}` against the predicted disturbance
  trajectory `A^k d̂`.
- Move limits are enforced exactly by bounded least squares.
- Optional absolute bounds (`u_lower`/`u_upper`) are enforced by SLSQP.
- `px_weighting` ("uniform"/"central") sets the planner's reference
  distribution and is validated.

**After.**
- **Move limit** (limit 0.1, disturbance [1, 0, 0]):

  | Move | Cost |
  |---|---:|
  | Clipped unconstrained move | 1.106 |
  | No move | 1.000 |
  | Exact constrained optimum | 0.376 |

  Clipping is worse than not moving here.
- **Horizon:** 1 vs 20 is indistinguishable at the default `mpc_R = 0.1` (std
  0.390 both). It matters with heavy move suppression: at `mpc_R = 10`, std
  0.436 vs 0.411 on output 1. This is expected for a static-gain plant (L3).

Tests:
- `test_mpc_enforces_move_limits_exactly_not_by_clipping`
- `test_mpc_horizon_and_absolute_bounds_take_effect`
- `test_planner_rejects_unknown_px_weighting`

### KI-5 (Medium, fixed): identified dynamics never reached the simulator

**Before.** The pipeline simulator used a default AR 0.85 and no FF gains. The
stability clip triggered only at \|λ\| ≥ 1.5.

**Fix.**
- `simulator_factory` assembles the plant from C1's `M` and `A`
  (`assemble_state_space`).
- C1 now estimates FF gains by ridge on the counterfactual, with their own
  uncertainty, and the factory passes them.
- `A` is clipped to a stable diagonal (\|a\| ≤ 0.99) whenever \|λ\| ≥ 1.

**After.** FF-gain relative error at `ff_leak = 0.7` is 0.08–0.19 (seeds 0–4).
The pipeline simulator's `A_d` and `FF_gain` equal C1's.

Tests: `test_ff_gains_estimated_from_counterfactual`,
`test_pipeline_propagates_identified_dynamics_and_uncertainty`.

### KI-6 (Medium, fixed): the planner did not target the diagnosed gap

**Before.**
- The gap was computed but ignored.
- The GP was fitted to a zero response with fixed hyperparameters.
- Proposals were in coded units, with uniform cost.

**Fix.**
- `diagnose_gap` returns signed gap directions restricted to adjustable knobs.
  It includes knobs that C1 flags as low-confidence (passed by the pipeline as
  `target_features`).
- The infill acquisition is multiplied by `1 + gap_weight·alignment` (default
  4).
- When responses exist (`y_existing`), GP hyperparameters are fitted by
  marginal likelihood. Existing experiments replace the seed design.
- `CodedSpace` maps to engineering units (`knob_bounds`) and snaps integer and
  categorical levels. The GP uses `mixed_kernel` for categorical factors.
- Proposals carry a cost (`run_cost + Σ knob_cost·|excursion|`), a priority per
  unit cost, their gap alignment and a purpose. When there is no gap, the
  purpose says the runs are optional refinement.

**After** (confounded dataset, 5 planner seeds):

| Case | Mean alignment of proposals with the gap |
|---|---|
| `gap_weight = 0` | 0.47–0.75 |
| `gap_weight = 4` | 0.84–0.90 |
| Single knob flagged by C1 | 0.62–0.85 |

Tests:
- `test_planner_targets_the_gap`
- `test_planner_uses_target_features_and_says_when_there_is_no_gap`
- `test_planner_engineering_units_cost_and_categories`

### KI-7 (Low, fixed except one function): code not wired in or not tested

| Function | Now used by | Test |
|---|---|---|
| NF-05 cleaning | `SensitivityCore`/`FBIdentifier(covariance_cleaning=…)` on the ridge path (opt-in; ignored with a note on the SVD path) | `test_covariance_cleaning_is_wired_into_the_ridge_path` |
| REG-09 trace aggregation | `RegressionMachine.fit` for `EventTable.trace_cols`, with a held-out raw-retention check | `test_traces_aggregated_by_default_and_raw_kept_only_when_needed` |
| VAL-04 `control_off_gold` (A5) | `RegressionMachine` gold holdout (default on) | `test_regression_holds_out_control_off_gold` |
| VAL-03 embargo | `RegressionMachine(embargo=…)`, `forward_validate(embargo=…)` | `test_embargo_drops_training_rows_after_the_test_block` |
| A6 `residual_excitation` | `ExperimentPlanner.calibrate_rejection` | `test_rejection_fraction_calibrated_from_residual_excitation` |
| E11 `mixed_kernel` | planner GP when `categorical_levels` is set | `test_planner_engineering_units_cost_and_categories` |
| `assemble_state_space` | `GatePipeline.simulator_factory` | `test_pipeline_propagates_identified_dynamics_and_uncertainty` |
| REG-07 low variation, OPT-05 unstable rejection | unchanged | `test_regression_routes_low_variation_knob_to_planner`, `test_optimizer_rejects_unstable_configurations` |
| `subspace_order` | **still test-only**: `SensitivityCore` selects rank itself | `test_sensitivity_core_recovers_gain` |

On NF-05: in a closure probe on a collinear synthetic design, the
forward-validated linear shrinkage (`"cv"`) gave a lower gain error than analytic
Ledoit–Péché (0.041 vs 0.107). The analytic forms assume a balanced panel, so
`"cv"` is the recommended setting. Cleaning stays off by default because C1's
default SVD truncation already clips the spectrum.

### KI-8 (Low, fixed): statistical approximations

- **Cpk interval:** uses `n_eff = n_measured · ESS/len` with Geyer's
  initial-positive-sequence effective sample size. For AR(0.7), n = 300, ESS is
  57. `Metrics.n_effective` reports it. The Harris interval remains a block
  bootstrap.
- **ADF:** MacKinnon finite-sample critical values (5%: −2.92 at n = 50 vs
  −2.86 asymptotic) and an approximate p-value.
- **FF leakage:** the per-input p-value is Šidák-corrected over outputs, then
  Holm-corrected across inputs (`pvalue_holm`).

Tests:
- `test_effective_sample_size_discounts_autocorrelation`
- `test_adf_uses_finite_sample_critical_values_and_pvalue`
- `test_ff_leakage_controls_false_positives`

### KI-9 (Low, partly fixed): documentation gaps

**Fixed: §13.5 model-consistency diagnostic.**
- `EventTable.y_pred_cols` holds the controller's own prediction. The simulator
  and the synthetic generator record it.
- `model_consistency` (diagnostics node 2b) works on the prediction error
  `e = y − y_pred`. It controls for the controller's disturbance estimate and
  `e_{t−1}`, then tests whether `e` depends on the knobs, on the FF inputs, or
  only on its own past.
- Verdicts: `M_inaccurate`, `M_or_FF_model_error` (inseparable from this data,
  routed to the planner), `FF_model_error`, `fb_tracking_lag`, `consistent`.
- Calibration on synthetic loops, 10 seeds per case, α = 10⁻³:

  | Case | Verdict |
  |---|---|
  | Healthy and well-tuned loops | consistent 10/10 |
  | Lagging estimator | fb_tracking_lag 10/10 |
  | Gain ×2, ×1.3 and ×0.6 | M_inaccurate 10/10 each |
  | FF leakage | M_or_FF_model_error 10/10 |

- Test: `test_y_pred_consistency_separates_gain_error_from_tracking_lag`.

**Not fixed: the revised specification is not in the repository.** The code
and docs cite §1.4, §14–§16, A1–A6 and E1–E14 from it. The repository holds
only the original `design-requirements-specification.md` (§1–§13), which has
none of these. The document must be supplied by its owner. It cannot be
reconstructed from the code.

### 8.10 Residual issues after the fixes

These are new or remaining in 0.3.0.

| ID | Severity | Issue |
|---|---|---|
| R-1 | Medium | **The gold holdout costs FF-selection power.** Holding control-off episodes out of the C6 fit (VAL-04) removes the rows where FF inputs are least collinear with knob moves. On the walkthrough dataset (`ff_leak = 0.4`), `ff0`'s selection frequency falls from 0.89 to 0.38, so it is no longer reported as a candidate. At `ff_leak = 0.7`, detection is unchanged (8/8 seeds with and without the holdout). The diagnostics FF-leakage node still flags `ff0`. Turn the holdout off (`gold_holdout=False`) only when an honest gold error is not needed. |
| R-2 | Medium | **Gain error and FF-model error can be inseparable.** When knob moves are driven by FF compensation, `model_consistency` returns `M_or_FF_model_error` and routes the question to the planner. Resolving it needs FF-independent excitation. A real gain error combined with FF leakage reads as `M_or_FF_model_error` in 9 of 10 seeds. |
| R-3 | Medium | **The feedback test has blind spots.** It sees feedback only through the logged outputs. A controller reacting to signals that are not in the table, very sparse metrology, or a very gentle controller can pass as open-loop. Those rows would then enter the gain with only the DOE-consistency bias term as a check. |
| R-4 | Low | **With DOE-only identification the bias term is zero.** Drift between the DOE campaign and current operation is not in the reported uncertainty. Use DOE staleness/shift verdicts and the A1 drift monitor. |
| R-5 | Low | **Runtime grew.** The test suite takes ≈19 s (from ≈8 s) and the walkthrough ≈18 s (from ≈6 s), mainly the constrained MPC solve per step. |
| R-6 | Low | `subspace_order` remains exported but test-only (KI-7). |

## 9. Heuristic thresholds (calibrated on synthetic data)

These are engineering defaults, not derived bounds. Recalibrate them on real
data before relying on a verdict.

| Threshold | Default | Where |
|---|---|---|
| Excitation gate (RMS Δu ÷ knob spread) | 0.75 | `diagnostics/precedence.py` |
| Gain-underestimate ratio (realized/model) | 1.3 | `diagnostics/gain_mismatch.py` |
| Oscillation (mean lag-1 autocorrelation) | < −0.3 | `diagnostics/gain_mismatch.py` |
| Gain-drift trend (relative change / correlation), window | 0.2 / 0.6, 60 events | `diagnostics/gain_mismatch.py` |
| `y_pred` consistency (α, lag-1 autocorrelation for tracking lag) | 10⁻³, 0.4 | `diagnostics/consistency.py` |
| Harris "at floor" tolerance | 1.25 | `diagnostics/achievability.py` |
| FF leakage (minimum \|r\|, α after Šidák + Holm) | 0.2, 0.05 | `diagnostics/ff_leakage.py` |
| Phantom-FF envelope, t threshold | 2σ, \|t\| > 2 | `regression_machine/decouple.py` |
| Stability-selection threshold, group correlation | 0.6, \|r\| ≥ 0.8 | `regression_machine/`, `estimation/regularized/` |
| Raw-trace retention margin (held-out \|r\| gain) | 0.1 | `regression_machine/traces.py` |
| Low-variation fraction of median spread | 0.05 | `regression_machine/select.py` |
| Numerical-rank relative tolerance (excitation, planner gap) | 1e-2 | `core/excitation/spectrum.py`, `PlannerConfig.gap_rtol` |
| Feedback test (α, output lags) | 0.01, 2 | `core/excitation/feedback.py`, `DataQualityLayer` |
| Inline down-weight factor | 0.5 | `DataQualityLayer(inline_downweight=…)` |
| Endogenous-fallback bias floor | 1.0 (100%) | `FBIdentifier(endogenous_bias_floor=…)` |
| FF-gain ridge λ | 1.0 | `FBIdentifier(ff_ridge_lambda=…)` |
| Disturbance stability clip | \|λ\| ≥ 1 → diagonal, \|a\| ≤ 0.99 | `estimation/dynamics/state_space.py` |
| Planner gap weight | 4 | `PlannerConfig.gap_weight` |
| Noise-floor quantile, surrogate count | 0.99, 200 (C1) / 500 | `core/noise_floor/` |
| VIF warning, entropy-rank fraction | 10, 0.5 | `core/data_quality/inline.py` |
| DOE staleness horizon; covariate shift (down-weight / reject) | 180 d; 1 / 3 | `core/data_quality/doe.py` |
| `u0` drift span | 1 knob std | `core/counterfactual/baseline.py` |
| Control-off anomaly distance | 3σ | `core/validation/control_off.py` |
| Closed-loop rejection fraction (until calibrated) | 0.6 | `PlannerConfig.rejection_fraction` |
| Unstable-configuration rejection (controlled/uncontrolled std) | 5× | `controller_optimizer/optimize.py` |
| Minimum samples for metrics | 8 | `RunConfig.min_samples` |

## 10. Future work, in priority order

**P1: needed before any real-data use**

1. **Back-test on historical process data.** Replay a period with known
   control-off gold episodes. Compare the following against what actually
   happened:
   - the predicted counterfactual;
   - the feedback-test verdict;
   - the Harris verdict;
   - the `y_pred` consistency verdict;
   - the FF candidates.
2. **Commit the revised specification** so the §1.4, §14–§16, A1–A6 and E1–E14
   citations can be checked (KI-9).
3. **Recalibrate the §9 thresholds on real data**, especially the feedback-test
   α, the consistency α and lag threshold, and the excitation gate. Expose every
   one through `RunConfig`.

**P2: capability**

4. Recover FF-selection power under the gold holdout (R-1). Options: hold out
   only part of the control-off episodes, or estimate FF effects on control-off
   rows in a separate stage.
5. Design FF-independent excitation automatically when the consistency
   diagnostic returns `M_or_FF_model_error` (R-2). Feed it to the planner as a
   target direction.
6. Add a block-bootstrap-over-campaigns bias term for DOE-only identification
   (R-4). Strengthen the feedback test with the recommended-vs-used divergence
   and any logged controller state (R-3).
7. Select hyperparameters (ridge λ, elastic-net λ, PLS components) inside
   forward validation. Support multi-output regression in C6 and the pipeline
   (L6, L7).
8. Replace the variance-share allocation with a counterfactual estimate per
   cause: simulate with each cause removed (L8).
9. Add plant dynamics in the gain (a transfer-function or state-space `M`) if
   carry-over between runs matters (L3).

**P3: productization**

10. Add artifact load (deserialization), data connectors and a CLI. Improve
    performance by vectorizing `_is_measured` / `y_matrix`, running surrogates in
    parallel, and warm-starting the MPC (L4, L5, R-5).
11. Add continuous integration. The repository has no CI checks configured.
12. Remove `subspace_order` or use it for order selection in C1 (R-6).

## 11. Handoff

### 11.1 Run

```bash
pip install -e ".[test]"
pytest                       # 90 tests, ≈19 s
python examples/walkthrough.py
```

If `pytest` fails in `test_diagnostics_renders_figures` with
`cannot import name '_imaging' from 'PIL'`, a system Pillow built for a
different Python is shadowing the venv. Fix it with
`python -m pip install --ignore-installed pillow`.

### 11.2 Reading order

1. `README.md`
2. `docs/architecture.md` (theory, layers, guards)
3. This report, sections 5–8
4. `docs/requirements-traceability.md`
5. `src/r2r_control/pipeline/gates.py` (how the components connect)

### 11.3 Reproducing the closure findings

The probes below were used at the 0.2.0 closure. The comments give the 0.2.0
result and the 0.3.0 result.

```python
import numpy as np
from r2r_control import synthetic
from r2r_control.components import FBIdentifier, Simulator, ControllerConfig
from r2r_control.estimation.sensitivity.core import SensitivityCore

rel = lambda A, B: np.linalg.norm(A - B) / np.linalg.norm(B)

# Probe A (KI-1, KI-2): fused vs DOE-only gain, reported vs actual uncertainty
#   0.2.0: fused 0.099-0.310 vs DOE 0.016-0.048; error/reported 2.4-4.8
#   0.3.0: fused == DOE (gain source "DOE only"); error/reported 0.60-1.39
for seed in range(5):
    for leak in (0.0, 0.4):
        ds = synthetic.make_dataset(seed=seed, doe_age_days=20, ff_leak=leak)
        fb = FBIdentifier(estimator="svd").identify(ds.inline, ds.doe, doe_age_days=20)
        doe = SensitivityCore(estimator="svd").fit(ds.doe.used_knobs() - ds.truth.u0,
                                                   ds.doe.y_matrix())
        print(seed, leak, fb.gain_source, round(rel(fb.M, ds.truth.M), 3),
              round(rel(doe.coef.T, ds.truth.M), 3), round(fb.relative_uncertainty(), 3))

# Probe B (KI-3, KI-4): metrology delay and mpc_horizon
#   0.2.0: delay 0 == delay 1; horizon 1 == horizon 20
#   0.3.0: std rises with delay (0.390 -> 0.427 -> 0.479 on output 1); horizon
#          matters under heavy move suppression (mpc_R=10: 0.436 vs 0.411)
t = synthetic.make_dataset(seed=0).truth
def std(**kw):
    return Simulator(M=t.M, targets=t.targets, lsl=t.targets - 3, usl=t.targets + 3,
                     u0=t.u0, controller=ControllerConfig(**kw), seed=0
                     ).simulate(n_steps=300).metrics_on.std
print(std(metrology_delay=0), std(metrology_delay=1), std(metrology_delay=2))
print(std(mpc_horizon=1, mpc_R=10.0), std(mpc_horizon=20, mpc_R=10.0))
```
