# Project Closure Report — `r2r_control` 0.2.0

**Status:** closed on 2026-10-24.
**Final code:** `main` at merge commit `1fc041f` (PR #1, head `e674dac`).
**This report** adds documentation only. No algorithm was changed at closure.

Read this before using any output on real process data. It states what the
toolchain assumes, where it applies, how it works, what it cannot do, the
defects found during closure, and what to do next.

---

## 1. Summary

`r2r_control` is an offline Python toolchain for run-to-run (R2R) process
control. It identifies the feedback gain, plans experiments, simulates and tunes
an estimator/controller, diagnoses residual variance, and screens feedforward
(FF) candidates. Every decision is framed as *signal versus a noise floor*:

- the Marchenko–Pastur edge, or a structure-preserving surrogate floor, for estimation;
- the Harris minimum-variance benchmark for control.

At closure all 64 tests pass and the end-to-end walkthrough runs (§3). The
architecture, the refusal behavior and the assumption guards work as specified
**on synthetic data**.

The closure review found three issues that matter most before real-data use
(§8 has the full list):

1. **The fused gain estimate is biased by closed-loop inline data (KI-1).** On
   the synthetic benchmark, the DOE-only gain error is 2–5%, but the shipped
   DOE + inline fusion gives 10–31%.
2. **The reported gain uncertainty is 2.4–4.8× too small (KI-2).** Monte-Carlo
   bands and the phantom-FF envelope inherit this optimism.
3. **The simulator does not delay measurements (KI-3).** `metrology_delay` only
   switches where a measurement is routed, so delay effects on control
   performance are not simulated.

## 2. Deliverables

| Item | Location |
|---|---|
| Python package (six components plus shared services) | `src/r2r_control/` (≈5,000 lines) |
| Test suite (64 tests: unit, injected-fault, refusal, architecture, integration) | `tests/` |
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
| `pytest` | **64 passed** (63 test functions; one runs for both surrogate types) in ≈8 s |
| `pytest -o filterwarnings=default -rw` | 64 passed, **no warnings** with numpy 2 / pandas 3 |
| `python examples/walkthrough.py` | completes in ≈6 s; every guard fires as intended |
| Closure probes (§11.3) | reproduce KI-1 to KI-4 |

The package declares `requires-python >=3.9`. Only 3.11 was re-checked at
closure.

**What the tests establish:**
- the layering and decoupling rules hold;
- each guard and refusal fires on the fault it targets;
- the shared services behave correctly on constructed cases;
- the pipeline is reproducible and serializable.

**What they do not establish:**
- **Real-world accuracy.** Every dataset comes from `synthetic.py`, which shares
  the methods' structural assumptions (L1).
- **Gain-estimate quality.** Several tolerances are loose. The FB gain test
  accepts up to 30% relative error, which hides KI-1.

## 4. Methodology

| Stage | Method | Module |
|---|---|---|
| Data model | Event-indexed table with recommended and used knobs, FF inputs, and sparse array-valued metrology. A single presence predicate `_is_measured` handles all-NaN arrays. | `contracts/` |
| Data quality | DOE: alias/resolution parsing, staleness decay (180 d horizon), covariate-shift distance (down-weight above 1, reject above 3). Inline: VIF, condition number, entropy rank, ADF/KPSS/variance-ratio drift tests, effective measured count. Each check emits a use / down-weight / reject decision with its rationale. | `core/data_quality/` |
| Noise floor | Balanced panel: MP edge `(1+√(N/T))²` on the correlation matrix. Any missing cells: the 99th percentile of the top eigenvalue under block-permutation (default) or phase-randomization surrogates, which keep each series' marginal, autocorrelation and missingness. A plain shuffle is refused. | `core/noise_floor/` |
| Excitation | Numerical rank (singular values above 1% of the largest) for identifiability, and MP signal rank for low-rank structure. Unidentifiable directions are expressed in knob/FF terms. Overrides and control-off episodes are mined as accidental excitation. | `core/excitation/` |
| Counterfactual | `y_nocontrol = y − M(u_used − u0)`. `u0` comes from control-off episodes, with a drift check and an optional interpolated time-varying `u0`. Monte-Carlo bands from relative uncertainty on `M` and `u0`. | `core/counterfactual/` |
| Validation | Forward chaining with purge and embargo (naive k-fold is refused), block bootstrap, control-off gold holdout with a representativeness check, concept-drift slope test. | `core/validation/` |
| C1 FB identification | Weighted fusion of DOE and inline rows, then SVD-truncated least squares at the numerical rank (ridge as an alternative). Parameter covariance from the truncated SVD. First-order disturbance `A` by least squares on the counterfactual. Ljung–Box innovation check. FB-08 refusal recorded. | `components/fb_identification/`, `estimation/sensitivity/`, `estimation/dynamics/` |
| C2 Experiment planner | Gap diagnosis by VIF and rank. D/I-optimal seed design by coordinate exchange over a 400-point random pool. GP sequential infill by ALC/IMSE/EIG. Information credited to an experiment is discounted by `1 − rejection_fraction` when the loop is closed (A6). Integer rounding (E11). Manual approval required. | `components/experiment_planner/`, `core/gp.py` |
| C3 Simulator | Plant `y = d + FF·g + M(u−u0) + e`, with `d` = AR + random-walk drift. Kalman-filter disturbance estimator (State QR). One-step static-gain LQ move (Controller QR) with a move limit. Over-horizon measurements go to a slow-bias estimator. Monte-Carlo over multiplicative `M` perturbation. Metrics with and without control. | `components/simulator/`, `controller/`, `metrics/` |
| C4 Optimizer | Robust objective = mean output std + λ·(p95 − p05 Monte-Carlo band). A configuration is rejected if its controlled std exceeds 5× the uncontrolled std. Random, grid, Nelder–Mead, evolutionary or GP-EI Bayesian search over a simulator factory. Finite-difference sensitivity per tuning dimension. | `components/controller_optimizer/` |
| C5 Diagnostics | Decision tree: excitation gate (RMS Δu ÷ knob spread above 0.75) → gain mismatch (realized/model gain ratio above 1.3 plus lag-1 ringing) → FF leakage (Δstate vs ΔFF correlation) → QR attribution (innovation whiteness, then move/track ratio) → variance-share allocation → Harris verdict (mean index ≤ 1.25 means at floor). Blocked nodes return "unidentifiable — excitation required". | `components/diagnostics/` |
| C6 Regression machine | Decouple the controller via the counterfactual. Phantom-FF quarantine: a sensor is quarantined if `|t| > 2` and its slope lies within `2·rel_unc·‖m_row‖·‖a_j‖`. Ridge, PLS or elastic net is chosen by 4-fold forward chaining. Elastic-net stability selection over 100 block-bootstrap resamples, reported per feature and per correlated group. Low-variation inputs are routed to the planner. | `components/regression_machine/`, `estimation/regularized/` |
| Metrics | Cpk, std, %OOS, mean off target, Harris. Optional finite-sample intervals: Cpk with `Var ≈ 1/(9n) + Cpk²/(2(n−1))`, Harris by block bootstrap. Minimum-sample gate (default 8). | `metrics/` |
| Orchestration | `GatePipeline` runs identify → plan → simulate (→ optimize) → diagnose → regress from one `RunConfig`, with full provenance. | `pipeline/` |

## 5. Assumptions

### 5.1 Load-bearing assumptions with guards (§1.4)

Each guard detects the stated violation. The right-hand column says what it
does **not** catch.

| ID | Assumption | Guard | What the guard does not catch |
|---|---|---|---|
| A1 | Gain `M` is time-invariant (drift is additive) | Realized-gain trend monitor | Fast or step changes inside one 60-event window; drift masked by tight control (realized gain undefined) |
| A2 | `u0` is valid over the analysis horizon | Control-off baseline trend, validity horizon, time-varying `u0` | Needs at least 2 control-off episodes. With none it falls back to the global knob mean and only warns. |
| A3 | `M` is accurate enough to decouple | Monte-Carlo propagation plus phantom-FF quarantine | The envelope uses the **configured** `rel_uncertainty_M` (default 0.1), not the identified uncertainty. Actual error is often larger (KI-1, KI-2). |
| A4 | Metrology delay is shorter than the MHE horizon | Over-horizon measurements routed to a slow-bias estimator | Routing only. The simulator applies no actual delay (KI-3). |
| A5 | Control-off episodes are representative | Tool-state distance from normal operation above 3σ excludes an episode | Needs a tool-state signal. The check is not wired into C1/C6 (KI-7). |
| A6 | Injected excitation reaches the process | Closed-loop discount on information gain; `residual_excitation` | The discount is a configured constant (0.6 rejection), not measured. `residual_excitation` is not called anywhere (KI-7). |

### 5.2 Modelling assumptions without a guard

These hold implicitly throughout. Nothing in the code detects their violation.

| ID | Assumption | Consequence if violated |
|---|---|---|
| M1 | **Linear** gain: `Δy = M Δu` over the operating range. No curvature or interactions in the FB model. | Gain varies with operating point. DOE "curvature" information is unused. |
| M2 | **Superposition:** output = disturbance + FF + control + noise, with the disturbance independent of knob settings | The counterfactual and the decoupling are wrong. |
| M3 | Rows used for gain identification are **exogenous** (knob moves uncorrelated with the disturbance) | True for DOE, overrides and control-off episodes. **False for closed-loop inline events**, so the fused gain is biased (KI-1). |
| M4 | The disturbance is roughly first-order AR plus random-walk drift; a low-order AR gives the minimum-variance benchmark | The Harris benchmark and simulated performance are mis-stated for richer disturbances. |
| M5 | **Run-to-run:** one event is one run, the process settles within a run, and there are no within-run dynamics | Unsuitable for continuous or dynamic processes. |
| M6 | Outputs are roughly normal and serially independent (for Cpk intervals) | Intervals are too narrow on autocorrelated data (KI-8). |
| M7 | Simulation treats the identified `M` as the true plant. Model error is only i.i.d. multiplicative noise of size `rel_uncertainty_M`. | Structured or biased plant–model mismatch is not represented. |
| M8 | Missing metrology is ignorable (not caused by the process state) | Surrogate floors and metrics are biased if, say, only bad lots are measured. |
| M9 | DOE factors are coded around `u0`, and the planner works in coded units of [−1, 1] | Proposals must be mapped to engineering units and limits by a person (KI-6). |

### 5.3 Decisions that belong to a person (§13)

These are configuration, not hidden defaults. Each needs confirming per process.

| Decision | Where | Default at closure |
|---|---|---|
| `u0` is the true open-loop baseline | `FBIdentifier.identify(u0=…)`, `RunConfig.allow_time_varying_u0` | Control-off mean; static |
| Targets and spec limits | `RunConfig.targets/lsl/usl` | Required, no default |
| Minimum sample count for metrics | `RunConfig.min_samples` | 8 (low; 30 or more is advisable) |
| Relative uncertainty on `M` | `RunConfig.rel_uncertainty_M` | 0.1 (below the error actually observed; see KI-2) |
| Planner objective, loop status, budget, integer dimensions | `RunConfig` → `PlannerConfig` | D-optimal, open loop, 8 runs |
| Regression target (raw or innovation) | `RunConfig.regression_target` | raw |
| Consistency diagnostic using `y_pred` (§13.5) | not implemented | see KI-9 |

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
- Hundreds of sensors or raw traces without aggregation (scale is untested; L5).
- Real-time control, automatic actuation, or safety-critical use (L4).

**Data prerequisites:**
- An event table with recommended and used knobs, FF inputs, and timestamped
  outputs.
- Targets and spec limits.
- At least 2 control-off episodes for the `u0` drift check.
- **A DOE, or override/control-off excitation, for the gain.** Closed-loop inline
  data alone cannot identify it (DQ-06, KI-1).
- Enough measured events per output for metrics. The gate is 8; meaningful Cpk
  needs far more.

## 7. Limitations (by design or scope)

| ID | Limitation |
|---|---|
| L1 | **Validated on synthetic data only.** The generator encodes M1, M2, M4 and M7, so the tests check self-consistency, not real-world validity. No back-test on historical process data has been done. |
| L2 | Linear static-gain FB model with no curvature, interactions or operating-point dependence (M1). |
| L3 | The controller model is one-step and static-gain. The disturbance model is first order. Neither suits processes with run-to-run carry-over dynamics in the gain. |
| L4 | A library only: no data connectors, CLI, UI, or service. `ModelArtifact` serializes (`to_dict`/`to_json`) but cannot be loaded back. |
| L5 | Scale is untested beyond the test sizes. Cell-by-cell Python loops (`y_matrix`, `_is_measured`), an empirical floor costing O(n_perm·N·T), and coordinate exchange costing O(runs·pool·passes) will be slow on large data. |
| L6 | Fixed hyperparameters: ridge λ = 1, elastic-net λ = 0.1, PLS at most 5 components, fixed GP length scales. None are tuned inside forward validation. |
| L7 | The regression machine fits one output per call, and `GatePipeline` regresses only the first output. |
| L8 | Variance decomposition is an **allocation rule**: the recoverable share is split equally among the causes the tree flagged. It does not estimate each cause's contribution. |
| L9 | The FB-08 refusal is structural: it is always emitted. Separating gain error from estimator tuning always needs active excitation or the estimator specification, as the spec intends. |
| L10 | Decision thresholds are heuristics calibrated on synthetic loops (§9). |
| L11 | Only Python 3.11 was verified at closure; the declared minimum is 3.9. |

## 8. Known issues (found at closure)

Severity: **High** means it can mislead a real-data decision. **Medium** means a
specified capability is weaker than its name suggests. **Low** covers gaps,
hygiene and documentation.

### KI-1 (High): Closed-loop inline data biases the fused gain estimate

Relative gain error `‖M̂ − M‖/‖M‖` on the synthetic benchmark (§11.3, probe A):

| seed | ff_leak | fused (shipped) | inline only | DOE only | inline verdict / weight used |
|---:|---:|---:|---:|---:|---|
| 0 | 0.0 | 0.127 | 0.783 | 0.039 | down_weight / 1.00 |
| 0 | 0.4 | 0.290 | 1.365 | 0.039 | use / 1.00 |
| 1 | 0.0 | 0.190 | 0.608 | 0.048 | use / 1.00 |
| 1 | 0.4 | 0.310 | 0.837 | 0.048 | use / 1.00 |
| 2 | 0.0 | 0.120 | 0.395 | 0.016 | use / 1.00 |
| 2 | 0.4 | 0.210 | 0.750 | 0.016 | use / 1.00 |
| 3 | 0.0 | 0.099 | 0.532 | 0.032 | use / 1.00 |
| 3 | 0.4 | 0.184 | 0.798 | 0.032 | use / 1.00 |
| 4 | 0.0 | 0.128 | 0.683 | 0.040 | down_weight / 1.00 |
| 4 | 0.4 | 0.278 | 1.158 | 0.040 | down_weight / 1.00 |

Adding inline data makes the gain 3–8× worse than DOE alone.

**Causes:**
- **The down-weight verdict is not applied.** `core/data_quality/inline.py`
  returns its `QualityReport` without a weight, so the weight stays 1.0 even
  under a `down_weight` verdict. `identify.py` then weights closed-loop rows
  equally with DOE rows. This breaks DQ-07 ("weight of every contribution").
- **The checks cannot see endogeneity.** VIF and rank detect *collinearity*. The
  closed-loop problem is *endogeneity*: the controller moves knobs in response
  to the disturbance (M3). Most cases therefore pass as "use". This conflicts
  with DQ-06, under which inline data should supply only local correction and
  noise where DOE supplies gain.
- **The test is too loose to catch it.** `test_fb_identification_recovers_gain`
  accepts up to 30% error, and seed 1 with `ff_leak = 0.4` already reaches 31%.

**Impact:** the gain feeds the counterfactual, the decoupling, the simulator and
the diagnostics. All of them inherit the bias.

**Suggested fix:**
- Identify the gain from DOE plus *exogenous* inline rows only (overrides and
  control-off episodes).
- Use the remaining inline data for an operating-point bias correction and the
  noise model.
- Honor verdict weights.
- Add an endogeneity check, such as the correlation of knob moves with the
  lagged innovation.
- Tighten the test to roughly 2× the DOE-only error.

### KI-2 (High): Reported gain uncertainty is far smaller than the actual error

Across the ten runs above, the actual error divided by
`FBModel.relative_uncertainty()` is 2.4–4.8 (median 3.7). The reported values
are 0.037–0.070, while the actual errors are 0.099–0.31.

**Cause:** `param_cov` captures variance only, not the closed-loop bias of KI-1.

**Also:** `GatePipeline` never propagates the identified uncertainty. It uses
the configured `rel_uncertainty_M = 0.1`. As a result, the simulator's
Monte-Carlo bands (SIM-02), the counterfactual bands (CF-03) and the phantom-FF
envelope (A3) are all roughly 2× too narrow in this benchmark.

**Suggested fix:**
- Fix KI-1.
- Estimate the uncertainty with a bias term, for example the discrepancy
  between the DOE-only and fused estimates, or a block bootstrap over campaigns.
- Pass it to `rel_uncertainty_M` by default.

### KI-3 (Medium): The simulator does not model metrology delay

In `components/simulator/run.py`, the measurement `y_t` is used at step `t`
whatever `metrology_delay` is. The delay only selects the route: state update if
the delay is no longer than `mhe_horizon`, slow bias otherwise.

Probe B: delay 0 and delay 1 give identical results. With delay 3 against a
horizon of 1, every update goes to the slow-bias estimator, which is not
equivalent to a late measurement.

The A4/E7 test checks routing only. Simulated performance for delayed metrology
is therefore not representative.

**Suggested fix:** add a delay buffer that releases the innovation `delay` steps
late, and perform a true moving-horizon re-estimate for late measurements that
still fall within the horizon.

### KI-4 (Medium): Configuration fields that do nothing, and a simplified MHE/MPC

- `ControllerConfig.mpc_horizon` is never read. Probe B shows horizon 1 and 20
  give identical results. With a static gain and no model dynamics, the
  "finite-horizon" MPC reduces to a one-step LQ move.
- `PlannerConfig.px_weighting` is never read.
- `mhe_horizon` is used only for routing. The "MHE" is a one-step Kalman filter
  (`KalmanMHE`) with no constraints.
- The move limit is applied by clipping after the unconstrained solve. That is
  not the constrained optimum when several knobs hit their limits.
- There are no absolute knob bounds.

**Suggested fix:** either implement a multi-step MPC as a constrained QP with
bounds and a horizon MHE, or remove the inert fields and rename the components
for what they are.

### KI-5 (Medium): The identified dynamics never reach the simulator

- `FBModel.A` is identified, but `GatePipeline.simulator_factory` builds a
  default `PlantConfig` (diagonal AR 0.85) and passes no FF gains. In pipeline
  runs the FF-leakage branch therefore has nothing to detect.
- The stability clip in `identify_first_order_dynamics` only triggers at
  `max|eig| ≥ 1.5`, so a mildly unstable `A` (1 ≤ |λ| < 1.5) passes unchanged.

**Suggested fix:** pass `A_d = fb.A` and the identified or configured FF gains
to the simulator, and clip whenever `max|eig| ≥ 1`.

### KI-6 (Medium): The planner does not target the diagnosed gap

- `ExperimentPlanner.plan` computes a `GapReport` but designs over all
  adjustable knobs regardless of it. The walkthrough reports
  "0 adjustable directions unidentifiable" and still proposes 5 experiments.
- The infill GP is fitted to a zero response with fixed hyperparameters, so ALC
  acts as a space-filling criterion.
- Proposals are in coded units with no mapping to engineering units or process
  limits.
- Every experiment is given the same cost (1.0).

**Suggested fix:**
- Project candidates onto the unidentifiable directions.
- Fit the GP to the observed responses.
- Add unit mapping, feasibility constraints and a cost model.

### KI-7 (Low): Code that is implemented but not wired in or not tested

| Function | Wired into a component? | Dedicated test? |
|---|---|---|
| NF-05 `eigenvalue_clip`, `ledoit_peche_shrinkage` | no | no |
| REG-09 `aggregate_trace`, `aggregation_destroys_signal` | no | no |
| VAL-04 `control_off_gold` (A5) | no (standalone) | yes |
| VAL-03 `apply_embargo` | no (C6 uses purge = 1, embargo = 0) | no |
| A6 `residual_excitation` | no | no |
| E11 `mixed_kernel` | no (planner GP uses plain RBF) | yes |
| `assemble_state_space`, `subspace_order` | no (`subspace_order` is used only in tests) | partly |
| REG-07 `low_variation`, OPT-05 unstable-configuration rejection | yes | no dedicated assertion |

**Suggested fix:** wire each in where the specification calls for it and add
tests, or delete it.

### KI-8 (Low): Statistical approximations

- **Cpk and Harris intervals** use the raw measured count as `n`, with no
  discount for autocorrelation. The Cpk interval is the normal-theory formula.
  Intervals are therefore too narrow on autocorrelated outputs.
- **ADF** uses asymptotic critical values with no small-sample adjustment or
  p-value.
- **FF leakage** takes the largest |r| over outputs and tests it as a single
  correlation, with no multiple-comparison correction.

### KI-9 (Low): Documentation gaps

- **The revised specification is not in the repository.** The code and docs
  cite §1.4, §14–§16, A1–A6 and E1–E14 from it, but the repo holds only
  the original `design-requirements-specification.md` (§1–§13), which has none
  of these.
- **§13.5 is not implemented.** That item asks for a model-consistency
  diagnostic using a `y_pred` column, and `EventTable` has no such column.

## 9. Heuristic thresholds (calibrated on synthetic data)

These are engineering defaults, not derived bounds. Recalibrate them on real
data before relying on a verdict.

| Threshold | Default | Where |
|---|---|---|
| Excitation gate (RMS Δu ÷ knob spread) | 0.75 | `diagnostics/precedence.py` |
| Gain-underestimate ratio (realized/model) | 1.3 | `diagnostics/gain_mismatch.py` |
| Oscillation (mean lag-1 autocorrelation) | < −0.3 | `diagnostics/gain_mismatch.py` |
| Gain-drift trend (relative change / correlation), window | 0.2 / 0.6, 60 events | `diagnostics/gain_mismatch.py` |
| Harris "at floor" tolerance | 1.25 | `diagnostics/achievability.py` |
| FF leakage (minimum \|r\|, α) | 0.2, 0.05 | `diagnostics/ff_leakage.py` |
| Phantom-FF envelope, t threshold | 2σ, \|t\| > 2 | `regression_machine/decouple.py` |
| Stability-selection threshold, group correlation | 0.6, \|r\| ≥ 0.8 | `regression_machine/`, `estimation/regularized/` |
| Low-variation fraction of median spread | 0.05 | `regression_machine/select.py` |
| Numerical-rank relative tolerance | 1e-2 | `core/excitation/spectrum.py` |
| Noise-floor quantile, surrogate count | 0.99, 200 (C1) / 500 | `core/noise_floor/` |
| VIF warning, entropy-rank fraction | 10, 0.5 | `core/data_quality/inline.py` |
| DOE staleness horizon; covariate shift (down-weight / reject) | 180 d; 1 / 3 | `core/data_quality/doe.py` |
| `u0` drift span | 1 knob std | `core/counterfactual/baseline.py` |
| Control-off anomaly distance | 3σ | `core/validation/control_off.py` |
| Closed-loop rejection fraction | 0.6 | `experiment_planner` (`PlannerConfig`) |
| Unstable-configuration rejection (controlled/uncontrolled std) | 5× | `controller_optimizer/optimize.py` |
| Minimum samples for metrics | 8 | `RunConfig.min_samples` |

## 10. Future work, in priority order

**P1: needed before any real-data use**

1. **Fix KI-1 and KI-2.**
   - Identify the gain from exogenous rows only, and use the remaining inline
     data as a local correction.
   - Apply verdict weights and add an endogeneity check.
   - Propagate an uncertainty that includes bias.
   - Tighten the FB test.
2. **Back-test on historical process data.** Replay a period with known
   control-off gold episodes, then compare the predicted counterfactual,
   Harris verdict and FF candidates against what actually happened.
3. **Make the simulator honest about time (KI-3, KI-4).**
   - Add a measurement-delay buffer and a true horizon MHE.
   - Use a multi-step constrained-QP MPC with absolute bounds, or remove the
     inert fields.

**P2: capability**

4. Feed the identified `A` and FF gains into the pipeline's simulator, and
   correct the stability clip (KI-5).
5. Target the planner at the diagnosed gap, fit the GP to observed responses,
   map to engineering units and limits, add a cost model, and use `mixed_kernel`
   for categorical factors (KI-6).
6. Wire the VAL-04 gold holdout and the embargo into C1/C6 validation. Use NF-05
   cleaning for covariance estimates. Use REG-09 trace aggregation. Test or
   delete each unwired function (KI-7).
7. Select hyperparameters inside forward validation, and support multi-output
   regression in C6 and the pipeline (L6, L7).
8. Use an autocorrelation-aware effective sample size and bootstrap intervals
   for Cpk, and control multiple testing in FF leakage (KI-8).
9. Replace the variance-share allocation with a counterfactual estimate per
   cause: simulate with each cause removed (L8).

**P3: productization**

10. Add artifact load (deserialization), data connectors and a CLI. Improve
    performance by vectorizing `_is_measured` / `y_matrix` and running
    surrogates in parallel (L4, L5).
11. Add continuous integration. The repository has no CI checks configured;
    PR #1 merged without any.
12. Commit the revised specification and implement the §13.5 `y_pred`
    diagnostic (KI-9).
13. Recalibrate the §9 thresholds on real data and expose every one through
    `RunConfig`.

## 11. Handoff

### 11.1 Run

```bash
pip install -e ".[test]"
pytest                       # 64 tests, ≈8 s
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

```python
import numpy as np
from r2r_control import synthetic
from r2r_control.components import FBIdentifier, Simulator, ControllerConfig
from r2r_control.estimation.sensitivity.core import SensitivityCore

rel = lambda A, B: np.linalg.norm(A - B) / np.linalg.norm(B)

# Probe A (KI-1, KI-2): fused vs DOE-only gain, reported vs actual uncertainty
for seed in range(5):
    for leak in (0.0, 0.4):
        ds = synthetic.make_dataset(seed=seed, doe_age_days=20, ff_leak=leak)
        fb = FBIdentifier(estimator="svd").identify(ds.inline, ds.doe, doe_age_days=20)
        doe = SensitivityCore(estimator="svd").fit(ds.doe.used_knobs() - ds.truth.u0,
                                                   ds.doe.y_matrix())
        print(seed, leak, round(rel(fb.M, ds.truth.M), 3),
              round(rel(doe.coef.T, ds.truth.M), 3), round(fb.relative_uncertainty(), 3))

# Probe B (KI-3, KI-4): metrology delay and mpc_horizon have no effect on results
t = synthetic.make_dataset(seed=0).truth
def std(**kw):
    return Simulator(M=t.M, targets=t.targets, lsl=t.targets - 3, usl=t.targets + 3,
                     u0=t.u0, controller=ControllerConfig(**kw), seed=0
                     ).simulate(n_steps=300).metrics_on.std
print(std(metrology_delay=0), std(metrology_delay=1))   # identical
print(std(mpc_horizon=1), std(mpc_horizon=20))          # identical
```
