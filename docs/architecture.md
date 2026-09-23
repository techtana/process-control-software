# Architecture & Theory Reference

The self-contained handoff document (NFR-05): theory → workflow →
implementation → validation, so a future session can continue without
re-deriving context.

## The one idea: a noise floor everywhere

Every estimation or control decision asks the same thing: is this variation
signal or noise? Two floors answer it:

- **Estimation floor: Marchenko–Pastur.** Eigenvalues of a pure-noise sample
  correlation matrix stay below `(1 + √q)²`, `q = N/T`
  (`core/noise_floor/analytic.py`). When the balanced-panel assumption fails
  (sparse metrology), the analytic edge is **refused** and an empirical null
  replaces it.
- **Control floor: minimum variance / Harris.** The smallest achievable output
  variance is the variance of the unforecastable delay-step-ahead disturbance.
  `harris_index = achieved / minimum`, and a value near 1 means the controller
  is done.

### The empirical floor must preserve structure (NF-03, E6)

A plain per-column time shuffle kills cross-correlation, which is intended, but
it also kills **autocorrelation**. For autocorrelated or drifting data, that
makes the null too small, so autocorrelated noise gets read as signal.
`core/noise_floor/empirical.py` therefore builds each null from surrogates
that keep each series' marginal distribution, autocorrelation, and missingness
pattern:

- **block permutation** (default): the series is cut into contiguous blocks
  whose order is shuffled. The block logic is shared with the block bootstrap
  (`core/validation/block_bootstrap.py`, VAL-05), so "block" means the same
  thing everywhere.
- **phase randomization**: this keeps the power spectrum exactly.

In the tests, pure AR(1) noise (φ = 0.85) yields a top eigenvalue of about
1.43. The plain-shuffle 99th-percentile floor is about 1.37, which would call
that noise signal. The block and phase floors are about 1.85 and 1.95, which
correctly call it noise. The API rejects a `"shuffle"` surrogate outright.

### Two notions of rank

| Notion | Question | Used for |
|---|---|---|
| **numerical rank** (singular values above a relative gap) | how many directions are excited enough to identify? | model order (FB-02), gap localization (EP-01) |
| **MP signal rank** (eigenvalues above the floor) | how many directions carry common low-rank structure? | high-dimensional screening (REG-03) |

A designed orthogonal experiment has full numerical rank but **no** MP spikes,
so order selection has to use numerical rank.

## Layers and the dependency rule (§16)

```
contracts → config → core → estimation → {controller, metrics} → components → pipeline
```

- **contracts** (`EventTable`, `_is_measured`, shape contract, `ProvenanceLog`,
  `ModelArtifact`) imports nothing else from the package.
- **core** contains the shared services. Each has exactly one implementation (§IF-06).
- **estimation** holds the corrected unification (below).
- **controller** (MHE/MPC) and **metrics** (capability with uncertainty) are used
  by the simulator and diagnostics.
- **components** never import each other. The optimizer receives a *simulator
  factory*, and the pipeline supplies it.
- **pipeline** is the only layer that wires components together.

`tests/test_architecture.py` resolves every relative import in `src/` and fails
the build if an import points upward, or if one component imports another.

## The corrected unification (§1.3, REG-01)

The original design used one `EstimationEngine` with a `mode` flag. That put two
different problems into one class: Component 1 identifies **dynamics** for a
controller, and Component 6 does **high-dimensional selection** for a
predictor. They now share only what they really have in common:

```
                 ┌──────────── SensitivityCore (estimation/sensitivity) ────────────┐
                 │  centered cross-sectional gain · ridge / SVD-truncated solvers   │
                 │  noise-floor identifiability report · parameter covariance       │
                 └───────────────┬──────────────────────────────────┬───────────────┘
                                 │                                  │
      estimation/dynamics (C1 only)                 estimation/regularized (C6 only)
      subspace order · first-order A                PLS · elastic net · group-aware
      state-space assembly                          stability selection (E12)
                                 │                                  │
            components/fb_identification            components/regression_machine
```

Both still route the noise floor, the counterfactual, data quality and
validation through the same `core` services. A test checks this through their
imports: C1 imports `dynamics` but not `regularized`, C6 the reverse, and both
import `sensitivity`.

## Closed-loop confounding

Passive inline data reflects the controller's corrections, not the open-loop
process. The system:

1. **quantifies** it with VIF, condition number, and numerical-rank deficiency;
2. **exploits** accidental excitation (overrides, control-off episodes);
3. **requests** deliberate excitation by routing adjustable gaps to the planner.

Two new places where this confounding comes back are now guarded:

- **inside experiments (A6/E9).** A closed loop cancels part of a designed
  perturbation. `PlannerConfig.loop_status` / `rejection_fraction` discount the
  information credited to each experiment (`experiment_planner/loop_status.py`),
  and `residual_excitation` measures the surviving fraction after the fact.
- **inside diagnostics (DIAG-07/E4).** Tight control drives Δu to about 0, so
  realized gain is undefined. See the precedence tree below.

## Diagnostic precedence (DIAG-07, E4, E5)

The diagnostics have overlapping signatures. Over-aggressive correction can come
from an underestimated gain, a hot Controller QR, or an over-trusting State QR.
They therefore run as a decision tree (`diagnostics/precedence.py`):

1. **Excitation gate.** Compute RMS per-event knob move ÷ knob operating spread.
   Rate-limited loops score 0.25–0.59, actively correcting loops 0.95–2.7, and
   over-correcting loops 2–4. The floor is 0.75.
2. **Gain mismatch** runs only if the gate passes. Otherwise it returns
   *"unidentifiable from available data — excitation required"* and the
   question is routed to the planner.
3. **FF leakage** needs no knob movement.
4. **QR attribution** runs only if the gate passes **and** no gain mismatch was
   found. With a gain mismatch it is *deferred* (E5).
5. **Variance decomposition.** Blocked nodes contribute nothing; their share
   stays "unattributed recoverable".
6. **Achievability verdict** (Harris).

## Load-bearing assumptions (§1.4)

Every `ModelArtifact` records which assumptions it relies on
(`contracts/artifact.py::LOAD_BEARING_ASSUMPTIONS`), and each assumption has a
guard:

- **A1 gain time-invariance.** `gain_drift_monitor` estimates realized gain in
  half-overlapping windows and regresses its magnitude on time. A strong,
  consistent trend means re-identify.
- **A2 `u0` validity.** `estimate_baseline` compares control-off episode
  baselines over time. A fitted change of more than one knob standard deviation
  flags drift, gives a re-baseline horizon, and can return an interpolated
  time-varying `u0`, which `reconstruct` accepts.
- **A3 `M` accuracy.** In `decouple`, for each FF sensor, the apparent effect
  `b_j` on the decoupled target is compared with the largest slope an `M` error
  could create along the knob directions the sensor loads on,
  `2·rel_unc·‖M_row‖·‖a_j‖`. A sensor whose effect is significant but inside
  that envelope is **quarantined** instead of reported as a feedforward
  candidate. Real FF disturbances sit 3–40× outside the envelope in the tests.
- **A4 delay < horizon.** `route_measurement` sends over-horizon measurements to
  `SlowBiasEstimator` instead of the Kalman state update.
- **A5 representative control-off.** `control_off_gold` drops episodes whose
  tool state is more than 3σ from normal operation.
- **A6 excitation lands.** See the closed-loop excitation discount above.

## Finite-sample metrics (SIM-06, E10)

Synthetic simulator output is plentiful, but real metrology is sparse. With
`finite_sample_ci=True`, `metrics.compute_metrics` attaches a large-sample Cpk
interval (`Var ≈ 1/(9n) + Cpk²/(2(n−1))`) and a block-bootstrap Harris
interval. Any output below `min_samples` effective points is suppressed and
reported as "insufficient samples". This is separate from the model-uncertainty
Monte Carlo bands (SIM-02).

## Workflow

```
DOE + inline ─ DataQualityLayer ─► FBIdentifier ─► FBModel / ModelArtifact (M, cov, A, u0, assumptions)
                     │                                   │
                     │ gaps                              │ M + uncertainty
                     ▼                                   ▼
             ExperimentPlanner ◄── identifiable dirs ─ Simulator(factory) ◄── ControllerOptimizer
                                                         │
                                                         ▼
                                              DiagnosticsEngine (precedence tree)

all data ─ RegressionMachine ─► FF candidates (phantoms quarantined), low-variation gaps ─► planner
```

`pipeline/gates.py::GatePipeline` runs this sequence from one `RunConfig`.

## Open design decisions (§13), kept as configuration

| Decision | Where |
|---|---|
| `u0` baseline and time-varying option | `RunConfig.allow_time_varying_u0`, `FBIdentifier.identify(u0=…)` |
| Spec inputs + minimum-sample floor | `RunConfig.targets/lsl/usl/min_samples` |
| `M` uncertainty | `RunConfig.rel_uncertainty_M` (MC bands and phantom-FF envelope) |
| Planner objective, loop status, budget, integer dims | `RunConfig` → `PlannerConfig` |
| Regression target raw vs innovation | `RunConfig.regression_target` |
| Noise-floor choice | `noise_floor.estimate` picks analytic or empirical and records why |
