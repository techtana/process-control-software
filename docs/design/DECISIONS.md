# Design decisions

A running log, newest first. Each entry gives the date, the decision, why, any
rejected alternative, and where it is recorded. Entries are never deleted; a
reversed one is marked "Superseded". Dates are commit or PR-merge dates.

Sources: the spec (`design-requirements-specification.md` §13), `CHANGELOG.md`,
[`docs/architecture.md`](../architecture.md),
[`docs/project-closure.md`](../project-closure.md), PR #1, PR #2 and commit
messages.

## 2026-10-26: review follow-up (branch `review-fixes-and-readme`)

- **Rebuild the review fixes on top of 0.3.0, and keep only what 0.3.0 does not
  already fix.** This was the owner's decision on 2026-10-26, after PR #2 was
  found to overlap. 0.3.0 already covered the review's gain-bias and
  inline-confidence findings, so those versions were dropped.
  Plan: [2026-10-25 code review](plans/2026-10-25-code-review-and-fixes.md).
- **Choose feedforward candidates from the FF sensors alone, and require a
  Bonferroni-corrected correlation test.** With the knob columns as
  co-predictors, unrelated sensors were selected purely to cancel the FF part of
  the controller's knob moves. Rejected: knobs as co-predictors. Commit
  `9a2a055`.
- **Drop the ARX realized-gain fix.** On 0.3.0's receding-horizon MPC, a
  one-lag ARX regression gave meaningless ratios (15.9 on a healthy loop) and
  broke three tests. The feedback bias in the gain-mismatch check is still open.
- **Read source files as UTF-8 in the tests.** The default Windows code page
  could not decode the 0.3.0 sources. Commit `9a2a055`.

## 2026-10-24: 0.3.0 closure fixes (PR #2)

Full evidence for each entry is in [project-closure.md §8](../project-closure.md).

- Exclude closed-loop inline rows from gain identification, using a feedback
  test (alpha 0.01, Bonferroni over the knobs). Fused gain error was 10–31%,
  against 2–5% for the DOE alone. Rejected: fusing closed-loop rows (the 0.2.0
  behaviour).
- Without a DOE, keep the closed-loop fit but flag it with a 100% bias floor and
  mark every knob low-confidence. No clean gain source exists in that case.
- `down_weight` verdicts carry a weight of 0.5. Before this, the verdict had no
  effect.
- Gain uncertainty is `hypot(variance, DOE-vs-inline bias)`.
  `rel_uncertainty_M` defaults to the identified value. The fixed 0.1 was 2.4–4.8
  times too small.
- Propagate the identified uncertainty to the simulator, the Monte Carlo bands
  and the phantom-FF envelope. Overriding it with a lower value logs a warning.
- Release measurements `metrology_delay` steps late. Late measurements that
  still fall within the MHE horizon are replayed through the Kalman time
  updates. Before this, the delay had no effect.
- Use a receding-horizon MPC with exact move limits (bounded least squares) and
  optional absolute bounds (SLSQP). Rejected: clipping the unconstrained move,
  which could cost more than not moving at all.
- The simulator uses component 1's `A`, FF gains and uncertainty. The stability
  clip now triggers at |λ| ≥ 1 (it was 1.5).
- The planner targets the gap directions and component 1's low-confidence knobs
  (`gap_weight`). It fits GP hyperparameters by marginal likelihood, works in
  engineering units, snaps integer and categorical inputs, has a cost model, and
  can calibrate the rejection fraction from measurements.
- Hold out control-off episodes as gold by default (VAL-04), and make the
  embargo configurable (VAL-03). The cost is lower FF-selection power, recorded
  as residual issue R-1.
- Cpk interval uses an autocorrelation-discounted `n_eff`. ADF uses
  finite-sample critical values. FF leakage applies Šidák across outputs and
  Holm across inputs.
- Covariance cleaning (NF-05) is opt-in, and `"cv"` is recommended over analytic
  Ledoit–Péché. The analytic forms assume a balanced panel.
- Add a model-consistency check based on `y_pred` (spec §13.5) that separates an
  inaccurate M from FB tracking lag.
- Harris delay = metrology delay + 1.
- Scope: an offline decision-support tool on synthetic data only, not for
  real-time or safety-critical use (closure §1, §6).

## 2026-09-23: 0.2.0 re-architecture (PR #1, merged 2026-10-12)

- **Split the estimator.** Component 1 gets a dynamics layer and component 6 a
  regularized-selection layer, both on a shared `SensitivityCore`. They solve
  different problems. Rejected: one engine with a mode flag. *Supersedes
  2026-06-27 "one estimation engine"*. Spec §13.1 still states the old
  decision.
- Build the empirical noise floor from block-permutation (default) or
  phase-randomization surrogates. The API rejects a plain shuffle, because it
  destroys autocorrelation and so understates the floor.
- Set model order by numerical rank (the singular-value gap), not by MP signal
  rank. An orthogonal DOE has full rank but no MP spikes.
- Give Cpk and Harris finite-sample intervals, with a minimum-sample gate of 8.
  The closure report advises 30 or more.
- Run the diagnostics as a precedence tree behind an excitation gate (floor
  0.75). QR attribution is deferred while a gain mismatch is present.
- Make the assumptions explicit and guard each one: A1 drift monitor, A2 u0
  validity horizon, A3 phantom-FF quarantine (2σ envelope, |t| > 2), A4
  slow-bias routing, A5 control-off anomaly exclusion (> 3σ), A6 closed-loop
  excitation discount.
- Rename the package to `r2r_control`, move it under `src/`, and enforce a
  strict layer order with `tests/test_architecture.py`.
- The optimizer takes a simulator factory, and only `pipeline/` wires the
  components together, so no component imports another.

## 2026-06-27: first implementation

- One shared core for all six components: variance accounting against a noise
  floor.
- ~~Components 1 and 6 are one estimation engine (spec §13.1).~~
  **Superseded by 2026-09-23.**
- Refuse the analytic Marchenko–Pastur floor on unbalanced panels, and use an
  empirical null instead.
- Every stochastic step is seedable, and every artifact carries a provenance
  log.
- Refuse attributions the data cannot support, such as separating gain error
  from estimator tuning (FB-08).
- The planner proposes experiments on adjustable inputs only, and a person must
  approve them. Nothing actuates a tool.
- Validate with forward chaining plus purge/embargo, and refuse naive k-fold.
- Counterfactual: `y_nocontrol = y − M(u_used − u0)`.
- Every spec-level choice is an explicit setting (`RunConfig`, `PlannerConfig`),
  never a hidden default.

## Open decisions for the owner (spec §13)

| # | Decision | Status in code |
|---|---|---|
| 13.1 | Estimator unification | Split in code; the spec text still says unified |
| 13.2 | Open-loop baseline `u0` | Configurable (`u0=`, `allow_time_varying_u0`); not confirmed |
| 13.3 | Targets and spec limits | Required in `RunConfig`; real values not supplied |
| 13.4 | Uncertainty of M | Defaults to the identified value; no owner estimate |
| 13.5 | Model-consistency check | Implemented on `y_pred`; calibrated on synthetic data only |
| 13.6 | Planner choices (D vs I, loop status, budget) | Config, default D-optimal / open / 8 runs; not confirmed |
| 13.7 | Regression target (raw vs innovation) | Config, default raw; not confirmed |
| 13.8 | Noise-floor choice per dataset | Automatic, with the reason recorded; not confirmed |
