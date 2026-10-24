# Changelog

## 0.3.0 — Fixes for the closure known issues (2026-10-24)

Resolves KI-1 to KI-8 and part of KI-9 from the 0.2.0 closure review.
`docs/project-closure.md` §8 gives the evidence before and after each fix.
90 tests pass (64 in 0.2.0).

### Behavior changes
- **Gain identification excludes closed-loop inline rows (KI-1, DQ-06).**
  - A new feedback test (`core/excitation/feedback.py`) asks whether the history
    of measured outputs predicts the knob settings. If it does, the inline
    report sets `gain_usable = False`.
  - `FBIdentifier` builds the gain from the DOE plus open-loop inline rows only.
  - Without a DOE, the closed-loop fit is flagged with a 100% bias floor and
    every knob is low-confidence.
  - `down_weight` verdicts now carry weight 0.5.
  - FB test tolerances were tightened from 30–35% to 10%.
- **Gain uncertainty includes bias and is propagated (KI-2).**
  - `relative_uncertainty = hypot(variance, bias)`. The variance term is now
    summed over outputs.
  - `RunConfig.rel_uncertainty_M` defaults to `None`, meaning "use the
    identified value". An override below it logs a warning.
- **Simulator (KI-3, KI-4).**
  - Measurements are released `metrology_delay` steps late. Late measurements
    within `mhe_horizon` are folded in by replaying the Kalman time updates.
  - `MPCController` is a receding-horizon MPC over the predicted disturbance
    trajectory. Move limits are enforced exactly (bounded least squares) and
    optional `u_lower`/`u_upper` bounds by SLSQP.
  - `y_pred` is recorded.
  - Harris delay = metrology delay + 1.
  - `_run_once` now returns 7 values.
- **Pipeline (KI-5).**
  - The simulator uses C1's `A`, FF gains (new: ridge on the counterfactual)
    and identified uncertainty.
  - The disturbance stability clip triggers at |λ| ≥ 1 (was 1.5).
  - Low-confidence knobs are passed to the planner.
- **Planner (KI-6, E11, A6).**
  - Signed gap directions, plus `target_features`; the acquisition is boosted
    toward the gap (`gap_weight`).
  - GP hyperparameters are fitted by marginal likelihood when responses exist.
  - `CodedSpace` handles engineering units (`knob_bounds`), integer and
    categorical levels (`mixed_kernel`), and `px_weighting`.
  - New cost model; proposals record their purpose.
  - `calibrate_rejection` sets the closed-loop rejection fraction from measured
    residual excitation.
  - `plan()` accepts existing experiments and responses.
- **Regression machine (KI-7).**
  - Control-off episodes are held out as gold by default (VAL-04).
  - `embargo` is configurable (VAL-03).
  - Trace columns are aggregated, with a held-out raw-retention check (REG-09).
  - **Note:** the gold holdout can lower FF-selection frequency where
    control-off episodes are the main FF-independent excitation (closure report
    R-1).
- **Covariance cleaning (NF-05).** `covariance_cleaning = "clip" |
  "ledoit_peche" | "cv"` on the ridge path. It is opt-in.
- **Statistics (KI-8).**
  - The Cpk interval uses an autocorrelation-discounted `n_eff`
    (`Metrics.n_effective`).
  - ADF uses finite-sample critical values and a p-value.
  - FF leakage applies Šidák across outputs and Holm across inputs.
- **§13.5 model consistency (KI-9).**
  - `EventTable.y_pred_cols` stores the controller's prediction.
  - `model_consistency` (diagnostics node 2b) returns `M_inaccurate`,
    `M_or_FF_model_error`, `FF_model_error`, `fb_tracking_lag` or `consistent`.
  - The synthetic generator emits `<y>_pred` columns and takes an
    `estimator_gain` parameter.

### Not fixed
- The revised specification is still not in the repository (KI-9).
- `subspace_order` is still used only in tests.

### Other
- The test suite takes about 19 s, up from 8 s, mainly the constrained MPC.
- Docs: closure report, README, architecture and traceability updated for
  0.3.0.

## Project closure review — 2026-10-24 (documentation, on 0.2.0)

No algorithm or behavior changes in this entry; the fixes followed in 0.3.0.

- Added `docs/project-closure.md`. It covers the assumptions (guarded and
  unguarded), applicability, methodology, limitations, nine known issues with
  evidence and suggested fixes, the heuristic thresholds, and prioritized
  future work.
- Re-verified on Python 3.11 with numpy 2.4, pandas 3.1 and scipy 1.17:
  64 tests pass with no warnings, and the walkthrough runs end to end.
- README: corrected the test count (63 → 64), added a project-status section
  and a documentation index, and corrected the layer order (`config` was
  missing).
- `docs/requirements-traceability.md`: added the requirements found to be only
  partially met, mapped to the known issues.
- `docs/architecture.md`: added a pointer to the known issues.
- `.gitignore`: ignore generated files under `artifacts/` (the placeholder
  `.gitkeep` stays).

## 0.2.0 — Re-architecture to revised specification

Aligns the codebase with the revised design requirements specification
(spec §14 corrections, §1.4 load-bearing assumptions A1–A6, §15 edge cases
E1–E14, and the §16 repository layout).

### Architectural corrections
- **De-unified the estimator (§1.3, REG-01).** Components 1 and 6 are no longer
  one class. They share a `SensitivityCore` (cross-sectional gain estimation)
  and the cross-cutting services, but Component 1 wraps it with a dynamics layer
  and Component 6 with a regularized-selection layer.
- **Structure-preserving empirical noise floor (NF-03, E6).** Replaced the plain
  per-column shuffle (which destroys autocorrelation and understates the floor)
  with block-permutation / phase-randomization surrogates that preserve marginal
  distribution, autocorrelation, and missingness — sharing the block logic with
  the block bootstrap (VAL-05).
- **Finite-sample metric uncertainty (SIM-06, E10).** Cpk/Harris on real sparse
  data carry confidence intervals, and a minimum-sample gate suppresses
  under-powered metrics ("insufficient samples").
- **Diagnostic precedence (DIAG-07, E4, E5).** Diagnostics run as an explicit
  decision tree behind an excitation-sufficiency gate; nodes can emit
  "unidentifiable from available data — excitation required".
- **Load-bearing assumptions made explicit (§1.4, A1–A6)** with named guards:
  gain-drift monitor (A1/E1), u0 validity horizon (A2/E3), phantom-FF quarantine
  (A3/E2), out-of-sequence measurement routing (A4/E7), control-off
  representativeness check (A5/E8), closed-loop excitation discount (A6/E9).
- Group-aware stability selection (E12) and integer/categorical-aware GP
  helpers (E11).

### Repository structure (§16)
- Package renamed `process_control` → `r2r_control`, moved under `src/`.
- Strict layers `contracts → core → estimation → {controller, metrics} →
  components → pipeline`, enforced by `tests/test_architecture.py`.
- Components no longer import one another; the optimizer takes a simulator
  factory, and the pipeline wires components together.

## 0.1.0 — Initial implementation

First implementation against the original specification: six components on one
shared core, 39 tests including deliberate identifiability-limit faults.
