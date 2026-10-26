# r2r_control

A Python toolkit for building, testing and diagnosing run-to-run (R2R) process
controllers from imperfect plant data.

**Version** 0.3.0 · **License** proprietary · **Status** research prototype,
exercised on synthetic data only (as of 2026-10-26)

## Summary

Run-to-run control adjusts a process's knobs between runs (wafers, lots,
batches) so the measured outputs stay on target. `r2r_control` helps an engineer
build that controller from the data they already have:

- it estimates the process gain and says which parts the data cannot support;
- it proposes experiments to fill the gaps;
- it simulates and tunes the controller;
- it diagnoses where leftover variation comes from;
- it screens sensors for feedforward opportunities.

When the data cannot answer a question, it says so instead of guessing.

## When to use it

Use it for offline engineering analysis of an R2R loop with a handful of knobs
and outputs (about ten of each), hundreds of runs, sparse metrology, and ideally
some DOE history. Typical questions it answers:

- How reliable is our gain model?
- Is an experiment worth running, and which knobs should it move?
- Which controller settings are best in simulation?
- Is the controller already as good as it can be?
- Which upstream sensors would help as feedforward?

It is for process and control engineers and the people who own the decision to
spend on experiments. It is not a controller to run on a tool.

## How it works

Six components share one data table (`EventTable`) and one set of statistical
services:

```
DOE + inline data ─► 1 gain identification ─► gain M, uncertainty, low-confidence knobs
                            │                              │
                            ▼                              ▼
                     2 experiment planner        3 simulator ◄── 4 optimizer
                                                           │
                                                           ▼
                                                     5 diagnostics
all data ─────────► 6 regression ─► feedforward candidates
```

The common idea is a noise floor. For estimation that is the Marchenko–Pastur
edge; for control it is the minimum-variance (Harris) benchmark. Variation below
the floor is treated as noise, not signal.

Since 0.3.0, the gain is identified only from data where feedback does not bias
it: the DOE plus open-loop inline runs. Every result carries a provenance log
that records what data was used and why. The theory is in
[docs/architecture.md](docs/architecture.md).

## Setup

This is a single Python package. There are no devices or services to set up.
Install it on any machine with Python 3.9 or newer; see the
[installation manual](docs/manuals/installation.md).

## Quick start

1. Install the package with its test extras (see the
   [installation manual](docs/manuals/installation.md)).
2. Run the tests. All 92 should pass, in about 30 s:

   ```bash
   pytest
   ```

3. Run the walkthrough, which exercises all six components on synthetic data
   and takes about 25 s:

   ```bash
   python examples/walkthrough.py
   ```

4. To use your own data, follow the [user manual](docs/manuals/user-guide.md).

## Repository layout

| Path | What it holds |
|---|---|
| `src/r2r_control/contracts/` | `EventTable`, the measurement-presence check, the gain-shape check, the provenance log, model artifacts |
| `src/r2r_control/config/` | `RunConfig`: every open design decision as an explicit setting |
| `src/r2r_control/core/` | Shared services: data quality, excitation and feedback tests, noise floors, counterfactual, validation, statistics, Gaussian process |
| `src/r2r_control/estimation/` | Shared gain core, dynamics (component 1), regularized selection (component 6) |
| `src/r2r_control/controller/` | Kalman estimator with delay handling, receding-horizon MPC |
| `src/r2r_control/metrics/` | Cpk, %OOS, Harris, with finite-sample intervals |
| `src/r2r_control/components/` | The six components |
| `src/r2r_control/pipeline/` | `GatePipeline`, which runs the components in order |
| `src/r2r_control/synthetic.py` | Closed-loop data generator with injectable faults |
| `examples/` | `walkthrough.py` |
| `tests/` | 92 tests, including the architecture rules |
| `docs/manuals/` | Installation, user and troubleshooting manuals |
| `docs/design/` | Plans and the decision log |
| `docs/` | Architecture, traceability, 0.3.0 closure report |

Imports flow one way: contracts → config → core → estimation → controller and
metrics → components → pipeline. No component imports another, and
`tests/test_architecture.py` enforces both rules.

## Assumptions and where it applies

- **The process is linear around its operating point:** output change = M ×
  knob change + feedforward + disturbance + noise. The disturbance drifts slowly
  and is autocorrelated.
- **Six load-bearing assumptions, each with a check:** constant gain (drift
  monitor); a correct open-loop baseline `u0` (taken from control-off runs);
  M accurate enough to subtract the controller's effect (phantom-feedforward
  guard); metrology delay within the estimator horizon; representative
  control-off episodes; experiments that actually reach the process.
  [Closure report §5](docs/project-closure.md) lists what each check misses.
- **The data:** one row per run in time order, with both recommended and used
  knob values. Knob columns must be complete. Outputs may be sparse.
- **Decisions the owner must make**, kept as settings: targets and spec limits,
  the uncertainty of M (it defaults to the identified value), the planner
  objective and budget, and the regression target. See
  [DECISIONS.md](docs/design/DECISIONS.md#open-decisions-for-the-owner-spec-13).

## Limitations and where it doesn't apply

- It is not a production controller: no real-time loop, no equipment or MES
  interface, no CLI, no user interface. The planner only proposes experiments;
  a person approves and runs them.
- It has been validated on synthetic data only. Every threshold (for example
  the excitation gate 0.75, the gain ratio 1.3 and the Harris tolerance 1.25)
  was calibrated on the built-in generator; see
  [closure report §9](docs/project-closure.md).
- The models are simple. The gain is static and linear, and the dynamics are a
  first-order disturbance. Ridge, elastic-net and PLS settings are fixed, not
  tuned.
- Without a DOE, the gain comes from closed-loop data. It is flagged as
  unreliable (100% bias floor, every knob low-confidence), and in tests it was
  39–76% wrong.
- Variance shares in the diagnosis split the recoverable variance equally among
  the causes found. They are not a quantitative decomposition.
- Regression in the pipeline covers only the first output.
- It does not apply to within-run control, strongly nonlinear processes, large
  knob counts, records of a few dozen runs, or anything safety-critical.

## Known issues

There are no tracker issues yet. Residual issues R-1 to R-6 are in
[closure report §8.10](docs/project-closure.md). The following were re-checked
on this branch on 2026-10-26:

1. **The gain-mismatch check is biased under feedback.** On a correctly
   modelled simulated loop the realized-to-model gain ratio reads 1.46–1.54,
   above the 1.3 alarm level. A fix tried on 2026-10-26 did not work with the
   new MPC (see the [plan](docs/design/plans/2026-10-25-code-review-and-fixes.md)).
2. **The planner ignores the gap unless it is told the target features.**
   `plan()` called on unrelated passive data with the same seed proposes the
   same experiments. The pipeline passes `target_features`; direct calls should
   pass them too.
3. **The estimator-versus-controller tuning verdict depends on units.**
   Rescaling the knobs by 10 changes it from `neither_dominant` to
   `Controller_QR`.
4. **One missing value in a knob or feedforward column** makes
   `FBIdentifier.identify` fail with `LinAlgError: SVD did not converge`.
5. **Smaller defects:** the alias check reports "no aliasing detected" for a
   resolution-IV DOE; `reconstruct_with_bands` fails with a time-varying `u0`;
   on Windows, printing a report to a redirected stream raises
   `UnicodeEncodeError` (use `python -X utf8`).
6. **The revised specification is not in the repository.** The checked-in spec
   still requires the single estimation engine that 0.2.0 replaced.

## Path forward

The running list is in
[docs/design/plans/2026-10-24-future-work.md](docs/design/plans/2026-10-24-future-work.md).
In order:

1. Back-test on historical data, and recalibrate the thresholds (P1).
2. Commit the revised specification.
3. Fix the open issues above, starting with the gain-mismatch bias (added
   2026-10-26).
4. Recover the FF-selection power lost to the gold holdout (R-1), and tune the
   hyperparameters inside validation.
5. Productize: load saved artifacts, add a CLI, set up CI.

## Design decisions

The most important ones, newest first. The full log is in
[docs/design/DECISIONS.md](docs/design/DECISIONS.md).

- 2026-10-26: feedforward candidates are chosen from the FF sensors alone, with
  a Bonferroni test, because knob co-predictors produced false candidates.
- 2026-10-24: closed-loop inline runs are excluded from gain identification,
  because feedback biased the fused gain (10–31% error).
- 2026-10-24: gain uncertainty includes a bias term and is propagated
  everywhere, because the old fixed 10% was 2.4–4.8 times too small.
- 2026-10-24: receding-horizon MPC with exact move limits, because clipping
  could cost more than not moving.
- 2026-09-23: components 1 and 6 share a gain core but are separate classes.
  This supersedes the 2026-06-27 single engine.
- 2026-09-23: the empirical noise floor uses structure-preserving surrogates;
  a plain shuffle is refused.
- 2026-06-27: attributions the data cannot support are refused, and every
  result carries a provenance log.

## Conventions, tests, contributing

- Python on numpy, pandas and scipy only. Statistical tests are written from
  scratch.
- Every random step is seeded; the same seed gives the same result.
- Each fix lands with a test that fails before it. Run `pytest` before every
  commit.
- Record design decisions in `docs/design/DECISIONS.md`, and record releases in
  `CHANGELOG.md`.
