# Run-to-Run Process Control Software (`r2r_control`)

A Python toolchain for **building, simulating, tuning, and diagnosing run-to-run
(R2R) process controllers** from imperfect industrial data. It implements the
revised design requirements specification: six functional components on a
shared set of cross-cutting services, with the load-bearing assumptions made
explicit and guarded.

The unifying idea is **variance accounting against a noise floor**: identifying
the feedback (FB) model, finding feedforward (FF) opportunities, tuning the
estimator and controller weights, and deciding whether the controller is "done"
are all the same question:

> *How many directions of variation in the data can be trusted as signal, given
> the excitation actually present, and where is the floor below which variation
> is indistinguishable from noise?*

That floor is the **Marchenko–Pastur edge** for estimation and the
**minimum-variance / Harris benchmark** for control. It is computed analytically
where the assumptions hold and from **structure-preserving surrogates** where
they do not.

## Project status: closed (October 2026)

Version 0.3.0 is the final release. It is an offline decision-support library
validated on **synthetic data only**. Read
[`docs/project-closure.md`](docs/project-closure.md) before using any output on
real process data. That report covers the assumptions, applicability,
methodology, limitations, known issues and future work.

The 0.2.0 closure review found nine known issues. 0.3.0 fixes KI-1 to KI-8 and
part of KI-9; the report gives the evidence before and after each fix:

- **Closed-loop bias in the gain (KI-1).** Closed-loop inline rows are now
  detected by a feedback test and excluded from gain identification. The fused
  gain error equals the DOE-only error (2–5%); before, it was 10–31%.
- **Gain uncertainty (KI-2).** The reported uncertainty now includes a bias term
  and is propagated downstream. The actual error is 0.6–1.4× the reported value;
  before, it was 2.4–4.8×.
- **Simulated metrology delay (KI-3).** Measurements now arrive late, and
  control degrades with delay as it should.

What remains:

- Validation is on synthetic data only.
- The control-off gold holdout lowers FF-selection power on some data.
- A gain error and an FF-model error can be inseparable from passive data.
- The revised specification is not in the repository.

## Install

```bash
pip install -e .            # core (numpy, pandas, scipy)
pip install -e ".[test]"    # + pytest, matplotlib
pytest                      # 90 tests, incl. injected-fault and refusal tests
python examples/walkthrough.py
```

Verified at closure on Python 3.11 with numpy 2.4, pandas 3.1 and scipy 1.17:
90 passed in about 19 s, no warnings.

## Repository layout (§16)

```
src/r2r_control/
├── contracts/     schema, _is_measured, shape contract, provenance, ModelArtifact
├── config/        RunConfig — the §13 open decisions as explicit config
├── core/          shared services, ONE implementation each (§IF-06)
│   ├── data_quality/   doe · inline · joint · decisions
│   ├── excitation/     vif · spectrum · directions · accidental
│   ├── noise_floor/    analytic · empirical (block/phase surrogates) · cleaning · select
│   ├── counterfactual/ reconstruct · bands · baseline (u0 validity, A2)
│   ├── validation/     forward_chaining · purge_embargo · block_bootstrap
│   │                   · control_off (representativeness, A5) · concept_drift
│   ├── stats.py        ADF, KPSS, variance ratio, Ljung-Box (from scratch)
│   └── gp.py           GP surrogate + integer/categorical helpers (E11)
├── estimation/
│   ├── sensitivity/    SensitivityCore — shared by Components 1 and 6
│   ├── dynamics/       Component 1 only: order selection, state space
│   └── regularized/    Component 6 only: PLS, elastic net, group stability selection
├── controller/    mhe (+ late-measurement replay / routing, A4) · mpc (receding
│                  horizon, exact move limits, absolute bounds)
├── metrics/       Cpk/Harris with finite-sample CIs + minimum-sample gate (E10)
├── components/    fb_identification · experiment_planner · simulator ·
│                  controller_optimizer · diagnostics · regression_machine
├── pipeline/      gate workflow wiring the components together (EP-07)
└── synthetic.py   fault-injectable closed-loop data generator (NFR-04)
```

Imports flow only `contracts → config → core → estimation → {controller,
metrics} → components → pipeline`, and no component imports another.
`tests/test_architecture.py` enforces both rules, so breaking them fails the
build. The top-level `artifacts/` directory is an empty, git-ignored place for
locally generated outputs, such as diagnostic figures written via `figure_dir`.
Nothing writes there by default.

### The corrected unification (§1.3)

Components 1 and 6 are **not** one estimator class. They share the
`SensitivityCore` (how outputs respond to inputs) and the cross-cutting services,
so the noise floor, the counterfactual and the validation scheme mean the same
thing for both. Component 1 then wraps the core with a **dynamics** layer, and
Component 6 wraps it with a **regularized-selection** layer.

## Quickstart

```python
from r2r_control import synthetic
from r2r_control.config import default_run_config
from r2r_control.pipeline import GatePipeline

ds = synthetic.make_dataset(seed=0, doe_age_days=20)
t = ds.truth
cfg = default_run_config(targets=t.targets.tolist(),
                         lsl=(t.targets - 3).tolist(), usl=(t.targets + 3).tolist())
res = GatePipeline(cfg).run(ds.inline, ds.doe, doe_age_days=20, run_optimizer=True)

print("\n".join(res.gate_log))
print(res.fb.provenance.to_json())               # full audit trail
print(res.diagnosis.precedence_log)              # DIAG-07 decision tree
print(res.regression.feedforward_candidates)
```

Each component can also be used on its own; see
[`examples/walkthrough.py`](examples/walkthrough.py).

## Assumptions and guards (§1.4, §15)

| Assumption | Guard | Where |
|---|---|---|
| A1 gain is time-invariant | realized-gain drift monitor | `diagnostics/gain_mismatch.py::gain_drift_monitor` |
| A2 `u0` valid over the horizon | control-off baseline drift check, validity horizon, optional time-varying `u0` | `core/counterfactual/baseline.py` |
| A3 `M` accurate enough to decouple | identified M-uncertainty propagation + phantom-FF quarantine | `regression_machine/decouple.py` |
| A4 metrology delay < MHE horizon | delayed measurements folded in within the horizon, older ones routed to a slow-bias estimator | `components/simulator/run.py`, `controller/mhe/` |
| A5 control-off episodes are representative | tool-state check excludes anomalous episodes from the C6 gold holdout | `core/validation/control_off.py` |
| A6 injected excitation lands | closed-loop excitation discount on information gain, calibratable from measured residual excitation | `experiment_planner/loop_status.py` |
| M3 gain rows are exogenous | feedback test excludes closed-loop inline rows from gain identification | `core/excitation/feedback.py` |

Each guard has blind spots, and further modelling assumptions (linear gain,
superposition, run-to-run settling) have no guard at all. See sections 5.1 and
5.2 of the closure report.

## Honesty over convenience (NFR-06)

The system reports hard limits instead of papering over them, and a test covers
each one:

- The **analytic MP floor is refused** on unbalanced panels. The empirical floor
  **never uses a plain shuffle**, which would erase autocorrelation and push
  the floor too low (NF-03/E6).
- It **refuses to separate gain error from estimator tuning** using passive
  closed-loop data (FB-08). When the controller's prediction `y_pred` is
  logged, the §13.5 diagnostic separates an inaccurate gain from tracking lag,
  and reports when a gain error and an FF-model error cannot be told apart.
- It **excludes closed-loop inline rows from gain identification**. Without a
  DOE it flags the closed-loop fallback with a 100% bias floor (DQ-06).
- Under tight control, gain and QR diagnoses come back as **"unidentifiable from
  available data — excitation required"** and are routed to the planner, not
  guessed (DIAG-07/E4). QR attribution waits until any gain mismatch is resolved (E5).
- **Cpk from a handful of points is suppressed** as "insufficient samples"
  (SIM-06/E10).
- It **refuses naive k-fold** on temporally ordered data (VAL-01).

## Documentation

| Document | Contents |
|---|---|
| [`docs/project-closure.md`](docs/project-closure.md) | Closure report: assumptions, applicability, methodology, limitations, known issues, future work |
| [`docs/architecture.md`](docs/architecture.md) | Theory, layers, assumption guards and workflow |
| [`docs/requirements-traceability.md`](docs/requirements-traceability.md) | Requirement → implementation → test, plus conformance caveats |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history |
