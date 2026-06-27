# Run-to-Run Process Control Software

A cohesive Python toolchain for **building, simulating, tuning, and diagnosing
run-to-run (R2R) process controllers** from imperfect industrial data. It
implements the [design requirements specification](design-requirements-specification.md)
in full, binding six functional components onto one conceptual core.

The unifying idea is **variance accounting against a noise floor**: every
estimation problem in the system — identifying the feedback (FB) model, the
feedforward (FF) opportunities, tuning the estimator and controller weights, and
deciding whether the controller is "done" — is the same question:

> *How many directions of variation in the data can be trusted as signal, given
> the excitation actually present, and where is the floor below which variation
> is indistinguishable from noise?*

That floor is the **Marchenko–Pastur edge** for estimation and the
**minimum-variance / Harris benchmark** for control. The system computes it
analytically where the assumptions hold and empirically (permutation /
cross-validation) where they do not, and uses it consistently to gate which
model terms to trust, decide whether more data is needed, and declare when
residual variance is irreducible rather than a controller defect.

## Install

```bash
pip install -e .            # core (numpy, pandas, scipy)
pip install -e ".[test]"    # + pytest, matplotlib
pytest                      # 39 tests, incl. deliberate identifiability-limit faults
```

## Architecture

```
process_control/
├── provenance.py            NFR-03  audit trail (decisions, assumptions, floors)
├── synthetic.py             NFR-04  fault-injectable closed-loop data generator
├── data/
│   ├── schema.py            §4.1    EventTable, _is_measured, shape contract
│   └── quality.py           §4.2    use/down-weight/reject gatekeeper
└── foundations/             §4.3-4.6 SHARED services (one implementation each, §IF-06)
    ├── stats.py             NFR-01  ADF, KPSS, variance-ratio, Ljung-Box (from scratch)
    ├── excitation.py        §4.3    VIF, spectrum, effective rank, accidental excitation
    ├── noise_floor.py       §4.4    MP analytic + permutation empirical + Harris
    ├── counterfactual.py    §4.5    M @ (u_used − u0) subtraction service
    ├── validation.py        §4.6    forward-chaining, purge/embargo, block bootstrap
    ├── gp.py                NFR-01  Gaussian-process surrogate
    └── engine.py            §REG-01 ONE estimation engine, two configurations

process_control/components/
├── component1_identification.py  §5   FB / process-dynamic model ID
├── component2_planner.py         §6   experiment planner (D/I-optimal + GP infill)
├── component3_simulator.py       §7   MHE/MPC simulator (control-on / control-off)
├── component4_optimizer.py       §8   controller optimizer (outer loop)
├── component5_diagnostics.py     §9   diagnostics + visualization + achievability verdict
└── component6_regression.py      §10  all-data regression machine
```

The FB identifier (Component 1) and the regression machine (Component 6) are
**two configurations of one `EstimationEngine`** (`mode='fb'` vs
`mode='regression'`), sharing the data model, data-quality layer, excitation
analysis, noise-floor estimator, counterfactual service, and validation
framework — the unification is a hard requirement (§REG-01).

## Quickstart

```python
import numpy as np
from process_control import synthetic
from process_control.components import (
    FBIdentifier, Simulator, PlantConfig, ControllerConfig,
    ControllerOptimizer, SearchDimension, DiagnosticsEngine,
    RegressionMachine, ExperimentPlanner, PlannerConfig)

# 0. Synthetic closed-loop dataset (DOE + sparse inline metrology)
ds = synthetic.make_dataset(seed=0, doe_age_days=20)
t = ds.truth

# 1. Identify the FB gain (DOE wide-range gain fused with inline local correction)
fb = FBIdentifier(estimator="svd").identify(ds.inline, ds.doe, doe_age_days=20)
print("gain error:", np.linalg.norm(fb.M - t.M) / np.linalg.norm(t.M))
print(fb.provenance.to_json())          # full audit trail

# 3. Simulate control-on vs control-off with Monte-Carlo uncertainty bands
lsl, usl = t.targets - 3, t.targets + 3
sim = Simulator(M=t.M, targets=t.targets, lsl=lsl, usl=usl, u0=t.u0,
                FF_gain=t.FF_gain, controller=ControllerConfig(mpc_R=0.05),
                rel_uncertainty_M=fb.relative_uncertainty())
res = sim.simulate(n_steps=300)
print("Cpk on/off:", res.metrics_on.cpk, res.metrics_off.cpk)

# 4. Optimize the controller weights against a robust objective
dims = [SearchDimension("mpc_R", 1e-3, 1.0, log=True),
        SearchDimension("move_limit", 0.2, 1.5)]
opt = ControllerOptimizer(lambda c: Simulator(M=t.M, targets=t.targets, lsl=lsl,
        usl=usl, u0=t.u0, FF_gain=t.FF_gain, controller=c), dims,
        objective_metric="std")
best = opt.optimize(algorithm="bayesian", budget=24)

# 5. Diagnose: where does the residual variance live, and is it irreducible?
diag = DiagnosticsEngine().diagnose(res.y_on, res.u, fb.M,
        state_est=res.state_est, ff_inputs=res.ff_inputs, innovation=res.innovation)
print(diag.achievability["verdict"])

# 6. Regress all sensors against the open-loop-equivalent signal; find FF candidates
rm = RegressionMachine().fit(ds.inline, "y0", M=fb.M, u0=t.u0,
        adjustable_features=ds.inline.knob_cols_used)
print("feedforward candidates:", rm.feedforward_candidates)

# 2. Turn the identifiability gap into prioritized, justified experiments
pl = ExperimentPlanner(PlannerConfig(primary_objective="D"))
plan = pl.plan(ds.inline.regressors(), ds.inline.regressor_names,
               ds.inline.knob_cols_used)
print(plan.gap.confounding_summary)
```

See [`examples/walkthrough.py`](examples/walkthrough.py) for a runnable
end-to-end version.

## Honesty over convenience (NFR-06)

The system surfaces hard limits rather than papering over them. A few examples,
each exercised by a test:

- It **refuses the analytic MP floor** on an unbalanced (sparse-metrology) panel
  and falls back to the empirical permutation floor, recording why (`NF-02/04`).
- It **refuses to separate gain error from estimator tuning** on passive
  closed-loop data — that product is structurally unidentifiable (`FB-08`).
- It **refuses naive k-fold** on temporally ordered data (`VAL-01`).
- A correct *"cannot be determined from this data"* is a required output, routed
  to the experiment planner, not a silent guess.

## Requirements traceability

Every `SHALL` clause in the specification is implemented and referenced by ID in
the relevant docstring. The per-component reference documents in
[`docs/`](docs/) map each requirement to its implementation (NFR-05).
