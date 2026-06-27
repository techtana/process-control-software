# Architecture & Theory Reference

This is the self-contained handoff document (NFR-05): theory → workflow →
implementation → validation, so a future session continues without re-deriving
context.

## The one idea: a noise floor everywhere

Every estimation/control decision is "is this variation signal or noise?".
Two floors answer it, and the system uses them consistently:

- **Estimation floor — Marchenko–Pastur.** Eigenvalues of a sample correlation
  matrix of pure noise fall below `(1 + √q)²`, `q = N/T`. A direction above the
  edge carries common structure (the "8-vs-392" split in high-dim regression).
  Implemented in `foundations/noise_floor.py`. When the balanced-panel
  assumption fails (sparse metrology), the analytic edge is **refused** and a
  permutation null replaces it.
- **Control floor — minimum-variance / Harris.** The smallest output variance
  any controller can achieve is the variance of the delay-step-ahead
  unforecastable disturbance. `harris_index = achieved / minimum`. A value near
  1 means "done; the rest is irreducible". Implemented in the same module and
  consumed by the simulator (SIM-06) and diagnostics (DIAG-05).

Two notions of "rank" are kept distinct because they answer different questions:

| Notion | Question | Where used |
|--------|----------|------------|
| **numerical rank** (singular values above a relative gap) | how many directions are *excited enough to identify*? | model-order selection (FB-02), gap localization (EP-01) |
| **MP signal rank** (eigenvalues above the floor) | how many directions carry *common low-rank structure*? | high-dim feature/latent screening (REG-03), confounding narrative |

A designed orthogonal experiment is full **numerical** rank (every direction
excited) but has **no MP spikes** (excitation is uniform, not low-rank). Using
the MP spike-count for DOE order selection would wrongly report rank 0 — so the
engine selects order by numerical rank. This distinction is the single most
important implementation subtlety in the codebase.

## The central obstacle: closed-loop confounding

Passive inline data reflects the controller's corrections, not the open-loop
process: `u` becomes a function of estimated state, so knobs and disturbances
are collinear and the covariance is rank-deficient in exactly the directions you
need. The system:

1. **quantifies** it — VIF + condition number + numerical-rank deficiency
   (`foundations/excitation.py`), reported in stakeholder terms by the
   data-quality layer;
2. **exploits** accidental excitation — manual overrides and control-off
   episodes inject "free" variance (`mine_accidental_excitation`), fed to
   Component 1 and used as validation gold;
3. **requests** deliberate excitation — the gap is localized to specific
   adjustable directions and routed to the experiment planner (Component 2).

No quantity of passive data resolves the gain-error-vs-estimator-tuning product
(FB-08); the system records that as a non-holding assumption rather than
emitting a false attribution.

## The two-estimator unification (§1.3, REG-01)

`foundations/engine.py::EstimationEngine` is **one class with two
configurations**:

- `mode='fb'` (Component 1): structured MIMO gain `M` (`n_out × n_knob`) for the
  MPC, fused from DOE (wide-range gain) and inline (local correction + noise).
- `mode='regression'` (Component 6): broad predictor across all sensors,
  decoupling the controller via the counterfactual, finding feedforward
  candidates.

Both share: the data model, the data-quality layer, the excitation analysis,
the noise-floor estimator, the counterfactual service, and the validation
framework. The confounding and noise-floor logic is therefore literally the same
code for both — the unification is enforced, not aspirational.

## Workflow (the simulate → optimize → diagnose loop)

```
DOE + inline ── DataQualityLayer ──► EstimationEngine(mode=fb) ──► FBModel(M, uncertainty)
                       │                                                  │
                       │ gaps                                             │ M + uncertainty
                       ▼                                                  ▼
              ExperimentPlanner ◄──── identifiable-direction report ─── Simulator (on/off arms)
                                                                          │ metrics + distributions
                                                                          ▼
                                                              ControllerOptimizer (outer loop)
                                                                          │ best config
                                                                          ▼
                              counterfactual + noise-floor ──────► DiagnosticsEngine
                                                                  (variance decomposition,
                                                                   achievability verdict)

all data ── EstimationEngine(mode=regression) ──► feedforward candidates, low-variation gaps ──► planner
```

## Validation strategy

- **Counterfactual** is validated to machine precision on synthetic MIMO data
  (the disturbance cancels algebraically): `test_counterfactual_exact_on_synthetic_mimo`.
- **Identification** is validated by gain-recovery error against the known
  synthetic truth.
- **Diagnostics** are validated by *injected faults*: a 4× gain underestimate
  must be diagnosed as over-correction with Harris ≫ 1; a healthy loop must be
  declared at the floor.
- **Refusals** are validated as first-class behavior: the MP floor is refused on
  unbalanced panels, naive k-fold is refused on temporal data, and gain-vs-tuning
  separation is refused on passive data.

## Open design decisions (§13) — tracked as configuration

These are surfaced as config, not silently defaulted:

| Decision | Where |
|----------|-------|
| Counterfactual baseline `u0` | `FBIdentifier.identify(u0=…)`; defaults to control-off mean with a recorded caveat |
| Spec inputs (targets, limits) | `Simulator(targets, lsl, usl)` |
| `M` uncertainty for MC bands | `FBModel.relative_uncertainty()` → `Simulator(rel_uncertainty_M=…)` |
| Planner objective D vs I, `p(x)`, loop status, budget | `PlannerConfig` |
| Regression target raw vs innovation | `RegressionMachine.fit(target_kind=…)` |
| Noise-floor default per dataset | `noise_floor.estimate` chooses analytic/empirical and records why |
