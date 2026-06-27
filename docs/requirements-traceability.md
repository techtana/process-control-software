# Requirements Traceability Matrix

Every `SHALL` clause in the
[design requirements specification](../design-requirements-specification.md) maps
to an implementation site and (where behavioral) a test. IDs also appear in the
relevant source docstrings.

## §4.1 Data Model and Schema Contract

| Req | Implementation | Test |
|-----|----------------|------|
| DM-01 unified event-indexed table | `data/schema.py::EventTable` | `test_components` (via fixtures) |
| DM-02 sparse/asynchronous metrology | `EventTable` (`y_time_cols`, `measured_mask`) | `tests/conftest` dataset |
| DM-03 `_is_measured` predicate for array cells | `data/schema.py::_is_measured` | `test_foundations::test_is_measured_handles_all_nan_array` |
| DM-04 shape contract `n_out × n_knob` | `data/schema.py::check_shape_contract` | `test_foundations::test_shape_contract_fails_loudly` |
| DM-05 recommended-vs-used retained | `EventTable.recommended_used_divergence` | `test_integration` (DQ uses it) |

## §4.2 Data-Quality and Assumption Validation Layer

| Req | Implementation | Test |
|-----|----------------|------|
| DQ-01/02/03 DOE assumptions, alias/resolution, covariate shift, staleness | `data/quality.py::DataQualityLayer.assess_doe`, `analyze_alias_structure`, `covariate_shift` | `test_integration::test_data_quality_rejects_stale_shifted_doe` |
| DQ-04/05/06 inline violations, VIF/rank, divergence, metrology, stationarity, decision | `DataQualityLayer.assess_inline` | `test_integration::test_data_quality_downweights_confounded_inline` |
| DQ-07 joint-use contract (provenance-weighted) | `DataQualityLayer.combine` | exercised by Component 1 |
| DQ-08 decisions persisted with diagnostics | `provenance.py::Decision`, `ProvenanceLog` | `test_integration::test_provenance_is_serializable_and_complete` |

## §4.3 Excitation and Confounding Analysis

| Req | Implementation | Test |
|-----|----------------|------|
| EX-01 VIF, stakeholder framing | `foundations/excitation.py::vif` | `test_foundations::test_vif_flags_collinear_regressor` |
| EX-02 spectrum + effective rank | `regressor_spectrum`, `numerical_rank` | `test_foundations::test_numerical_rank_*` |
| EX-03 map deficiency to specific directions | `unidentifiable_directions` | `test_components::test_planner_localizes_gap...` |
| EX-04 mine accidental excitation | `mine_accidental_excitation` | Component 1 uses it |

## §4.4 Noise-Floor Estimation

| Req | Implementation | Test |
|-----|----------------|------|
| NF-01 analytic MP edge + Harris benchmark | `noise_floor.py::marchenko_pastur_edge`, `analytic_floor`, `minimum_variance_benchmark`, `harris_index` | `test_foundations::test_mp_edge_monotone` |
| NF-02 declare assumptions, refuse silently | `analytic_floor`, `estimate` | `test_foundations::test_empirical_floor_takes_precedence...` |
| NF-03 empirical permutation floor (preserves missingness) | `empirical_floor` | `test_foundations::test_empirical_floor_separates_signal_from_noise` |
| NF-04 empirical precedence + record why | `estimate` | `test_foundations::test_empirical_floor_takes_precedence...` |
| NF-05 eigenvalue cleaning (clip + shrinkage) | `eigenvalue_clip`, `ledoit_peche_shrinkage` | — |

## §4.5 Counterfactual Reconstruction Service

| Req | Implementation | Test |
|-----|----------------|------|
| CF-01 subtraction, not simulation; exact on synthetic MIMO | `counterfactual.py::reconstruct` | `test_foundations::test_counterfactual_exact_on_synthetic_mimo` |
| CF-02 `u0` sensitivity recorded/propagated | `reconstruct` (`u0_source`), Component 1 baseline selection | walkthrough |
| CF-03 Monte-Carlo uncertainty bands | `reconstruct_with_bands` | `test_foundations::test_counterfactual_bands_widen_with_uncertainty` |
| CF-04 shared by Components 3/5/6 | imported in components 3, 5, 6 | `test_integration` |

## §4.6 Validation Framework

| Req | Implementation | Test |
|-----|----------------|------|
| VAL-01 refuse naive k-fold | `validation.py::naive_kfold_guard` | `test_foundations::test_naive_kfold_is_refused` |
| VAL-02 forward-chaining default | `forward_chaining_splits`, `forward_validate` | `test_foundations::test_forward_chaining_trains_on_past_only` |
| VAL-03 purge/embargo | `forward_chaining_splits(purge=…)` | same |
| VAL-04 control-off gold holdout | `control_off_gold` | — |
| VAL-05 block bootstrap | `block_bootstrap` | `test_foundations::test_block_bootstrap_preserves_contiguity` |
| VAL-06 concept-drift test | `concept_drift_test` | `test_foundations::test_concept_drift_detected...` |

## §5 Component 1 — FB / Process-Dynamic Model Identification

| Req | Implementation | Test |
|-----|----------------|------|
| FB-01 fuse DOE+inline, provenance-weighted | `component1_identification.py::FBIdentifier.identify` | `test_components::test_fb_identification_recovers_gain` |
| FB-02 order by singular-value noise gap | `engine.py::_select_rank`, `numerical_rank` | `test_foundations::test_numerical_rank_full_for_orthogonal_design` |
| FB-03 ridge/SVD/shrinkage regularization | `engine.py::ridge_fit`, `svd_truncated_fit` | — |
| FB-04 report identifiability limits, route to planner | `FBIdentifier` (`low_confidence_entries`) | `test_components::test_fb_routes_unidentifiable_to_planner...` |
| FB-05 incorporate accidental excitation | `FBIdentifier` (weights override/control-off events) | — |
| FB-06 innovation/residual diagnostics, whiteness | `FBIdentifier._innovation_diagnostics` | walkthrough |
| FB-07 emit parameter-uncertainty covariance | `FBModel.param_cov`, `relative_uncertainty` | `test_components::test_fb_identification_recovers_gain` |
| FB-08 refuse to separate gain error from estimator tuning | `FBIdentifier.identify` (assumption holds=False) | `test_components::test_fb_refuses_to_separate_gain_from_estimator_tuning` |

## §6 Component 2 — Experiment Planner

| Req | Implementation | Test |
|-----|----------------|------|
| EP-01 quantify/localize gap from confounding | `component2_planner.py::diagnose_gap` | `test_components::test_planner_localizes_gap...` |
| EP-02 adjustable inputs only | `plan`, `seed_design`, `sequential_infill` | same |
| EP-03 D-/I-optimal seed design | `seed_design`, `_design_score` | `test_components::test_planner_seed_design_objectives_differ` |
| EP-04 GP infill, ALC/IMSE/EIG acquisition | `sequential_infill`, `gp.py` | — |
| EP-05 priority metric + estimated value | `ExperimentProposal`, `sequential_infill` | `test_components::test_planner_localizes_gap...` |
| EP-06 manual approval (propose, not execute) | `plan` (gate log) | same |
| EP-07 gate-based workflow + stopping | `plan` | walkthrough |
| EP-08 open design decisions as config | `PlannerConfig` | — |

## §7 Component 3 — MHE/MPC Simulator

| Req | Implementation | Test |
|-----|----------------|------|
| SIM-01 control-on/off via shared counterfactual | `component3_simulator.py::Simulator.simulate` | `test_components::test_control_reduces_variance` |
| SIM-02 model-error inputs, Monte-Carlo bands | `Simulator` (`rel_uncertainty_M/FF`, `_aggregate_metrics`) | `test_components::test_monte_carlo_bands_present...` |
| SIM-03 MHE State QR + MPC Controller QR independent | `ControllerConfig` (`mhe_Q/R`, `mpc_Q/R`) | — |
| SIM-04 exposed config params | `ControllerConfig`, `PlantConfig` | — |
| SIM-05 drift/AR/measurable-disturbance plant | `PlantConfig`, `_run_once` | — |
| SIM-06 Cpk/%OOS/Harris/mean-off-target | `compute_metrics` | walkthrough |
| SIM-07 deterministic given seed | `Simulator` (seeded RNG) | `test_components::test_simulator_deterministic_given_seed` |

## §8 Component 4 — Controller Optimizer

| Req | Implementation | Test |
|-----|----------------|------|
| OPT-01 outer loop wrapping simulator | `component4_optimizer.py::ControllerOptimizer.optimize` | `test_components::test_optimizer_improves_objective` |
| OPT-02 selectable algorithms | `random/grid/nelder_mead/evolutionary/bayesian` | same |
| OPT-03 robust objective against uncertainty | `_robust_objective` (`risk_lambda`) | — |
| OPT-04 objective sensitivity per dimension | `_sensitivity` | `test_components::test_optimizer_improves_objective` |
| OPT-05 reject unstable configs | `_robust_objective` (`stability_std_blowup`) | `test_components::test_optimizer_rejects_unstable_config` |

## §9 Component 5 — Diagnostics and Visualization

| Req | Implementation | Test |
|-----|----------------|------|
| DIAG-01 FF-leakage (state vs FF correlation) | `component5_diagnostics.py::ff_leakage` | `test_components::test_ff_leakage_flags...` |
| DIAG-02 gain-mismatch instability | `gain_mismatch` | `test_components::test_gain_underestimate_diagnosed...` |
| DIAG-03 Controller-QR vs State-QR attribution | `qr_attribution` | — |
| DIAG-04 variance decomposition | `variance_decomposition` | walkthrough |
| DIAG-05 achievability verdict (MP/Harris analog) | `achievability_verdict` | `test_components::test_healthy_controller_declared_at_floor` |
| DIAG-06 visualizations | `visualize` | walkthrough (figures) |

## §10 Component 6 — All-Data Regression Machine

| Req | Implementation | Test |
|-----|----------------|------|
| REG-01 one engine, two configurations | `engine.py::EstimationEngine` | `test_integration::test_unified_engine_two_configurations` |
| REG-02 controller decoupling; innovation fallback | `component6_regression.py::RegressionMachine.fit` | `test_components::test_regression_decouples...`, `..._falls_back_to_innovation...` |
| REG-03 high-dim PLS/elastic-net/ridge | `_select_estimator`, `engine.py` estimators | — |
| REG-04 innovation-target option / FF candidates | `_innovation_target`, `_feedforward_candidates` | `test_integration::test_regression_feedforward_matches_diagnostics_object` |
| REG-05 forward validation, never k-fold | `_select_estimator` (forward CV) | `test_components::test_regression_uses_forward_validation_not_kfold` |
| REG-06 stability selection (block bootstrap) | `_stability_selection` | `test_components::test_regression_decouples...` |
| REG-07 low-variation handling + planner hand-off | `_low_variation` | — |
| REG-08 method discipline (no transformer default) | `_select_estimator` (candidates linear/PLS) | — |
| REG-09 physics-aware trace aggregation | documented default; `EventTable.y_matrix(reduce=…)` | — |

## §11 Interfaces / §12 Non-Functional

| Req | Implementation |
|-----|----------------|
| IF-01..06 shared schema & single shared services | `foundations/` single implementations imported across components |
| NFR-01 numpy/scipy stack, tests from scratch, GP surrogate | `foundations/stats.py`, `foundations/gp.py` |
| NFR-02 seedable/deterministic | RNGs threaded through every stochastic routine |
| NFR-03 provenance/auditability | `provenance.py` attached to every artifact |
| NFR-04 synthetic smoke tests w/ injected faults + refusals | `synthetic.py`, `tests/` (FB-08, NF-02, VAL-01) |
| NFR-05 reference docs + modules | `docs/`, `README.md`, `examples/walkthrough.py` |
| NFR-06 honesty over convenience | refusals surfaced as outputs, routed to planner |
