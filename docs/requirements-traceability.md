# Requirements Traceability Matrix

Maps each requirement of the revised specification to its implementation
(paths relative to `src/r2r_control/`) and test (paths relative to `tests/`).
Requirement IDs also appear in the source docstrings.

## §1.3 / §1.4 Unification and load-bearing assumptions

| Req | Implementation | Test |
|---|---|---|
| §1.3 shared sensitivity core, separate wrappers | `estimation/sensitivity/core.py`, `estimation/dynamics/`, `estimation/regularized/` | `estimation/test_estimation.py::test_components_share_core_but_have_their_own_layers` |
| A1 gain time-invariance guard (E1) | `components/diagnostics/gain_mismatch.py::gain_drift_monitor` | `synthetic/faults/test_faults.py::test_gain_drift_monitor_*` |
| A2 `u0` validity horizon (E3) | `core/counterfactual/baseline.py::estimate_baseline` | `core/test_core.py::test_baseline_*` |
| A3 `M` accuracy / phantom FF (E2) | `components/regression_machine/decouple.py` | `synthetic/faults/test_faults.py::test_phantom_*` |
| A4 out-of-sequence measurements (E7) | `controller/mhe/oosm.py`, `components/simulator/run.py` | `synthetic/limits/test_limits.py::test_oosm_*` |
| A5 control-off representativeness (E8) | `core/validation/control_off.py` | `core/test_core.py::test_control_off_gold_excludes_anomalous_episode` |
| A6 closed-loop excitation discount (E9) | `components/experiment_planner/loop_status.py` | `components/test_components.py::test_closed_loop_discounts_information` |
| Assumption ledger on artifacts | `contracts/artifact.py::ModelArtifact` | `contracts/test_contracts.py::test_model_artifact_records_load_bearing_assumptions` |

## §4.1 Data model

| Req | Implementation | Test |
|---|---|---|
| DM-01/02 event table, sparse metrology | `contracts/schema.py::EventTable` | fixtures |
| DM-03 `_is_measured` (E14) | `contracts/measured.py` | `contracts/test_contracts.py::test_is_measured_*`, `test_effective_count_uses_is_measured` |
| DM-04 shape contract | `contracts/shapes.py` | `contracts/test_contracts.py::test_shape_contract_fails_loudly` |
| DM-05 recommended vs used | `EventTable.recommended_used_divergence` | `contracts/test_contracts.py::test_recommended_used_alignment_enforced` |

## §4.2 Data quality

| Req | Implementation | Test |
|---|---|---|
| DQ-01..03 DOE: alias/resolution (E13), shift, staleness | `core/data_quality/doe.py` | `core/test_core.py::test_alias_structure_detects_aliasing`, `test_stale_doe_is_downweighted` |
| DQ-04..06 inline confounding, metrology, stationarity | `core/data_quality/inline.py` | via Component 1 |
| DQ-07 joint-use contract | `core/data_quality/joint.py` | via Component 1 |
| DQ-08 persisted decisions | `core/data_quality/decisions.py`, `contracts/provenance.py` | `test_integration.py::test_pipeline_end_to_end_and_serializable` |

## §4.3 Excitation

| Req | Implementation | Test |
|---|---|---|
| EX-01 VIF | `core/excitation/vif.py` | via data quality |
| EX-02 spectrum, rank | `core/excitation/spectrum.py` | `core/test_core.py::test_numerical_rank_*` |
| EX-03 unidentifiable directions | `core/excitation/directions.py` | `components/test_components.py::test_planner_proposes_only_adjustable` |
| EX-04 accidental excitation | `core/excitation/accidental.py` | via Component 1 |

## §4.4 Noise floor

| Req | Implementation | Test |
|---|---|---|
| NF-01 MP edge, Harris | `core/noise_floor/analytic.py` | `core/test_core.py::test_mp_edges` |
| NF-02/04 assumption check, empirical precedence | `core/noise_floor/select.py` | `core/test_core.py::test_analytic_on_balanced_empirical_on_unbalanced` |
| NF-03 structure-preserving surrogates, no plain shuffle (E6) | `core/noise_floor/empirical.py` | `core/test_core.py::test_structure_preserving_floor_*`, `test_plain_shuffle_surrogate_refused`, `test_empirical_floor_preserves_missingness` |
| NF-05 cleaning | `core/noise_floor/cleaning.py` | — |

## §4.5 Counterfactual

| Req | Implementation | Test |
|---|---|---|
| CF-01 subtraction, exact | `core/counterfactual/reconstruct.py` | `core/test_core.py::test_counterfactual_exact_on_synthetic_mimo`, `test_counterfactual_accepts_time_varying_u0` |
| CF-02 `u0` sourcing/validity | `core/counterfactual/baseline.py` | `core/test_core.py::test_baseline_*` |
| CF-03 MC bands | `core/counterfactual/bands.py` | `core/test_core.py::test_counterfactual_bands_widen_with_uncertainty` |
| CF-04 shared by C3/C5/C6 | imports in `components/` | architecture test |

## §4.6 Validation

| Req | Implementation | Test |
|---|---|---|
| VAL-01 refuse naive k-fold | `core/validation/forward_chaining.py::naive_kfold_guard` | `core/test_core.py::test_naive_kfold_refused` |
| VAL-02/03 forward chaining, purge/embargo | `forward_chaining.py`, `purge_embargo.py` | `core/test_core.py::test_forward_chaining_trains_on_past_with_purge` |
| VAL-04 control-off gold (A5) | `core/validation/control_off.py` | `core/test_core.py::test_control_off_gold_excludes_anomalous_episode` |
| VAL-05 block bootstrap (shared with NF-03) | `core/validation/block_bootstrap.py` | `core/test_core.py::test_block_permutation_preserves_marginal` |
| VAL-06 concept drift | `core/validation/concept_drift.py` | `core/test_core.py::test_concept_drift_detected` |

## §5 Component 1 — FB identification

| Req | Implementation | Test |
|---|---|---|
| FB-01 fuse DOE + inline | `components/fb_identification/identify.py` | `components/test_components.py::test_fb_identification_recovers_gain` |
| FB-02 order by singular-value gap | `estimation/dynamics/subspace.py`, `SensitivityCore` | `estimation/test_estimation.py::test_sensitivity_core_recovers_gain` |
| FB-03 ridge / SVD regularization | `estimation/sensitivity/core.py` | same |
| FB-04 identifiability limits → planner | `FBModel.low_confidence_entries` | `components/test_components.py::test_fb_identification_flags_confounded_knobs` |
| FB-05 accidental excitation | `identify.py` weights | — |
| FB-06 innovation diagnostics | `FBIdentifier._innovation_diagnostics` | walkthrough |
| FB-07 parameter covariance | `FBModel.param_cov` | `components/test_components.py::test_fb_identification_recovers_gain` |
| FB-08 refuse gain-vs-tuning separation | `identify.py` | `synthetic/limits/test_limits.py::test_fb_refuses_to_separate_gain_from_estimator_tuning` |

## §6 Component 2 — Experiment planner

| Req | Implementation | Test |
|---|---|---|
| EP-01/02 gap, adjustable only | `experiment_planner/gap.py` | `components/test_components.py::test_planner_proposes_only_adjustable` |
| EP-03 D/I-optimal seed | `experiment_planner/seed_design.py` | via planner |
| EP-04 GP infill, integer-aware (E11) | `experiment_planner/infill.py`, `core/gp.py` | `components/test_components.py::test_planner_respects_integer_dims`, `core/test_core.py::test_round_to_feasible_and_mixed_kernel` |
| EP-05 priority / value | `experiment_planner/value.py` | via planner |
| EP-06 manual approval | `experiment_planner/approval.py` | `test_planner_proposes_only_adjustable` |
| EP-07 gate workflow | `ExperimentPlanner.plan`, `pipeline/gates.py` | `test_integration.py` |
| EP-08 decisions as config, loop status (A6) | `PlannerConfig`, `loop_status.py` | `test_closed_loop_discounts_information` |

## §7 Component 3 — Simulator

| Req | Implementation | Test |
|---|---|---|
| SIM-01 control on/off via counterfactual | `components/simulator/run.py` | `components/test_components.py::test_control_reduces_variance` |
| SIM-02 MC model-error bands | `Simulator.simulate`, `_aggregate` | `test_simulator_mc_bands_with_model_uncertainty` |
| SIM-03/04 independent State QR / Controller QR, config | `controller/mhe/estimator.py`, `controller/mpc/` | via simulator |
| SIM-05 plant | `PlantConfig` | — |
| SIM-06 metrics + finite-sample CIs + gate (E10) | `metrics/` | `synthetic/limits/test_limits.py::test_min_sample_gate_*`, `test_cpk_interval_shrinks_with_samples` |
| SIM-07 deterministic | seeded RNG | `test_simulator_deterministic` |

## §8 Component 4 — Optimizer

| Req | Implementation | Test |
|---|---|---|
| OPT-01 outer loop over a simulator factory | `components/controller_optimizer/optimize.py` | `components/test_components.py::test_optimizer_improves_and_reports_sensitivity` |
| OPT-02 algorithms | random / grid / Nelder-Mead / evolutionary / Bayesian | same |
| OPT-03/04 robust objective, sensitivity | `_robust_objective`, `_sensitivity` | same |
| OPT-05 reject unstable | `_robust_objective` | — |

## §9 Component 5 — Diagnostics

| Req | Implementation | Test |
|---|---|---|
| DIAG-01 FF leakage | `diagnostics/ff_leakage.py` | `synthetic/faults/test_faults.py::test_ff_leakage_detected` |
| DIAG-02 gain mismatch | `diagnostics/gain_mismatch.py` | `test_gain_underestimate_diagnosed_as_overcorrection` |
| DIAG-03 QR attribution | `diagnostics/qr_attribution.py` | via precedence |
| DIAG-04 variance decomposition | `diagnostics/variance_decomp.py` | `test_tight_control_routes_gain_and_qr_to_planner` |
| DIAG-05 achievability | `diagnostics/achievability.py` | `test_healthy_controller_at_floor_and_gain_ok` |
| DIAG-06 visualizations | `diagnostics/viz.py` | `test_diagnostics_renders_figures` |
| DIAG-07 precedence + excitation gate (E4, E5) | `diagnostics/precedence.py` | `synthetic/limits/test_limits.py::test_tight_control_routes_gain_and_qr_to_planner`, `test_qr_deferred_until_gain_resolved` |

## §10 Component 6 — Regression machine

| Req | Implementation | Test |
|---|---|---|
| REG-01 shared core, separate wrapper | `components/regression_machine/__init__.py` + `estimation/regularized/` | `estimation/test_estimation.py::test_components_share_core_but_have_their_own_layers` |
| REG-02 decoupling + innovation fallback, phantom FF (A3) | `regression_machine/decouple.py` | `test_regression_decouples_and_finds_ff`, `test_regression_innovation_fallback`, `test_phantom_*` |
| REG-03 ridge / PLS / elastic net | `estimation/regularized/` | `estimation/test_estimation.py::test_pls_*` |
| REG-04 innovation target, FF candidates | `regression_machine/target.py`, `select.py` | as above |
| REG-05 forward validation only | `RegressionMachine._select_estimator` | via regression tests |
| REG-06 group-aware stability selection (E12) | `estimation/regularized/stability_selection.py` | `estimation/test_estimation.py::test_group_aware_stability_selection` |
| REG-07 low variation → planner | `regression_machine/select.py::low_variation` | — |
| REG-08 method discipline | `_select_estimator` candidates | — |
| REG-09 trace aggregation | `regression_machine/traces.py` | — |

## §11 Interfaces / §12 Non-functional / §16 Layout

| Req | Implementation | Test |
|---|---|---|
| IF-01 shared artifact schema | `contracts/artifact.py::ModelArtifact`, `FBModel.as_artifact` | `components/test_components.py::test_fb_identification_recovers_gain` |
| IF-06 single shared services | `core/` | `test_integration.py::test_shared_noise_floor_is_one_implementation` |
| §11 components decoupled | optimizer takes a factory; pipeline wires | `test_architecture.py::test_layer_dependency_order` |
| §16 layer order enforced | — | `test_architecture.py` |
| NFR-02 seeded / reproducible, explicit config | `config/`, seeded RNGs | `test_integration.py::test_pipeline_is_reproducible`, `test_config_rejects_unknown_fields` |
| NFR-03 provenance | `contracts/provenance.py` | `test_pipeline_end_to_end_and_serializable` |
| NFR-04 synthetic faults + refusals | `synthetic.py`, `tests/synthetic/` | faults + limits suites |
| NFR-05 docs | `README.md`, `docs/`, `examples/walkthrough.py` | — |
| NFR-06 honesty over convenience | refusals are outputs routed to the planner | limits suite |
