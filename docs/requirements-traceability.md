# Requirements Traceability Matrix

Maps each requirement of the revised specification to its implementation
(paths relative to `src/r2r_control/`) and test (paths relative to `tests/`).
Requirement IDs also appear in the source docstrings.

A row here means the requirement has an implementation. It does not mean full
conformance. The 0.2.0 closure review found the requirements below to be only
partially met. 0.3.0 addresses them as shown. KI and R numbers refer to
section 8 of [`project-closure.md`](project-closure.md).

| Requirement | Gap found at 0.2.0 closure | KI | Status in 0.3.0 |
|---|---|---|---|
| DQ-06, DQ-07, FB-01 | Closed-loop inline rows entered gain identification at full weight; down-weight verdict not applied; endogeneity not detected | KI-1 | Fixed: feedback test, closed-loop rows excluded, verdict weights applied (residual: R-3) |
| FB-07, CF-03, SIM-02, A3 | Gain uncertainty ignored bias; pipeline used the configured `rel_uncertainty_M` | KI-2 | Fixed: bias term, identified value propagated (residual: R-4) |
| A4/E7, SIM-04 | Metrology delay only selected the route | KI-3 | Fixed: delay queue, in-horizon replay |
| SIM-03, EP-08 | `mpc_horizon` and `px_weighting` never read; one-step filter; clipped move limits | KI-4 | Fixed: receding-horizon MPC, exact limits, absolute bounds, `px_weighting` used (static-gain plant model remains, L3) |
| §5 dynamics, SIM-05 | Identified `A` and FF gains not passed to the pipeline's simulator | KI-5 | Fixed |
| EP-01/02/04/05 | Proposals not targeted at the gap; GP fitted to a zero response; coded units; uniform cost | KI-6 | Fixed |
| NF-05, REG-09, VAL-03, VAL-04, A6, E11 | Implemented but not wired in, partly untested | KI-7 | Fixed (VAL-04 trade-off: R-1; `subspace_order` still test-only: R-6) |
| SIM-06/E10, DIAG-01 | No autocorrelation discount in intervals; no multiple-comparison control in FF leakage | KI-8 | Fixed |
| §13.5 | No `y_pred` consistency diagnostic | KI-9 | Fixed (gain vs FF can be inseparable: R-2); revised spec still missing |

## §1.3 / §1.4 Unification and load-bearing assumptions

| Req | Implementation | Test |
|---|---|---|
| §1.3 shared sensitivity core, separate wrappers | `estimation/sensitivity/core.py`, `estimation/dynamics/`, `estimation/regularized/` | `estimation/test_estimation.py::test_components_share_core_but_have_their_own_layers` |
| A1 gain time-invariance guard (E1) | `components/diagnostics/gain_mismatch.py::gain_drift_monitor` | `synthetic/faults/test_faults.py::test_gain_drift_monitor_*` |
| A2 `u0` validity horizon (E3) | `core/counterfactual/baseline.py::estimate_baseline` | `core/test_core.py::test_baseline_*` |
| A3 `M` accuracy / phantom FF (E2) | `components/regression_machine/decouple.py` | `synthetic/faults/test_faults.py::test_phantom_*` |
| A4 out-of-sequence measurements (E7) | `controller/mhe/oosm.py`, `components/simulator/run.py` (delay queue, in-horizon replay) | `synthetic/limits/test_limits.py::test_oosm_*`, `components/test_components.py::test_metrology_delay_is_simulated` |
| A5 control-off representativeness (E8) | `core/validation/control_off.py`, used by the C6 gold holdout | `core/test_core.py::test_control_off_gold_excludes_anomalous_episode`, `components/test_components.py::test_regression_holds_out_control_off_gold` |
| A6 closed-loop excitation discount (E9) | `components/experiment_planner/loop_status.py`, `ExperimentPlanner.calibrate_rejection` | `components/test_components.py::test_closed_loop_discounts_information`, `test_rejection_fraction_calibrated_from_residual_excitation` |
| M3 exogenous gain rows (closed-loop endogeneity) | `core/excitation/feedback.py`, `core/data_quality/inline.py`, `fb_identification/identify.py` | `components/test_components.py::test_closed_loop_inline_rows_do_not_bias_the_gain`, `test_endogenous_fallback_is_flagged_with_bias_floor`, `test_open_loop_inline_is_used_and_checked_against_doe` |
| Assumption ledger on artifacts | `contracts/artifact.py::ModelArtifact` | `contracts/test_contracts.py::test_model_artifact_records_load_bearing_assumptions` |

## §4.1 Data model

| Req | Implementation | Test |
|---|---|---|
| DM-01/02 event table, sparse metrology | `contracts/schema.py::EventTable` | fixtures |
| DM-03 `_is_measured` (E14) | `contracts/measured.py` | `contracts/test_contracts.py::test_is_measured_*`, `test_effective_count_uses_is_measured` |
| DM-04 shape contract | `contracts/shapes.py` | `contracts/test_contracts.py::test_shape_contract_fails_loudly` |
| DM-05 recommended vs used | `EventTable.recommended_used_divergence` | `contracts/test_contracts.py::test_recommended_used_alignment_enforced` |
| §13.5 controller prediction `y_pred` | `EventTable.y_pred_cols`, `y_pred_matrix` | `components/test_components.py::test_y_pred_consistency_separates_gain_error_from_tracking_lag` |

## §4.2 Data quality

| Req | Implementation | Test |
|---|---|---|
| DQ-01..03 DOE: alias/resolution (E13), shift, staleness | `core/data_quality/doe.py` | `core/test_core.py::test_alias_structure_detects_aliasing`, `test_stale_doe_is_downweighted` |
| DQ-04..06 inline confounding, metrology, stationarity, feedback (endogeneity) | `core/data_quality/inline.py`, `core/excitation/feedback.py` | `components/test_components.py::test_closed_loop_inline_rows_do_not_bias_the_gain`, `test_inline_down_weight_verdict_carries_a_weight` |
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
| NF-05 cleaning | `core/noise_floor/cleaning.py`, `SensitivityCore(covariance_cleaning=…)` | `estimation/test_estimation.py::test_covariance_cleaning_is_wired_into_the_ridge_path` |

## §4.5 Counterfactual

| Req | Implementation | Test |
|---|---|---|
| CF-01 subtraction, exact | `core/counterfactual/reconstruct.py` | `core/test_core.py::test_counterfactual_exact_on_synthetic_mimo`, `test_counterfactual_accepts_time_varying_u0` |
| CF-02 `u0` sourcing/validity | `core/counterfactual/baseline.py` | `core/test_core.py::test_baseline_*` |
| CF-03 MC bands (identified uncertainty) | `core/counterfactual/bands.py`, `GatePipeline.effective_rel_uncertainty_M` | `core/test_core.py::test_counterfactual_bands_widen_with_uncertainty`, `test_integration.py::test_pipeline_propagates_identified_dynamics_and_uncertainty` |
| CF-04 shared by C3/C5/C6 | imports in `components/` | architecture test |

## §4.6 Validation

| Req | Implementation | Test |
|---|---|---|
| VAL-01 refuse naive k-fold | `core/validation/forward_chaining.py::naive_kfold_guard` | `core/test_core.py::test_naive_kfold_refused` |
| VAL-02/03 forward chaining, purge/embargo | `forward_chaining.py`, `purge_embargo.py`, `RegressionMachine(purge, embargo)` | `core/test_core.py::test_forward_chaining_trains_on_past_with_purge`, `test_embargo_drops_training_rows_after_the_test_block` |
| VAL-04 control-off gold (A5) | `core/validation/control_off.py`, `RegressionMachine(gold_holdout=True)` | `core/test_core.py::test_control_off_gold_excludes_anomalous_episode`, `components/test_components.py::test_regression_holds_out_control_off_gold` |
| VAL-05 block bootstrap (shared with NF-03) | `core/validation/block_bootstrap.py` | `core/test_core.py::test_block_permutation_preserves_marginal` |
| VAL-06 concept drift | `core/validation/concept_drift.py` | `core/test_core.py::test_concept_drift_detected` |

## §5 Component 1 — FB identification

| Req | Implementation | Test |
|---|---|---|
| FB-01 fuse DOE + inline (exogenous rows only) | `components/fb_identification/identify.py` | `components/test_components.py::test_fb_identification_recovers_gain`, `test_closed_loop_inline_rows_do_not_bias_the_gain` |
| FB-02 order by singular-value gap | `estimation/dynamics/subspace.py`, `SensitivityCore` | `estimation/test_estimation.py::test_sensitivity_core_recovers_gain` |
| FB-03 ridge / SVD regularization | `estimation/sensitivity/core.py` | same |
| FB-04 identifiability limits → planner | `FBModel.low_confidence_entries` | `components/test_components.py::test_fb_identification_flags_confounded_knobs` |
| FB-05 accidental excitation | `identify.py` weights | — |
| FB-06 innovation diagnostics | `FBIdentifier._innovation_diagnostics` | walkthrough |
| FB-07 parameter covariance and uncertainty (variance + bias) | `FBModel.param_cov`, `FBModel.relative_uncertainty` | `components/test_components.py::test_reported_gain_uncertainty_is_calibrated` |
| §5 dynamics and FF gains for the simulator | `estimation/dynamics/state_space.py`, `FBModel.A`, `FBModel.FF_gain` | `components/test_components.py::test_ff_gains_estimated_from_counterfactual`, `test_integration.py::test_pipeline_propagates_identified_dynamics_and_uncertainty` |
| FB-08 refuse gain-vs-tuning separation | `identify.py` | `synthetic/limits/test_limits.py::test_fb_refuses_to_separate_gain_from_estimator_tuning` |

## §6 Component 2 — Experiment planner

| Req | Implementation | Test |
|---|---|---|
| EP-01/02 gap, adjustable only, targeted | `experiment_planner/gap.py`, `infill.py::gap_alignment` | `components/test_components.py::test_planner_proposes_only_adjustable`, `test_planner_targets_the_gap`, `test_planner_uses_target_features_and_says_when_there_is_no_gap` |
| EP-03 D/I-optimal seed | `experiment_planner/seed_design.py` | via planner |
| EP-04 GP infill, integer/categorical-aware (E11), fitted hyperparameters | `experiment_planner/infill.py`, `experiment_planner/space.py`, `core/gp.py` | `components/test_components.py::test_planner_respects_integer_dims`, `test_planner_engineering_units_cost_and_categories`, `core/test_core.py::test_round_to_feasible_and_mixed_kernel` |
| EP-05 priority / value per unit cost | `experiment_planner/value.py` | `components/test_components.py::test_planner_engineering_units_cost_and_categories` |
| EP-06 manual approval | `experiment_planner/approval.py` | `test_planner_proposes_only_adjustable` |
| EP-07 gate workflow | `ExperimentPlanner.plan`, `pipeline/gates.py` | `test_integration.py` |
| EP-08 decisions as config, loop status (A6), p(x) | `PlannerConfig`, `loop_status.py`, `CodedSpace` | `test_closed_loop_discounts_information`, `test_planner_rejects_unknown_px_weighting` |

## §7 Component 3 — Simulator

| Req | Implementation | Test |
|---|---|---|
| SIM-01 control on/off via counterfactual | `components/simulator/run.py` | `components/test_components.py::test_control_reduces_variance` |
| SIM-02 MC model-error bands | `Simulator.simulate`, `_aggregate` | `test_simulator_mc_bands_with_model_uncertainty` |
| SIM-03/04 independent State QR / Controller QR, config | `controller/mhe/estimator.py`, `controller/mpc/controller.py` | `components/test_components.py::test_mpc_enforces_move_limits_exactly_not_by_clipping`, `test_mpc_horizon_and_absolute_bounds_take_effect` |
| SIM-05 plant (identified `A`, FF gains) | `PlantConfig`, `GatePipeline.simulator_factory` | `test_integration.py::test_pipeline_propagates_identified_dynamics_and_uncertainty` |
| SIM-06 metrics + finite-sample CIs (autocorrelation-aware) + gate (E10) | `metrics/`, `core/stats.py::effective_sample_size` | `synthetic/limits/test_limits.py::test_min_sample_gate_*`, `test_cpk_interval_shrinks_with_samples`, `core/test_core.py::test_effective_sample_size_discounts_autocorrelation` |
| SIM-07 deterministic | seeded RNG | `test_simulator_deterministic` |

## §8 Component 4 — Optimizer

| Req | Implementation | Test |
|---|---|---|
| OPT-01 outer loop over a simulator factory | `components/controller_optimizer/optimize.py` | `components/test_components.py::test_optimizer_improves_and_reports_sensitivity` |
| OPT-02 algorithms | random / grid / Nelder-Mead / evolutionary / Bayesian | same |
| OPT-03/04 robust objective, sensitivity | `_robust_objective`, `_sensitivity` | same |
| OPT-05 reject unstable | `_robust_objective` | `components/test_components.py::test_optimizer_rejects_unstable_configurations` |

## §9 Component 5 — Diagnostics

| Req | Implementation | Test |
|---|---|---|
| DIAG-01 FF leakage (Šidák + Holm) | `diagnostics/ff_leakage.py` | `synthetic/faults/test_faults.py::test_ff_leakage_detected`, `components/test_components.py::test_ff_leakage_controls_false_positives` |
| §13.5 model consistency via `y_pred` | `diagnostics/consistency.py`, precedence node 2b | `components/test_components.py::test_y_pred_consistency_separates_gain_error_from_tracking_lag`, `test_simulator_y_pred_flags_wrong_model_gain` |
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
| REG-07 low variation → planner | `regression_machine/select.py::low_variation` | `components/test_components.py::test_regression_routes_low_variation_knob_to_planner` |
| REG-08 method discipline | `_select_estimator` candidates | — |
| REG-09 trace aggregation, raw retention check | `regression_machine/traces.py`, `RegressionMachine.fit` | `components/test_components.py::test_traces_aggregated_by_default_and_raw_kept_only_when_needed` |

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
