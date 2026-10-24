"""Per-component behavior."""

import numpy as np

from r2r_control import synthetic
from r2r_control.components.fb_identification import FBIdentifier
from r2r_control.components.simulator import Simulator, PlantConfig, ControllerConfig
from r2r_control.components.controller_optimizer import ControllerOptimizer, SearchDimension
from r2r_control.components.diagnostics import DiagnosticsEngine
from r2r_control.components.regression_machine import RegressionMachine
from r2r_control.components.experiment_planner import (ExperimentPlanner, PlannerConfig,
                                                       closed_loop_excitation_discount)
from tests.conftest import make_sim


# ---- Component 1 ----------------------------------------------------------
def test_fb_identification_recovers_gain(dataset):
    fb = FBIdentifier(estimator="svd").identify(dataset.inline, dataset.doe, doe_age_days=20)
    assert fb.M.shape == dataset.truth.M.shape
    assert np.linalg.norm(fb.M - dataset.truth.M) / np.linalg.norm(dataset.truth.M) < 0.1
    assert fb.param_cov.shape == (fb.n_knob, fb.n_knob)
    assert fb.A is not None
    art = fb.as_artifact()
    assert art.kind == "fb" and "A1" in art.assumption_ledger()


def test_fb_identification_flags_confounded_knobs(confounded_dataset):
    fb = FBIdentifier().identify(confounded_dataset.inline, None)
    assert fb.identifiable.effective_rank <= fb.identifiable.n_directions


# ---- Component 3 ----------------------------------------------------------
def test_control_reduces_variance(dataset):
    res = make_sim(dataset.truth, mpc_R=0.05, move_limit=1.0).simulate(n_steps=300)
    assert np.all(res.metrics_on.std <= res.metrics_off.std)
    assert np.all(res.metrics_on.cpk >= res.metrics_off.cpk)


def test_simulator_deterministic(dataset):
    s = make_sim(dataset.truth, mpc_R=0.1)
    assert np.allclose(s.simulate(n_steps=200).metrics_on.std,
                       s.simulate(n_steps=200).metrics_on.std)


def test_simulator_mc_bands_with_model_uncertainty(dataset):
    t = dataset.truth
    sim = Simulator(M=t.M, targets=t.targets, lsl=t.targets - 3, usl=t.targets + 3,
                    u0=t.u0, rel_uncertainty_M=0.3, seed=0)
    res = sim.simulate(n_steps=150, n_mc=8)
    assert np.all(res.metrics_on.bands["std_p95"] >= res.metrics_on.bands["std_p05"])


# ---- Component 4 ----------------------------------------------------------
def test_optimizer_improves_and_reports_sensitivity(dataset):
    t = dataset.truth

    def factory(cfg):
        return make_sim(t, **cfg)

    dims = [SearchDimension("mpc_R", 1e-3, 1.0, log=True),
            SearchDimension("move_limit", 0.1, 1.5)]
    opt = ControllerOptimizer(factory, dims, objective_metric="std", n_steps=150, n_mc=3, seed=1)
    res = opt.optimize(algorithm="bayesian", budget=16)
    assert res.best_objective < opt._robust_objective(np.array([0.99, 0.0]))
    assert set(res.sensitivity) == {"mpc_R", "move_limit"}


# ---- Component 5 ----------------------------------------------------------
def test_healthy_controller_at_floor_and_gain_ok(dataset):
    res = make_sim(dataset.truth, mpc_R=0.05, move_limit=1.0).simulate(n_steps=300)
    d = DiagnosticsEngine().diagnose(res.y_on, res.u, dataset.truth.M,
                                     state_est=res.state_est, ff_inputs=res.ff_inputs,
                                     innovation=res.innovation)
    assert d.excitation["sufficient"]
    assert d.gain_mismatch["diagnosis"] == "none"
    assert d.achievability["at_achievability_floor"]
    assert len(d.precedence_log) == 6


def test_diagnostics_renders_figures(dataset, tmp_path):
    res = make_sim(dataset.truth, mpc_R=0.05, move_limit=1.0).simulate(n_steps=150)
    d = DiagnosticsEngine().diagnose(res.y_on, res.u, dataset.truth.M,
                                     innovation=res.innovation, figure_dir=str(tmp_path))
    assert len(d.figures) == 3


# ---- Component 6 ----------------------------------------------------------
def test_regression_decouples_and_finds_ff():
    ds = synthetic.make_dataset(seed=5, ff_leak=0.7)
    res = RegressionMachine(seed=0, n_stability_resamples=40).fit(
        ds.inline, "y0", M=ds.truth.M, u0=ds.truth.u0,
        adjustable_features=ds.inline.knob_cols_used)
    assert res.decoupled
    assert res.feedforward_candidates


def test_regression_innovation_fallback(dataset):
    res = RegressionMachine(seed=0, n_stability_resamples=20).fit(
        dataset.inline, "y0", control_model_error_known=False,
        adjustable_features=dataset.inline.knob_cols_used)
    assert res.target_kind == "innovation"
    assert not res.decoupled


# ---- Component 2 ----------------------------------------------------------
def test_planner_proposes_only_adjustable(confounded_dataset):
    ds = confounded_dataset
    res = ExperimentPlanner(PlannerConfig(budget=5), seed=1).plan(
        ds.inline.regressors(), ds.inline.regressor_names, ds.inline.knob_cols_used)
    assert res.gap.unidentifiable_features
    for p in res.proposals:
        assert set(p.setpoint) <= set(ds.inline.knob_cols_used)
    assert any("MANUAL APPROVAL" in line for line in res.gate_log)


def test_planner_respects_integer_dims(confounded_dataset):
    ds = confounded_dataset
    cfg = PlannerConfig(budget=3, level_low=0.0, level_high=5.0, integer_dims=[0])
    res = ExperimentPlanner(cfg, seed=1).plan(ds.inline.regressors(), ds.inline.regressor_names,
                                             ds.inline.knob_cols_used)
    first = ds.inline.knob_cols_used[0]
    for p in res.proposals:                                           # E11
        assert float(p.setpoint[first]).is_integer()


def test_closed_loop_discounts_information(confounded_dataset):
    # A6/E9: the same experiments are worth less if the loop stays closed
    ds = confounded_dataset
    args = (ds.inline.regressors(), ds.inline.regressor_names, ds.inline.knob_cols_used)
    open_ = ExperimentPlanner(PlannerConfig(budget=3, loop_status="open"), seed=1).plan(*args)
    closed = ExperimentPlanner(PlannerConfig(budget=3, loop_status="closed"), seed=1).plan(*args)
    assert closed_loop_excitation_discount("closed") < 1.0
    assert (closed.proposals[0].information_gain
            < open_.proposals[0].information_gain)
    assert "excitation discount 0.40" in closed.gate_log[0]


# ---- Closure fixes: Component 1 (KI-1, KI-2, KI-5) ------------------------
def _rel(A, B):
    return np.linalg.norm(A - B) / np.linalg.norm(B)


def test_closed_loop_inline_rows_do_not_bias_the_gain():
    # KI-1: with a DOE, closed-loop inline rows are excluded, so fusing never makes the
    # gain worse than the DOE alone (previously 10-31% error vs 2-5%)
    from r2r_control.estimation.sensitivity.core import SensitivityCore
    for seed in range(4):
        ds = synthetic.make_dataset(seed=seed, doe_age_days=20, ff_leak=0.4)
        fb = FBIdentifier().identify(ds.inline, ds.doe, doe_age_days=20)
        doe = SensitivityCore(estimator="svd").fit(ds.doe.used_knobs() - ds.truth.u0,
                                                   ds.doe.y_matrix())
        assert fb.gain_source == "DOE only"
        assert _rel(fb.M, ds.truth.M) <= 1.5 * _rel(doe.coef.T, ds.truth.M) + 1e-3
        assert _rel(fb.M, ds.truth.M) < 0.1
        assert any("EXCLUDED from gain identification" in n for n in fb.provenance.notes)


def test_endogenous_fallback_is_flagged_with_bias_floor(confounded_dataset):
    # KI-1/DQ-06: without a DOE the closed-loop fallback is reported, not trusted
    fb = FBIdentifier().identify(confounded_dataset.inline, None)
    assert fb.gain_source.startswith("closed-loop inline")
    assert fb.bias_rel_uncertainty >= 1.0 and fb.relative_uncertainty() >= 1.0
    assert {kn for _, kn in fb.low_confidence_entries} == set(fb.knob_names)
    assert any("exogenous" in a.name and not a.holds for a in fb.provenance.assumptions)


def test_open_loop_inline_is_used_and_checked_against_doe():
    # exogenous inline rows (no feedback) do enter the gain, with a DOE-consistency bias term
    ds = synthetic.make_dataset(seed=2, control_gain=0.0, override_fraction=0.5,
                                doe_age_days=20)
    fb = FBIdentifier().identify(ds.inline, ds.doe, doe_age_days=20)
    assert fb.gain_source == "DOE + open-loop inline"
    assert fb.bias_rel_uncertainty >= 0.0
    assert _rel(fb.M, ds.truth.M) < 0.1


def test_reported_gain_uncertainty_is_calibrated():
    # KI-2: actual relative error and reported uncertainty are the same order
    # (previously the error was 2.4-4.8x the reported value)
    ratios = []
    for seed in range(6):
        ds = synthetic.make_dataset(seed=seed, doe_age_days=20)
        fb = FBIdentifier().identify(ds.inline, ds.doe, doe_age_days=20)
        ratios.append(_rel(fb.M, ds.truth.M) / fb.relative_uncertainty())
    assert 0.3 < np.median(ratios) < 2.0
    assert max(ratios) < 2.5


def test_ff_gains_estimated_from_counterfactual():
    # KI-5: C1 publishes FF gains for the simulator
    ds = synthetic.make_dataset(seed=0, doe_age_days=20, ff_leak=0.7)
    fb = FBIdentifier().identify(ds.inline, ds.doe, doe_age_days=20)
    true_ff = 0.7 * ds.truth.FF_gain
    assert fb.FF_gain.shape == true_ff.shape
    assert _rel(fb.FF_gain, true_ff) < 0.5
    assert fb.A is not None and np.max(np.abs(np.linalg.eigvals(fb.A))) < 1.0


def test_inline_down_weight_verdict_carries_a_weight():
    # KI-1: a DOWN_WEIGHT verdict now changes the weight (it was always 1.0)
    from r2r_control.core.data_quality import DataQualityLayer
    from r2r_control.contracts import Verdict
    ds = synthetic.make_dataset(seed=0, metrology_sparsity=0.99)
    rep = DataQualityLayer(inline_downweight=0.4).assess_inline(ds.inline)
    assert rep.decision == Verdict.DOWN_WEIGHT and rep.weight == 0.4


# ---- Closure fixes: Component 2 (KI-6, KI-4 px_weighting, KI-7 E11/A6) ------
def test_planner_targets_the_gap(confounded_dataset):
    ds = confounded_dataset
    args = (ds.inline.regressors(), ds.inline.regressor_names, ds.inline.knob_cols_used)
    flat = ExperimentPlanner(PlannerConfig(budget=6, gap_weight=0.0), seed=1).plan(*args)
    aimed = ExperimentPlanner(PlannerConfig(budget=6, gap_weight=8.0), seed=1).plan(*args)
    assert aimed.gap.directions
    assert (np.mean([p.gap_alignment for p in aimed.proposals])
            > np.mean([p.gap_alignment for p in flat.proposals]))
    assert aimed.proposals[0].purpose.startswith("close identifiability gap")
    assert len(aimed.seed_design) > 0


def test_planner_uses_target_features_and_says_when_there_is_no_gap(dataset):
    args = (dataset.inline.regressors(), dataset.inline.regressor_names,
            dataset.inline.knob_cols_used)
    none = ExperimentPlanner(PlannerConfig(budget=3), seed=0).plan(*args)
    assert not none.gap.directions
    assert none.proposals[0].purpose.startswith("refine model")
    k = dataset.inline.knob_cols_used[1]
    tgt = ExperimentPlanner(PlannerConfig(budget=3), seed=0).plan(*args, target_features=[k])
    assert tgt.gap.unidentifiable_features == [k]
    assert np.mean([p.gap_alignment for p in tgt.proposals]) > 0.5


def test_planner_engineering_units_cost_and_categories(confounded_dataset):
    ds = confounded_dataset
    knobs = ds.inline.knob_cols_used
    bounds = {knobs[0]: (10.0, 20.0), knobs[1]: (0.0, 4.0)}
    cfg = PlannerConfig(budget=4, knob_bounds=bounds, integer_dims=[1],
                        categorical_levels={2: [-1.0, 0.0, 1.0]}, run_cost=2.0,
                        knob_cost={knobs[0]: 1.0}, px_weighting="central")
    res = ExperimentPlanner(cfg, seed=2).plan(ds.inline.regressors(),
                                              ds.inline.regressor_names, knobs)
    for p in res.proposals:
        eng = p.setpoint_engineering
        assert 10.0 <= eng[knobs[0]] <= 20.0
        assert float(eng[knobs[1]]).is_integer()            # integer in engineering units
        assert p.setpoint[knobs[2]] in (-1.0, 0.0, 1.0)      # categorical level (E11)
        assert p.cost >= 2.0
    assert res.seed_design_engineering is not None


def test_planner_rejects_unknown_px_weighting():
    import pytest
    from r2r_control.components.experiment_planner import CodedSpace
    with pytest.raises(ValueError):
        CodedSpace(["a"], px_weighting="bogus")


def test_rejection_fraction_calibrated_from_residual_excitation():
    # A6/E9: the measured surviving share replaces the configured guess
    rng = np.random.default_rng(0)
    commanded = rng.standard_normal(500)
    planner = ExperimentPlanner(PlannerConfig(loop_status="closed"))
    frac = planner.calibrate_rejection(commanded, 0.5 * commanded)
    assert abs(frac - 0.75) < 1e-9
    assert abs(planner.excitation_discount - 0.25) < 1e-9


# ---- Closure fixes: Component 6 (KI-7 VAL-03/04, REG-07, REG-09) -----------
def test_regression_holds_out_control_off_gold(dataset):
    res = RegressionMachine(seed=0, n_stability_resamples=20).fit(
        dataset.inline, "y0", M=dataset.truth.M, u0=dataset.truth.u0,
        adjustable_features=dataset.inline.knob_cols_used)
    assert res.gold["n_gold_episodes"] >= 1
    assert np.isfinite(res.gold["gold_error"])


def test_regression_routes_low_variation_knob_to_planner(dataset):
    import pandas as pd
    from r2r_control.contracts import EventTable
    t = dataset.inline
    frame = t.frame.copy()
    frame["u2_used"] = frame["u2_used"].mean() + 1e-6 * np.arange(len(frame))
    tab = EventTable(frame=frame, knob_cols_recommended=t.knob_cols_recommended,
                     knob_cols_used=t.knob_cols_used, ff_cols=t.ff_cols, y_cols=t.y_cols,
                     y_time_cols=t.y_time_cols, time_col=t.time_col,
                     control_off_col=t.control_off_col)
    res = RegressionMachine(seed=0, n_stability_resamples=10).fit(
        tab, "y0", control_model_error_known=False, adjustable_features=t.knob_cols_used)
    assert "u2_used" in res.low_variation_adjustable                # REG-07


def _trace_table(seed, kind):
    import pandas as pd
    from r2r_control.contracts import EventTable
    rng = np.random.default_rng(seed)
    n = 240
    traces, y = [], []
    for _ in range(n):
        L = int(rng.integers(30, 50))
        base = rng.standard_normal(L) * 0.3
        if kind == "ramp":
            slope = rng.standard_normal()
            tr = base + slope * np.linspace(0, 1, L)
            target = 2.0 * slope
        else:                                   # transient at 60% of the trace
            spike = rng.standard_normal()
            tr = base.copy()
            tr[int(0.6 * L)] += 3.0 * spike
            tr[int(0.6 * L) + 1] -= 3.0 * spike  # zero-mean blip: mean/ramp hide it
            target = 2.0 * spike
        traces.append(tr)
        y.append([np.array([target + 0.1 * rng.standard_normal()])])
    frame = pd.DataFrame({"timestamp": np.arange(n), "u_rec": np.zeros(n),
                          "u_used": rng.standard_normal(n), "trace": traces,
                          "y": [c[0] for c in y]})
    return EventTable(frame=frame, knob_cols_recommended=["u_rec"], knob_cols_used=["u_used"],
                      y_cols=["y"], time_col="timestamp", trace_cols=["trace"])


def test_traces_aggregated_by_default_and_raw_kept_only_when_needed():
    # REG-09: a ramp-rate signal survives aggregation; a zero-mean transient does not
    ramp = RegressionMachine(seed=0, n_stability_resamples=10).fit(
        _trace_table(0, "ramp"), "y", control_present=False)
    assert not ramp.traces["trace"]["retain_raw"]
    assert "trace_ramp_rate" in ramp.importance.names
    blip = RegressionMachine(seed=0, n_stability_resamples=10).fit(
        _trace_table(1, "transient"), "y", control_present=False)
    assert blip.traces["trace"]["retain_raw"]
    assert any(n.startswith("trace_raw") for n in blip.importance.names)


# ---- Closure fixes: Component 5 (KI-8 FF multiple testing, KI-9 y_pred) ------
def test_ff_leakage_controls_false_positives():
    from r2r_control.components.diagnostics import ff_leakage
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(40):
        state = np.cumsum(rng.standard_normal((200, 3)), axis=0)
        ff = rng.standard_normal((200, 8))
        hits += len(ff_leakage(state, ff)["leaking_inputs"])
    assert hits <= 3          # ~0 expected; uncorrected testing gave many more


def test_y_pred_consistency_separates_gain_error_from_tracking_lag():
    # §13.5: the y_pred column disambiguates inaccurate M from FB tracking lag
    # (the controller's model is passed as M_model; FF inputs are controlled for)
    cases = {"gain": dict(gain_underestimate=2.0), "lag": dict(estimator_gain=0.03),
             "ok": dict(estimator_gain=0.7), "ff": dict(ff_leak=0.7)}
    expected = {"gain": "M_inaccurate", "lag": "fb_tracking_lag", "ok": "consistent",
                "ff": "M_or_FF_model_error"}
    for seed in (3, 4):
        verdicts = {}
        for name, kw in cases.items():
            ds = synthetic.make_dataset(seed=seed, **kw)
            inl = ds.inline
            d = DiagnosticsEngine().diagnose(inl.y_matrix(), inl.used_knobs(),
                                             ds.truth.config["M_ctrl"], ff_inputs=inl.ff(),
                                             y_pred=inl.y_pred_matrix(),
                                             exclude=inl.control_off_mask())
            verdicts[name] = d.consistency["verdict"]
            if name == "ff":
                assert any("FF model" in r for r in d.routed_to_planner)
        assert verdicts == expected


def test_simulator_y_pred_flags_wrong_model_gain(dataset):
    t = dataset.truth
    res = make_sim(t, model_gain_scale=0.5, mpc_R=0.05, move_limit=1.0).simulate(n_steps=300)
    d = DiagnosticsEngine().diagnose(res.y_on, res.u, t.M * 0.5, innovation=res.innovation,
                                     y_pred=res.y_pred)
    assert d.consistency["verdict"] == "M_inaccurate"
    assert d.qr_attribution["primary_suspect"] == "deferred"


# ---- Component 4: OPT-05 --------------------------------------------------
def test_optimizer_rejects_unstable_configurations(dataset):
    t = dataset.truth

    def factory(cfg):
        # a 4x gain underestimate with no move suppression drives the loop unstable
        return make_sim(t, model_gain_scale=0.25, **cfg)

    dims = [SearchDimension("mpc_R", 1e-6, 1e-4, log=True),
            SearchDimension("move_limit", 50.0, 100.0)]
    res = ControllerOptimizer(factory, dims, n_steps=100, n_mc=1, seed=0).optimize(
        algorithm="random", budget=4)
    assert res.rejected_unstable > 0


# ---- Closure fixes: Component 3 (KI-3 delay, KI-4 MPC) ---------------------
def test_mpc_enforces_move_limits_exactly_not_by_clipping(dataset):
    from r2r_control.controller import MPCController
    M, u0 = dataset.truth.M, dataset.truth.u0
    lim, d_hat = 0.1, np.array([1.0, 0.0, 0.0])     # only some knobs hit the limit
    ctrl = MPCController(M, u0, mpc_Q=1.0, mpc_R=0.01, horizon=1, move_limit=lim)
    free = MPCController(M, u0, mpc_Q=1.0, mpc_R=0.01, horizon=1)
    target = np.zeros(M.shape[0])
    u_opt = ctrl.move(ctrl.disturbance_trajectory(d_hat), target, u0)
    u_clip = u0 + np.clip(free.move(free.disturbance_trajectory(d_hat), target, u0) - u0,
                          -lim, lim)

    def cost(u):
        return float(np.sum((M @ (u - u0) + d_hat) ** 2) + 0.01 * np.sum((u - u0) ** 2))
    assert np.all(np.abs(u_opt - u0) <= lim + 1e-9)
    assert cost(u_opt) < 0.5 * cost(u_clip)         # the clipped move is not the optimum
    assert cost(u_clip) > cost(u0)                  # here it is worse than not moving


def test_mpc_horizon_and_absolute_bounds_take_effect(dataset):
    from r2r_control.controller import MPCController
    M, u0 = dataset.truth.M, dataset.truth.u0
    d_hat, target, A = np.ones(M.shape[0]), np.zeros(M.shape[0]), 0.5 * np.eye(M.shape[0])
    moves = [MPCController(M, u0, 1.0, 1.0, horizon=h, A_d=A).move(
        MPCController(M, u0, 1.0, 1.0, horizon=h, A_d=A).disturbance_trajectory(d_hat),
        target, u0) for h in (1, 10)]
    assert not np.allclose(moves[0], moves[1], atol=1e-3)
    lo, hi = (u0 - 0.2).tolist(), (u0 + 0.2).tolist()
    res = make_sim(dataset.truth, mpc_R=0.01, move_limit=1.0, u_lower=lo,
                   u_upper=hi).simulate(n_steps=150)
    assert np.all(res.u >= np.array(lo) - 1e-9) and np.all(res.u <= np.array(hi) + 1e-9)


def test_metrology_delay_is_simulated(dataset):
    # KI-3: a delayed measurement reaches the estimator late, so control degrades
    stds = [make_sim(dataset.truth, mpc_R=0.05, move_limit=1.0, metrology_delay=d,
                     mhe_horizon=3).simulate(n_steps=300).metrics_on.std.mean()
            for d in (0, 2)]
    assert stds[1] > stds[0] * 1.1
    res = make_sim(dataset.truth, metrology_delay=2, mhe_horizon=3).simulate(n_steps=50)
    assert np.all(np.isnan(res.innovation[-2:]))    # last measurements not yet arrived
    assert np.all(np.isfinite(res.y_pred))
