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
    assert np.linalg.norm(fb.M - dataset.truth.M) / np.linalg.norm(dataset.truth.M) < 0.3
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
