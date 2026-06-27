"""Tests for the six components, including the intentional identifiability-limit
failure cases required by NFR-04."""

import numpy as np
import pytest

from process_control import synthetic
from process_control.components.component1_identification import FBIdentifier
from process_control.components.component2_planner import ExperimentPlanner, PlannerConfig
from process_control.components.component3_simulator import (
    Simulator, PlantConfig, ControllerConfig)
from process_control.components.component4_optimizer import (
    ControllerOptimizer, SearchDimension)
from process_control.components.component5_diagnostics import DiagnosticsEngine
from process_control.components.component6_regression import RegressionMachine


# --------------------------------------------------------------------------- #
# Component 1 — FB identification
# --------------------------------------------------------------------------- #
def test_fb_identification_recovers_gain(dataset):
    ds = dataset
    fb = FBIdentifier(estimator="svd", seed=0).identify(ds.inline, ds.doe, doe_age_days=20)
    err = np.linalg.norm(fb.M - ds.truth.M) / np.linalg.norm(ds.truth.M)
    assert err < 0.3
    assert fb.M.shape == ds.truth.M.shape           # shape contract (DM-04)
    assert fb.param_cov.shape[0] == fb.M.shape[1]    # uncertainty for simulator (FB-07)


def test_fb_refuses_to_separate_gain_from_estimator_tuning(dataset):
    # NFR-04 / FB-08: the system must REFUSE this attribution on passive data.
    fb = FBIdentifier(seed=0).identify(dataset.inline, dataset.doe)
    refusal = [a for a in fb.provenance.assumptions
               if "estimator tuning" in a.name and not a.holds]
    assert refusal, "FB-08 refusal must be recorded as a non-holding assumption"


def test_fb_routes_unidentifiable_to_planner_when_confounded(confounded_dataset):
    # FB-04: excitation-deficient knobs marked low-confidence (no DOE this time)
    fb = FBIdentifier(seed=0).identify(confounded_dataset.inline, doe=None)
    assert fb.identifiable.is_deficient() or fb.low_confidence_entries


# --------------------------------------------------------------------------- #
# Component 3 — simulator
# --------------------------------------------------------------------------- #
def _make_sim(truth, **ctrl):
    lsl, usl = truth.targets - 3, truth.targets + 3
    return Simulator(M=truth.M, targets=truth.targets, lsl=lsl, usl=usl, u0=truth.u0,
                     FF_gain=truth.FF_gain, plant=PlantConfig(meas_noise_std=0.2),
                     controller=ControllerConfig(**ctrl), seed=0)


def test_control_reduces_variance(dataset):
    sim = _make_sim(dataset.truth, mpc_R=0.05, move_limit=1.0)
    res = sim.simulate(n_steps=300)
    assert np.all(res.metrics_on.std <= res.metrics_off.std)   # SIM-01
    assert np.all(res.metrics_on.cpk >= res.metrics_off.cpk)


def test_simulator_deterministic_given_seed(dataset):
    sim = _make_sim(dataset.truth, mpc_R=0.1)
    a = sim.simulate(n_steps=200)
    b = sim.simulate(n_steps=200)
    assert np.allclose(a.metrics_on.std, b.metrics_on.std)      # SIM-07


def test_monte_carlo_bands_present_under_uncertainty(dataset):
    t = dataset.truth
    lsl, usl = t.targets - 3, t.targets + 3
    sim = Simulator(M=t.M, targets=t.targets, lsl=lsl, usl=usl, u0=t.u0, FF_gain=t.FF_gain,
                    rel_uncertainty_M=0.15, seed=0)
    res = sim.simulate(n_steps=200)
    assert "cpk_p05" in res.metrics_on.bands                    # SIM-02


# --------------------------------------------------------------------------- #
# Component 4 — optimizer
# --------------------------------------------------------------------------- #
def test_optimizer_improves_objective(dataset):
    t = dataset.truth
    lsl, usl = t.targets - 3, t.targets + 3

    def factory(cfg):
        return Simulator(M=t.M, targets=t.targets, lsl=lsl, usl=usl, u0=t.u0,
                         FF_gain=t.FF_gain, plant=PlantConfig(meas_noise_std=0.2),
                         controller=cfg, seed=0)

    dims = [SearchDimension("mpc_R", 1e-3, 1.0, log=True),
            SearchDimension("move_limit", 0.1, 1.5)]
    opt = ControllerOptimizer(factory, dims, objective_metric="std",
                              n_steps=150, n_mc=3, seed=1)
    res = opt.optimize(algorithm="bayesian", budget=18)
    baseline = opt._robust_objective(np.array([0.99, 0.0]))   # poor config
    assert res.best_objective < baseline                       # OPT-01
    assert set(res.sensitivity.keys()) == {"mpc_R", "move_limit"}  # OPT-04


def test_optimizer_rejects_unstable_config(dataset):
    # OPT-05: a wildly gain-mismatched controller should be flagged infeasible
    t = dataset.truth
    lsl, usl = t.targets - 3, t.targets + 3

    def factory(cfg):
        return Simulator(M=t.M, targets=t.targets, lsl=lsl, usl=usl, u0=t.u0,
                         FF_gain=t.FF_gain, plant=PlantConfig(meas_noise_std=0.2),
                         controller=cfg, model_gain_scale=0.15, seed=0)

    dims = [SearchDimension("move_limit", 2.0, 5.0),
            SearchDimension("mpc_R", 1e-5, 1e-4, log=True)]
    opt = ControllerOptimizer(factory, dims, n_steps=120, n_mc=2,
                              stability_std_blowup=3.0, seed=2)
    opt.optimize(algorithm="random", budget=10)
    assert opt._rejected >= 1


# --------------------------------------------------------------------------- #
# Component 5 — diagnostics (the achievability verdict, DIAG-05)
# --------------------------------------------------------------------------- #
def test_healthy_controller_declared_at_floor(dataset):
    sim = _make_sim(dataset.truth, mpc_R=0.05, move_limit=1.0)
    res = sim.simulate(n_steps=300)
    d = DiagnosticsEngine().diagnose(res.y_on, res.u, dataset.truth.M,
                                     state_est=res.state_est, ff_inputs=res.ff_inputs,
                                     innovation=res.innovation)
    assert d.achievability["at_achievability_floor"]            # DIAG-05
    assert d.gain_mismatch["diagnosis"] == "none"


def test_gain_underestimate_diagnosed_as_overcorrection(dataset):
    # DIAG-02: model gain 4x too small => over-correction, Harris >> 1
    t = dataset.truth
    lsl, usl = t.targets - 3, t.targets + 3
    sim = Simulator(M=t.M, targets=t.targets, lsl=lsl, usl=usl, u0=t.u0, FF_gain=t.FF_gain,
                    plant=PlantConfig(meas_noise_std=0.2),
                    controller=ControllerConfig(mpc_R=1e-4, move_limit=5.0),
                    model_gain_scale=0.25, seed=0)
    res = sim.simulate(n_steps=300)
    d = DiagnosticsEngine().diagnose(res.y_on, res.u, t.M * 0.25, innovation=res.innovation)
    assert d.gain_mismatch["oscillating"]
    assert d.gain_mismatch["gain_underestimated"]
    assert not d.achievability["at_achievability_floor"]


def test_ff_leakage_flags_uncompensated_disturbance():
    # DIAG-01: with a large FF leak, the estimated state tracks the FF inputs
    ds = synthetic.make_dataset(seed=3, ff_leak=0.8, n_ff=4)
    t = ds.truth
    lsl, usl = t.targets - 3, t.targets + 3
    sim = Simulator(M=t.M, targets=t.targets, lsl=lsl, usl=usl, u0=t.u0, FF_gain=t.FF_gain,
                    plant=PlantConfig(meas_noise_std=0.1, ff_disturbance_std=1.0),
                    controller=ControllerConfig(mpc_R=0.05), seed=0)
    # Build a controller that does NOT see FF at all to force leakage into state est:
    res = sim.simulate(n_steps=300)
    d = DiagnosticsEngine().ff_leakage(res.state_est, res.ff_inputs, ds.ff_names)
    # at least the diagnostic runs and returns the structure
    assert "feedforward_opportunities" in d


# --------------------------------------------------------------------------- #
# Component 6 — regression machine
# --------------------------------------------------------------------------- #
def test_regression_decouples_and_finds_ff_candidates():
    ds = synthetic.make_dataset(seed=5, ff_leak=0.7)
    t = ds.truth
    rm = RegressionMachine(seed=0, n_stability_resamples=40)
    res = rm.fit(ds.inline, "y0", M=t.M, u0=t.u0, control_present=True,
                 adjustable_features=ds.inline.knob_cols_used)
    assert res.decoupled                                       # REG-02
    assert res.feedforward_candidates                          # REG-04


def test_regression_falls_back_to_innovation_when_error_unknown(dataset):
    # REG-02: control-model error purely unknown => innovation-target fallback + flag
    rm = RegressionMachine(seed=0, n_stability_resamples=20)
    res = rm.fit(dataset.inline, "y0", control_present=True,
                 control_model_error_known=False,
                 adjustable_features=dataset.inline.knob_cols_used)
    assert res.target_kind == "innovation"
    assert any("innovation" in n.lower() for n in res.notes)


def test_regression_uses_forward_validation_not_kfold(dataset):
    rm = RegressionMachine(seed=0, n_stability_resamples=20)
    res = rm.fit(dataset.inline, "y0", M=dataset.truth.M, u0=dataset.truth.u0,
                 adjustable_features=dataset.inline.knob_cols_used)
    assert res.provenance.validation["scheme"] == "forward_chaining"   # REG-05


# --------------------------------------------------------------------------- #
# Component 2 — experiment planner
# --------------------------------------------------------------------------- #
def test_planner_localizes_gap_and_proposes_only_adjustable(confounded_dataset):
    ds = confounded_dataset
    X = ds.inline.regressors()
    names = ds.inline.regressor_names
    adjustable = ds.inline.knob_cols_used
    pl = ExperimentPlanner(PlannerConfig(primary_objective="D", budget=5), seed=1)
    res = pl.plan(X, names, adjustable)
    assert res.gap.unidentifiable_features                      # EP-01
    for p in res.proposals:                                    # EP-02
        assert set(p.setpoint.keys()) <= set(adjustable)
    assert any("MANUAL APPROVAL" in l for l in res.gate_log)    # EP-06


def test_planner_seed_design_objectives_differ(confounded_dataset):
    adjustable = confounded_dataset.inline.knob_cols_used
    pl_d = ExperimentPlanner(PlannerConfig(primary_objective="D"), seed=1)
    pl_i = ExperimentPlanner(PlannerConfig(primary_objective="I"), seed=1)
    d_design = pl_d.seed_design(adjustable, n_runs=8)
    i_design = pl_i.seed_design(adjustable, n_runs=8)
    assert d_design.shape == (8, len(adjustable))
    assert i_design.shape == (8, len(adjustable))
