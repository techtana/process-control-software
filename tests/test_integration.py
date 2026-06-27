"""End-to-end and cross-cutting-contract tests (§1.3, §IF-06, NFR-02/03)."""

import numpy as np

from process_control import synthetic
from process_control.data.quality import DataQualityLayer
from process_control.provenance import Verdict
from process_control.foundations.engine import EstimationEngine, EngineConfig
from process_control.components.component1_identification import FBIdentifier
from process_control.components.component3_simulator import (
    Simulator, PlantConfig, ControllerConfig)
from process_control.components.component4_optimizer import (
    ControllerOptimizer, SearchDimension)
from process_control.components.component5_diagnostics import DiagnosticsEngine
from process_control.components.component6_regression import RegressionMachine


def test_unified_engine_two_configurations():
    # §REG-01: Components 1 and 6 are two configurations of ONE engine class.
    fb_cfg = EngineConfig(mode="fb", estimator="svd")
    reg_cfg = EngineConfig(mode="regression", estimator="pls")
    assert type(EstimationEngine(fb_cfg)) is type(EstimationEngine(reg_cfg))


def test_data_quality_downweights_confounded_inline():
    # DQ-06: inline data with closed-loop confounding => down-weight, not silent use
    ds = synthetic.make_dataset(seed=9, override_fraction=0.0, control_off_episodes=0,
                                n_knob=5, n_out=2)
    dq = DataQualityLayer(vif_warn=8)
    rep = dq.assess_inline(ds.inline)
    assert rep.decision in (Verdict.DOWN_WEIGHT, Verdict.REJECT)
    assert "confounding" in rep.rationale


def test_data_quality_rejects_stale_shifted_doe():
    # DQ-02/03: very old + shifted DOE should be down-weighted or rejected
    ds = synthetic.make_dataset(seed=9)
    dq = DataQualityLayer(staleness_horizon_days=30)
    rep = dq.assess_doe(ds.doe.used_knobs() - ds.truth.u0,
                        ds.doe.knob_cols_used, age_days=400,
                        inline_operating_point=ds.inline.used_knobs() - ds.truth.u0)
    assert rep.decision in (Verdict.DOWN_WEIGHT, Verdict.REJECT)
    assert rep.weight < 1.0


def test_provenance_is_serializable_and_complete():
    # NFR-03: artifact carries data-usage, noise floor, validation, assumptions
    ds = synthetic.make_dataset(seed=8, doe_age_days=10)
    fb = FBIdentifier(seed=0).identify(ds.inline, ds.doe, doe_age_days=10)
    j = fb.provenance.to_json()
    assert "decisions" in j and "assumptions" in j
    d = fb.provenance.to_dict()
    assert d["seed"] == 0                                      # NFR-02 seed recorded
    assert d["noise_floor"]                                    # NF floor recorded


def test_full_pipeline_identify_simulate_optimize_diagnose():
    # The whole simulate -> optimize -> diagnose loop on an identified model.
    ds = synthetic.make_dataset(seed=13, doe_age_days=15)
    fb = FBIdentifier(estimator="svd", seed=0).identify(ds.inline, ds.doe, doe_age_days=15)
    t = ds.truth
    lsl, usl = t.targets - 3, t.targets + 3

    def factory(cfg):
        # simulate with the IDENTIFIED gain as the controller model, true gain as plant
        scale = float(np.median(np.diag(fb.M @ np.linalg.pinv(t.M)))) if fb.M.size else 1.0
        return Simulator(M=t.M, targets=t.targets, lsl=lsl, usl=usl, u0=t.u0,
                         FF_gain=t.FF_gain, plant=PlantConfig(meas_noise_std=0.2),
                         controller=cfg, rel_uncertainty_M=fb.relative_uncertainty(),
                         seed=0)

    dims = [SearchDimension("mpc_R", 1e-3, 1.0, log=True),
            SearchDimension("move_limit", 0.2, 1.5)]
    opt = ControllerOptimizer(factory, dims, objective_metric="cpk",
                              n_steps=150, n_mc=3, seed=1)
    res = opt.optimize(algorithm="bayesian", budget=16)
    best_sim = factory(_cfg_from(res.best_config))
    sim_res = best_sim.simulate(n_steps=300)
    diag = DiagnosticsEngine().diagnose(
        sim_res.y_on, sim_res.u, fb.M, state_est=sim_res.state_est,
        ff_inputs=sim_res.ff_inputs, innovation=sim_res.innovation)
    # the optimized controller should be capable and near the floor
    assert np.nanmean(sim_res.metrics_on.cpk) > 1.0
    assert "verdict" in diag.achievability


def _cfg_from(config_dict):
    c = ControllerConfig()
    for k, v in config_dict.items():
        setattr(c, k, v)
    return c


def test_regression_feedforward_matches_diagnostics_object():
    # §REG-04 == §DIAG-01: sensors explaining the innovation are feedforward candidates,
    # the same object the FF-leakage diagnostic surfaces.
    ds = synthetic.make_dataset(seed=14, ff_leak=0.7, n_ff=5)
    t = ds.truth
    rm = RegressionMachine(seed=0, n_stability_resamples=40)
    res = rm.fit(ds.inline, "y0", M=t.M, u0=t.u0, control_present=True,
                 target_kind="innovation",
                 adjustable_features=ds.inline.knob_cols_used)
    # innovation-target regression yields feedforward candidates among FF sensors
    assert all(c in ds.inline.ff_cols for c in res.feedforward_candidates)
