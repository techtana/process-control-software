"""Identifiability-limit tests (NFR-04/06): the system must refuse, not guess."""

import numpy as np

from r2r_control.components.fb_identification import FBIdentifier
from r2r_control.components.diagnostics import DiagnosticsEngine, UNIDENTIFIABLE
from r2r_control.controller.mhe.oosm import route_measurement
from r2r_control.metrics import compute_metrics, INSUFFICIENT
from tests.conftest import make_sim


def test_fb_refuses_to_separate_gain_from_estimator_tuning(dataset):
    # FB-08: structurally unidentifiable from passive data
    fb = FBIdentifier().identify(dataset.inline, dataset.doe)
    assert any("estimator tuning" in a.name and not a.holds for a in fb.provenance.assumptions)


def test_tight_control_routes_gain_and_qr_to_planner(dataset):
    # DIAG-07 / E4: Δu≈0 => gain and QR are unidentifiable, not guessed
    t = dataset.truth
    res = make_sim(t, mpc_R=5.0, move_limit=0.03).simulate(n_steps=300)
    d = DiagnosticsEngine().diagnose(res.y_on, res.u, t.M, innovation=res.innovation)
    assert not d.excitation["sufficient"]
    assert d.gain_mismatch["diagnosis"] == UNIDENTIFIABLE
    assert d.qr_attribution["primary_suspect"] == UNIDENTIFIABLE
    assert len(d.routed_to_planner) == 2
    # achievability needs no knob movement and is still reported
    assert "verdict" in d.achievability
    # no gain / estimator share is invented for the blocked nodes
    shares = d.variance_decomposition["shares"]
    assert shares["incorrect_process_gain"] == 0.0
    assert shares["suboptimal_estimator_tuning"] == 0.0


def test_qr_deferred_until_gain_resolved(dataset):
    # E5: with a gain mismatch present, QR attribution is deferred
    t = dataset.truth
    res = make_sim(t, model_gain_scale=0.25, mpc_R=1e-4, move_limit=5.0).simulate(n_steps=300)
    d = DiagnosticsEngine().diagnose(res.y_on, res.u, t.M * 0.25, innovation=res.innovation)
    assert d.qr_attribution["primary_suspect"] == "deferred"


def test_oosm_routes_over_horizon_measurement_to_slow_bias(dataset):
    # A4/E7: a measurement older than the horizon must not hit the state update
    assert route_measurement(delay=2, horizon=5) == "state_update"
    assert route_measurement(delay=9, horizon=5) == "slow_bias"
    res = make_sim(dataset.truth, mpc_R=0.05, metrology_delay=4, mhe_horizon=1).simulate(
        n_steps=200)
    assert np.all(np.isfinite(res.state_est))
    off = res.metrics_off.std
    assert np.all(res.metrics_on.std <= off)       # slow-bias path still controls


def test_min_sample_gate_suppresses_underpowered_cpk(rng):
    # SIM-06 / E10: Cpk from a handful of points is suppressed, not reported
    lsl, usl, tgt = -np.full(1, 3.0), np.full(1, 3.0), np.zeros(1)
    m = compute_metrics(rng.standard_normal((4, 1)), tgt, lsl, usl,
                        n_effective=np.array([4]), min_samples=8,
                        finite_sample_ci=True, rng=rng)
    assert m.suppressed[0] and np.isnan(m.cpk[0])
    assert m.to_dict()["note"] == [INSUFFICIENT]
    m2 = compute_metrics(rng.standard_normal((300, 1)), tgt, lsl, usl,
                         n_effective=np.array([300]), min_samples=8,
                         finite_sample_ci=True, rng=rng)
    assert not m2.suppressed[0]
    lo, hi = m2.cpk_ci[0]
    assert lo < m2.cpk[0] < hi
    lo_h, hi_h = m2.harris_ci[0]
    assert lo_h <= hi_h


def test_cpk_interval_shrinks_with_samples(rng):
    from r2r_control.metrics import cpk_confidence_interval
    w10 = np.diff(cpk_confidence_interval(1.0, 10))[0]
    w1000 = np.diff(cpk_confidence_interval(1.0, 1000))[0]
    assert w10 > 5 * w1000
