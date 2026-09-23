"""Injected-fault tests (NFR-04) for the assumption / edge-case guards."""

import numpy as np

from r2r_control import synthetic
from r2r_control.components.diagnostics import DiagnosticsEngine
from r2r_control.components.diagnostics.gain_mismatch import gain_drift_monitor
from r2r_control.components.regression_machine import RegressionMachine
from r2r_control.components.simulator import PlantConfig
from tests.conftest import make_sim


def test_gain_underestimate_diagnosed_as_overcorrection(dataset):
    # DIAG-02: model gain 4x too small => over-correction, Harris >> 1
    t = dataset.truth
    res = make_sim(t, model_gain_scale=0.25, mpc_R=1e-4, move_limit=5.0).simulate(n_steps=300)
    d = DiagnosticsEngine().diagnose(res.y_on, res.u, t.M * 0.25, innovation=res.innovation)
    assert d.excitation["sufficient"]                # big moves => gate passes
    assert d.gain_mismatch["oscillating"]
    assert d.gain_mismatch["gain_underestimated"]
    assert not d.achievability["at_achievability_floor"]


def _drift_record(gain_drift):
    ds = synthetic.make_dataset(seed=7, gain_drift=gain_drift, n_events=600,
                                override_fraction=0.3, control_off_episodes=0,
                                metrology_sparsity=0.0)
    Y = ds.inline.y_matrix()
    return Y, ds.inline.used_knobs(), np.all(np.isfinite(Y), axis=1)


def test_gain_drift_monitor_detects_drift():
    # A1/E1: a drifting true gain makes the realized gain trend over time
    Y, U, meas = _drift_record(1.2)
    assert gain_drift_monitor(Y, U, window=80, measured_mask=meas)["drift_detected"]


def test_gain_drift_monitor_quiet_on_fixed_gain():
    Y, U, meas = _drift_record(0.0)
    assert not gain_drift_monitor(Y, U, window=80, measured_mask=meas)["drift_detected"]


def test_phantom_ff_quarantined():
    # A3/E2: a sensor that tracks a knob whose M column is wrong must be quarantined,
    # not reported as a feedforward candidate.
    ds = synthetic.make_dataset(seed=8, coupled_sensor_knob=(0, 1))
    M_err = ds.truth.M.copy()
    M_err[:, 1] *= 1.6
    res = RegressionMachine(seed=0, n_stability_resamples=40).fit(
        ds.inline, "y0", M=M_err, u0=ds.truth.u0,
        adjustable_features=ds.inline.knob_cols_used, rel_uncertainty_M=0.3)
    assert "ff0" in res.quarantined_sensors
    assert "ff0" not in res.feedforward_candidates


def test_phantom_guard_keeps_real_feedforward_disturbances():
    # a real measurable disturbance produces an effect far outside the M-error
    # envelope, so the guard must not quarantine it
    ds = synthetic.make_dataset(seed=8, ff_leak=0.7)
    res = RegressionMachine(seed=0, n_stability_resamples=40).fit(
        ds.inline, "y0", M=ds.truth.M, u0=ds.truth.u0, rel_uncertainty_M=0.1)
    assert res.quarantined_sensors == []
    assert res.feedforward_candidates


def test_ff_leakage_detected():
    # DIAG-01: when the controller's FF model is missing, the measurable FF
    # disturbance leaks into the estimated state and is flagged as an opportunity.
    ds = synthetic.make_dataset(seed=3, n_ff=4)
    t = ds.truth
    sim = make_sim(t, plant=PlantConfig(meas_noise_std=0.1, ff_disturbance_std=1.0),
                   mpc_R=0.05)
    FF = sim.FF_gain

    def leakage(ff_model):
        _, _, est, _, _, ff = sim._run_once(300, np.random.default_rng(0), t.M, t.M, FF,
                                            ff_model)
        return DiagnosticsEngine().ff_leakage(est, ff, ds.ff_names)

    blind = leakage(np.zeros_like(FF))
    sighted = leakage(FF)
    assert blind["feedforward_opportunities"]
    assert len(sighted["feedforward_opportunities"]) < len(blind["feedforward_opportunities"])
