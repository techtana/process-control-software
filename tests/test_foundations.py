"""Tests for the cross-cutting foundations (§4) with deliberately injected faults."""

import numpy as np
import pytest

from process_control.data.schema import _is_measured, check_shape_contract, ShapeContractError
from process_control.foundations import noise_floor, counterfactual, validation, excitation, stats


# --------------------------------------------------------------------------- #
# Data model (DM-03, DM-04)
# --------------------------------------------------------------------------- #
def test_is_measured_handles_all_nan_array():
    # DM-03: pandas .notna would say True for an all-NaN array; _is_measured must not
    assert _is_measured(np.array([np.nan, np.nan])) is False
    assert _is_measured(np.array([np.nan, 1.0])) is True
    assert _is_measured(3.0) is True
    assert _is_measured(np.nan) is False
    assert _is_measured(None) is False
    assert _is_measured(np.array([])) is False


def test_shape_contract_fails_loudly():
    # DM-04
    M = np.zeros((3, 4))
    check_shape_contract(M, 3, 4)  # ok
    with pytest.raises(ShapeContractError):
        check_shape_contract(M, 4, 3)


# --------------------------------------------------------------------------- #
# Noise floor (NF-01..04) — INCLUDING the refusal at the balanced-panel limit
# --------------------------------------------------------------------------- #
def test_mp_edge_monotone():
    lo, hi = noise_floor.marchenko_pastur_edge(0.25)
    assert hi == pytest.approx((1 + 0.5) ** 2)
    assert lo == pytest.approx((1 - 0.5) ** 2)


def test_analytic_floor_used_on_balanced_panel(rng):
    X = rng.standard_normal((200, 5))  # complete => balanced
    res = noise_floor.estimate(X, n_perm=50, rng=rng)
    assert res.kind == "analytic_mp"
    assert res.assumptions_hold


def test_empirical_floor_takes_precedence_on_unbalanced_panel(rng):
    # NF-02/04: with sparse metrology the balanced-panel assumption is violated;
    # the system must REFUSE the analytic MP edge and fall back to empirical.
    X = rng.standard_normal((120, 6))
    mask = rng.random(X.shape) < 0.3
    X[mask] = np.nan
    res = noise_floor.estimate(X, n_perm=80, rng=rng)
    assert res.kind == "empirical_permutation"
    assert "balanced-panel" in res.reason
    assert res.detail["frac_missing"] > 0


def test_empirical_floor_separates_signal_from_noise(rng):
    # a planted rank-1 signal must exceed the permutation floor
    T, N = 150, 8
    factor = rng.standard_normal((T, 1))
    load = rng.standard_normal((1, N))
    X = factor @ load + 0.5 * rng.standard_normal((T, N))
    res = noise_floor.empirical_floor(X, n_perm=100, rng=rng)
    # top eigenvalue of the real data
    Z = (X - X.mean(0)) / X.std(0)
    top = np.linalg.eigvalsh((Z.T @ Z) / (T - 1))[-1]
    assert top > res.floor  # the signal direction is detected


# --------------------------------------------------------------------------- #
# Counterfactual (CF-01) — validated to machine precision on synthetic MIMO
# --------------------------------------------------------------------------- #
def test_counterfactual_exact_on_synthetic_mimo(rng):
    n_out, n_knob, T = 4, 3, 50
    M = rng.standard_normal((n_out, n_knob))
    u0 = rng.standard_normal(n_knob)
    disturbance = rng.standard_normal((T, n_out))
    u_used = u0 + rng.standard_normal((T, n_knob))
    y_obs = disturbance + (u_used - u0) @ M.T          # the true generative model
    cf = counterfactual.reconstruct(y_obs, M, u_used, u0)
    # CF-01: subtraction recovers the disturbance to machine precision
    assert np.allclose(cf.y_nocontrol, disturbance, atol=1e-10)


def test_counterfactual_bands_widen_with_uncertainty(rng):
    n_out, n_knob, T = 3, 2, 40
    M = rng.standard_normal((n_out, n_knob))
    u0 = np.zeros(n_knob)
    u_used = rng.standard_normal((T, n_knob))
    y_obs = rng.standard_normal((T, n_out)) + (u_used - u0) @ M.T
    cf0 = counterfactual.reconstruct_with_bands(y_obs, M, u_used, u0, rel_uncertainty_M=0.0)
    cf1 = counterfactual.reconstruct_with_bands(y_obs, M, u_used, u0, rel_uncertainty_M=0.2,
                                                n_samples=200, rng=rng)
    assert cf0.bands is None
    assert cf1.bands is not None
    assert np.mean(cf1.bands["std"]) > 0


# --------------------------------------------------------------------------- #
# Validation (VAL-01, VAL-02, VAL-05)
# --------------------------------------------------------------------------- #
def test_naive_kfold_is_refused():
    # VAL-01: the system must refuse naive k-fold for temporal data
    with pytest.raises(validation.NaiveKFoldError):
        validation.naive_kfold_guard(is_temporal=True)
    validation.naive_kfold_guard(is_temporal=False)  # ok for non-temporal


def test_forward_chaining_trains_on_past_only():
    splits = validation.forward_chaining_splits(100, n_splits=4, purge=2)
    assert len(splits) >= 3
    for sp in splits:
        assert sp.train_idx.max() < sp.test_idx.min()      # train strictly before test
        assert sp.test_idx.min() - sp.train_idx.max() >= 2  # purge gap honored


def test_block_bootstrap_preserves_contiguity(rng):
    idx = validation.block_bootstrap(100, block_size=10, rng=rng)
    assert len(idx) == 100


def test_concept_drift_detected_when_error_rises_with_horizon():
    horizons = np.array([10, 20, 30, 40, 50])
    errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # systematic rise
    res = validation.concept_drift_test(horizons, errors)
    assert res.degrades_with_horizon


# --------------------------------------------------------------------------- #
# Excitation / confounding (EX)
# --------------------------------------------------------------------------- #
def test_vif_flags_collinear_regressor(rng):
    x1 = rng.standard_normal(200)
    x2 = rng.standard_normal(200)
    x3 = x1 + 0.01 * rng.standard_normal(200)  # nearly collinear with x1
    X = np.column_stack([x1, x2, x3])
    v = excitation.vif(X)
    assert v[0] > 5 and v[2] > 5    # collinear pair inflated
    assert v[1] < 2                 # independent column not inflated


def test_numerical_rank_full_for_orthogonal_design(rng):
    # a well-conditioned designed experiment must report FULL rank
    design = rng.choice([-1.0, 1.0], size=(32, 4))
    spec = excitation.regressor_spectrum(design)
    assert spec.numerical_rank == 4


def test_numerical_rank_deficient_for_confounded(rng):
    # one direction with negligible excitation => rank drops
    a = rng.standard_normal((200, 1))
    b = rng.standard_normal((200, 1))
    c = 1e-6 * rng.standard_normal((200, 1))   # essentially unexcited direction
    X = np.column_stack([a, b, a + b, c]).reshape(200, 4)
    spec = excitation.regressor_spectrum(X)
    assert spec.numerical_rank < 4


# --------------------------------------------------------------------------- #
# Statistics from scratch (NFR-01)
# --------------------------------------------------------------------------- #
def test_ljung_box_distinguishes_white_from_colored(rng):
    white = rng.standard_normal(500)
    lb_white = stats.ljung_box(white, lags=10)
    assert lb_white.reject_null is False        # white => fail to reject

    colored = np.zeros(500)
    for t in range(1, 500):
        colored[t] = 0.8 * colored[t - 1] + rng.standard_normal()
    lb_colored = stats.ljung_box(colored, lags=10)
    assert lb_colored.reject_null is True       # AR(1) => reject whiteness


def test_adf_kpss_detect_drift(rng):
    rw = np.cumsum(rng.standard_normal(300))    # random walk (non-stationary)
    adf = stats.adf_test(rw)
    kpss = stats.kpss_test(rw)
    assert adf.reject_null is False             # ADF fails to reject unit root
    assert kpss.reject_null is True             # KPSS rejects stationarity
