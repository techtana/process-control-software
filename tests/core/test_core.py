"""Core shared services, including the NF-03 (E6), A2/E3 and A5/E8 corrections."""

import numpy as np
import pytest

from r2r_control.core import noise_floor, counterfactual, validation, excitation, stats
from r2r_control.core.noise_floor.empirical import empirical_floor
from r2r_control.core.counterfactual.baseline import estimate_baseline
from r2r_control.core.validation.control_off import control_off_gold
from r2r_control.core.data_quality import DataQualityLayer, analyze_alias_structure
from r2r_control.core.gp import round_to_feasible, mixed_kernel
from r2r_control.contracts import Verdict


# ---- noise floor ----------------------------------------------------------
def test_mp_edges():
    lo, hi = noise_floor.marchenko_pastur_edge(0.25)
    assert hi == pytest.approx(2.25)
    assert lo == pytest.approx(0.25)


def test_analytic_on_balanced_empirical_on_unbalanced(rng):
    assert noise_floor.estimate(rng.standard_normal((200, 5)), rng=rng).kind == "analytic_mp"
    X = rng.standard_normal((120, 6))
    X[rng.random(X.shape) < 0.3] = np.nan
    res = noise_floor.estimate(X, n_perm=60, rng=rng)
    assert res.kind.startswith("empirical")
    assert "balanced-panel" in res.reason


def _ar1_panel(rng, T=200, N=6, phi=0.85):
    X = np.zeros((T, N))
    for t in range(1, T):
        X[t] = phi * X[t - 1] + rng.standard_normal(N)
    return X


def _top_eig(X):
    Z = (X - X.mean(0)) / X.std(0)
    return np.linalg.eigvalsh((Z.T @ Z) / (len(X) - 1))[-1]


@pytest.mark.parametrize("surrogate", ["block", "phase"])
def test_structure_preserving_floor_not_fooled_by_autocorrelation(rng, surrogate):
    # E6/NF-03: autocorrelated PURE noise must not be read as signal, and the
    # structure-preserving floor must sit above the forbidden plain-shuffle floor.
    X = _ar1_panel(rng)
    floor = empirical_floor(X, n_perm=120, rng=np.random.default_rng(1),
                            surrogate=surrogate).floor
    r = np.random.default_rng(1)
    plain = np.quantile([_top_eig(np.column_stack([r.permutation(X[:, j])
                                                   for j in range(X.shape[1])]))
                         for _ in range(120)], 0.99)
    assert floor > plain
    assert _top_eig(X) <= floor


def test_plain_shuffle_surrogate_refused():
    with pytest.raises(ValueError):
        empirical_floor(np.zeros((20, 3)), surrogate="shuffle")


def test_empirical_floor_preserves_missingness(rng):
    X = _ar1_panel(rng, T=80, N=4)
    X[rng.random(X.shape) < 0.3] = np.nan
    res = empirical_floor(X, n_perm=20, rng=rng)
    assert np.isfinite(res.floor)


# ---- counterfactual -------------------------------------------------------
def test_counterfactual_exact_on_synthetic_mimo(rng):
    M = rng.standard_normal((4, 3))
    u0 = rng.standard_normal(3)
    disturbance = rng.standard_normal((50, 4))
    u_used = u0 + rng.standard_normal((50, 3))
    cf = counterfactual.reconstruct(disturbance + (u_used - u0) @ M.T, M, u_used, u0)
    assert np.allclose(cf.y_nocontrol, disturbance, atol=1e-10)


def test_counterfactual_accepts_time_varying_u0(rng):
    M = rng.standard_normal((2, 2))
    u0_t = np.cumsum(0.01 * rng.standard_normal((30, 2)), axis=0)
    u = u0_t + rng.standard_normal((30, 2))
    d = rng.standard_normal((30, 2))
    cf = counterfactual.reconstruct(d + (u - u0_t) @ M.T, M, u, u0_t)
    assert np.allclose(cf.y_nocontrol, d)


def test_counterfactual_bands_widen_with_uncertainty(rng):
    M = rng.standard_normal((2, 2))
    u = rng.standard_normal((40, 2))
    y = rng.standard_normal((40, 2))
    narrow = counterfactual.reconstruct_with_bands(y, M, u, np.zeros(2), 0.05, rng=rng)
    wide = counterfactual.reconstruct_with_bands(y, M, u, np.zeros(2), 0.5, rng=rng)
    assert wide.bands["std"].mean() > narrow.bands["std"].mean()


def _episodes_mask(T, starts, length=10):
    mask = np.zeros(T, dtype=bool)
    for s in starts:
        mask[s:s + length] = True
    return mask


def test_baseline_detects_u0_drift_and_builds_time_varying_u0(rng):
    # A2/E3: the open-loop point drifts across control-off episodes
    T = 300
    mask = _episodes_mask(T, [30, 150, 270])
    u = 0.1 * rng.standard_normal((T, 2))
    for k, s in enumerate([30, 150, 270]):
        u[s:s + 10, 0] += 1.0 * k
    base = estimate_baseline(u, mask)
    assert base.drift_detected and base.validity_horizon
    assert "re-baseline" in base.warning
    tv = estimate_baseline(u, mask, allow_time_varying=True)
    assert tv.time_varying and tv.u0.shape == (T, 2)
    assert tv.u0[280, 0] > tv.u0[40, 0]


def test_baseline_stable_when_u0_does_not_drift(rng):
    T = 300
    mask = _episodes_mask(T, [30, 150, 270])
    u = 0.1 * rng.standard_normal((T, 2))
    base = estimate_baseline(u, mask)
    assert not base.drift_detected
    assert base.source == "control-off episode mean"


def test_baseline_without_control_off_warns():
    base = estimate_baseline(np.ones((10, 2)))
    assert "single point of failure" in base.warning


# ---- validation -----------------------------------------------------------
def test_naive_kfold_refused():
    with pytest.raises(validation.NaiveKFoldError):
        validation.naive_kfold_guard(True)
    validation.naive_kfold_guard(False)


def test_forward_chaining_trains_on_past_with_purge():
    for sp in validation.forward_chaining_splits(100, n_splits=4, purge=2):
        assert sp.train_idx.max() < sp.test_idx.min()
        assert sp.test_idx.min() - sp.train_idx.max() >= 2


def test_control_off_gold_excludes_anomalous_episode():
    # A5/E8: a post-PM (off-normal tool state) episode is not used as gold
    T = 200
    mask = _episodes_mask(T, [50, 150])
    tool_state = np.random.default_rng(0).normal(0, 1, T)
    tool_state[150:160] += 10.0
    gold = control_off_gold(mask, tool_state=tool_state)
    assert gold.excluded_episodes == [(150, 159)]
    assert gold.n_gold_episodes == 1
    assert set(gold.gold_idx) == set(range(50, 60))
    assert not set(gold.gold_idx) & set(gold.fit_idx)


def test_block_permutation_preserves_marginal(rng):
    v = rng.standard_normal(100)
    p = validation.contiguous_block_permutation(v, 10, rng)
    assert np.allclose(np.sort(v), np.sort(p))


def test_concept_drift_detected():
    assert validation.concept_drift_test(np.array([10, 20, 30, 40, 50]),
                                         np.array([.1, .2, .3, .4, .5])).degrades_with_horizon


# ---- excitation, stats, data quality, GP ----------------------------------
def test_numerical_rank_full_for_design_and_deficient_for_confounded(rng):
    assert excitation.regressor_spectrum(rng.choice([-1.0, 1.0], (32, 4))).numerical_rank == 4
    a, b = rng.standard_normal((200, 1)), rng.standard_normal((200, 1))
    X = np.column_stack([a, b, a + b, 1e-6 * rng.standard_normal((200, 1))])
    assert excitation.regressor_spectrum(X).numerical_rank < 4


def test_ljung_box_and_adf_kpss(rng):
    assert stats.ljung_box(rng.standard_normal(500), 10).reject_null is False
    rw = np.cumsum(rng.standard_normal(300))
    assert stats.adf_test(rw).reject_null is False
    assert stats.kpss_test(rw).reject_null is True


def test_alias_structure_detects_aliasing(rng):
    base = rng.choice([-1.0, 1.0], (16, 2))
    design = np.column_stack([base, base[:, 0] * base[:, 1]])
    rep = analyze_alias_structure(design, ["A", "B", "C"])       # E13: C = AB
    assert rep.resolution == 3
    assert "C" in rep.aliased_terms and "A*B" in rep.aliased_terms


def test_stale_doe_is_downweighted(dataset):
    dq = DataQualityLayer()
    d = dataset.doe.used_knobs() - dataset.truth.u0
    assert dq.assess_doe(d, dataset.knob_names, age_days=30).decision == Verdict.USE
    assert dq.assess_doe(d, dataset.knob_names, age_days=400).decision == Verdict.DOWN_WEIGHT


def test_round_to_feasible_and_mixed_kernel():
    X = np.array([[1.4, 2.6], [3.2, 0.9]])
    out = round_to_feasible(X, integer_dims=[0], categories={1: [0.0, 1.0, 3.0]})  # E11
    assert np.allclose(out, [[1.0, 3.0], [3.0, 1.0]])
    K = mixed_kernel(np.array([[0.0, 1.0], [0.0, 2.0]]), np.array([[0.0, 1.0]]), 1.0, 1.0,
                     categorical_dims=[1])
    assert K[0, 0] == pytest.approx(1.0) and K[1, 0] == 0.0
