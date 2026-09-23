"""Estimation layer: shared SensitivityCore, the de-unification (§1.3), and E12."""

import ast
import pathlib

import numpy as np
import pytest

from r2r_control import synthetic
from r2r_control.estimation.sensitivity import SensitivityCore
from r2r_control.estimation.dynamics import subspace_order, identify_first_order_dynamics
from r2r_control.estimation.regularized import stability_selection, pls_fit
from r2r_control.components.fb_identification import FBIdentifier
from r2r_control.components.regression_machine import RegressionMachine

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "r2r_control"


def test_sensitivity_core_recovers_gain():
    ds = synthetic.make_dataset(seed=2)
    Xd = ds.doe.used_knobs() - ds.truth.u0
    fit = SensitivityCore(estimator="svd").fit(Xd, ds.doe.y_matrix())
    assert np.linalg.norm(fit.coef.T - ds.truth.M) / np.linalg.norm(ds.truth.M) < 0.2
    assert subspace_order(Xd) == ds.truth.n_knob


def test_sensitivity_core_has_no_selection_mode():
    # the old unified engine had a mode flag; the shared core only does gain
    with pytest.raises(ValueError):
        SensitivityCore(estimator="elastic_net")


def test_dynamics_identifies_stable_ar(rng):
    d = np.zeros((300, 2))
    for t in range(1, 300):
        d[t] = 0.8 * d[t - 1] + rng.standard_normal(2)
    A = identify_first_order_dynamics(d)
    assert np.allclose(np.diag(A), 0.8, atol=0.1)


def _imports(path):
    tree = ast.parse(path.read_text())
    return {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}


def test_components_share_core_but_have_their_own_layers():
    # §1.3 corrected REG-01: C1 and C6 wrap the SAME SensitivityCore but each has
    # its own layer — C1 dynamics, C6 regularized selection — and they are
    # different classes, not one estimator with a mode flag.
    assert FBIdentifier is not RegressionMachine
    c1 = _imports(SRC / "components/fb_identification/identify.py")
    c6 = _imports(SRC / "components/regression_machine/__init__.py")
    assert "estimation.sensitivity.core" in c1 and "estimation.sensitivity.core" in c6
    assert any("estimation.dynamics" in m for m in c1)
    assert not any("estimation.regularized" in m for m in c1)
    assert any("estimation.regularized" in m for m in c6)
    assert not any("estimation.dynamics" in m for m in c6)


def test_group_aware_stability_selection(rng):
    # E12: a correlated sensor group is reported as a group with its own frequency
    ds = synthetic.make_dataset(seed=3, coupled_sensor_knob=(0, 1))
    X, names = ds.inline.regressors(), ds.inline.regressor_names
    y = ds.inline.y_matrix()[:, 0]
    m = np.isfinite(y)
    ss = stability_selection(X[m], y[m], names, n_resamples=40, rng=rng)
    group = [g for g, members in ss.groups.items() if {"ff0", "u1_used"} <= set(members)]
    assert group
    g = group[0]
    per_member = max(ss.selection_frequency[names.index(n)] for n in ss.groups[g])
    assert ss.group_frequency[g] >= per_member


def test_pls_reduces_to_exact_fit_on_noise_free_data(rng):
    X = rng.standard_normal((50, 3))
    B_true = np.array([[1.0], [-2.0], [0.5]])
    B, _, _ = pls_fit(X, X @ B_true, 3)
    assert np.allclose(B, B_true, atol=1e-8)
