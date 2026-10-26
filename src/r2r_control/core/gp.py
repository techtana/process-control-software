"""A small from-scratch Gaussian-process surrogate (NFR-01).

Used by the controller optimizer (Component 4) as a response surface for
Bayesian optimization and by the experiment planner (Component 2) as the
sequential-infill surrogate whose acquisition criteria (integrated-MSE /
active-learning-Cohn / expected information gain) choose the next experiment.

Deterministic given its inputs; no external ML dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def rbf_kernel(A: np.ndarray, B: np.ndarray, length_scale: float, signal_var: float) -> np.ndarray:
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    sq = (np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :] - 2 * A @ B.T)
    return signal_var * np.exp(-0.5 * sq / (length_scale ** 2))


@dataclass
class GaussianProcess:
    length_scale: float = 1.0
    signal_var: float = 1.0
    noise_var: float = 1e-4
    categorical_dims: Optional[List[int]] = None   # E11: overlap kernel on these dims

    def __post_init__(self):
        self._X = None
        self._y = None
        self._L = None
        self._alpha = None
        self._y_mean = 0.0

    def kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self.categorical_dims:
            return mixed_kernel(A, B, self.length_scale, self.signal_var, self.categorical_dims)
        return rbf_kernel(A, B, self.length_scale, self.signal_var)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        self._X = X
        self._y_mean = float(y.mean())
        yc = y - self._y_mean
        K = self.kernel(X, X)
        K[np.diag_indices_from(K)] += self.noise_var
        self._L = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._L, yc)
        self._y = y
        return self

    def log_marginal_likelihood(self) -> float:
        """log p(y | X, θ) of the fitted model (for hyperparameter selection)."""
        yc = self._y - self._y_mean
        L = self._L[0]
        return float(-0.5 * yc @ self._alpha - np.sum(np.log(np.abs(np.diag(L))))
                     - 0.5 * len(yc) * np.log(2 * np.pi))

    def fit_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                            length_scales=(0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
                            noise_fracs=(1e-4, 1e-3, 1e-2, 1e-1, 0.3)) -> "GaussianProcess":
        """Choose length scale and noise by marginal likelihood on a grid, then fit.

        ``signal_var`` is set to the response variance; ``noise_var`` is a fraction
        of it. Used when real responses are available (EP-04, KI-6), so the
        surrogate reflects the observed response surface instead of fixed guesses.
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        sv = float(np.var(y)) if np.var(y) > 1e-12 else 1.0
        best, best_ll = None, -np.inf
        for ls in length_scales:
            for nf in noise_fracs:
                gp = GaussianProcess(ls, sv, nf * sv, self.categorical_dims)
                try:
                    ll = gp.fit(X, y).log_marginal_likelihood()
                except np.linalg.LinAlgError:
                    continue
                if ll > best_ll:
                    best, best_ll = (ls, nf * sv), ll
        if best is not None:
            self.length_scale, self.noise_var = best
        self.signal_var = sv
        return self.fit(X, y)

    def predict(self, Xs: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        Ks = self.kernel(Xs, self._X)
        mean = Ks @ self._alpha + self._y_mean
        if not return_std:
            return mean, None
        v = cho_solve(self._L, Ks.T)
        var = self.signal_var - np.sum(Ks * v.T, axis=1)
        var = np.clip(var, 1e-12, None)
        return mean, np.sqrt(var)

    # -- acquisition functions -----------------------------------------
    def expected_improvement(self, Xs: np.ndarray, best: float, xi: float = 0.01,
                             maximize: bool = False) -> np.ndarray:
        """EI for Bayesian optimization (Component 4, OPT-02)."""
        from scipy.stats import norm
        mean, std = self.predict(Xs, return_std=True)
        std = np.maximum(std, 1e-9)
        if maximize:
            imp = mean - best - xi
        else:
            imp = best - mean - xi
        z = imp / std
        return imp * norm.cdf(z) + std * norm.pdf(z)

    def predictive_variance(self, Xs: np.ndarray) -> np.ndarray:
        _, std = self.predict(Xs, return_std=True)
        return std ** 2

    def alc_score(self, candidate: np.ndarray, ref: np.ndarray) -> float:
        """Active-Learning-Cohn: expected reduction in integrated variance (EP-04).

        Approximated as the average reduction in predictive variance over the
        reference set if ``candidate`` were added (one-step look-ahead via the
        kernel correlation).
        """
        candidate = np.atleast_2d(candidate)
        kxc = self.kernel(ref, candidate).reshape(-1)
        kcc = float(self.kernel(candidate, candidate)[0, 0]) + self.noise_var
        kc_self = self.kernel(candidate, self._X).reshape(-1)
        v = cho_solve(self._L, kc_self)
        denom = max(kcc - float(kc_self @ v), 1e-9)
        # reduction in variance at ref points ~ (posterior covariance)^2 / denom
        corr = kxc - self.kernel(ref, self._X) @ v
        return float(np.mean(corr ** 2) / denom)

    def expected_information_gain(self, Xs: np.ndarray) -> np.ndarray:
        """Expected information gain ~ 0.5 log(1 + var/noise) per candidate (EP-04)."""
        var = self.predictive_variance(Xs)
        return 0.5 * np.log1p(var / self.noise_var)


# --------------------------------------------------------------------------- #
# Discrete / integer-aware support (E11)
# --------------------------------------------------------------------------- #
def round_to_feasible(X: np.ndarray, integer_dims=None, categories=None) -> np.ndarray:
    """Snap continuous proposals to feasible discrete values (E11).

    ``integer_dims`` lists column indices that must be integer (e.g. product
    count); they are rounded. ``categories`` maps a column index to its allowed
    levels; the nearest level is chosen. This keeps the planner from emitting
    infeasible (non-integer / off-grid) experiments.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float)).copy()
    if integer_dims:
        for j in integer_dims:
            X[:, j] = np.round(X[:, j])
    if categories:
        for j, levels in categories.items():
            levels = np.asarray(levels, dtype=float)
            idx = np.abs(X[:, j][:, None] - levels[None, :]).argmin(axis=1)
            X[:, j] = levels[idx]
    return X


def mixed_kernel(A: np.ndarray, B: np.ndarray, length_scale: float, signal_var: float,
                 categorical_dims=None) -> np.ndarray:
    """RBF on continuous dims times an overlap kernel on categorical dims (E11).

    Categorical inputs get a category-overlap kernel instead of Euclidean
    distance, so the surrogate is not asked to be smooth across category codes.
    """
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    cat = set(categorical_dims or [])
    cont = [j for j in range(A.shape[1]) if j not in cat]
    Ac, Bc = A[:, cont], B[:, cont]
    sq = (np.sum(Ac ** 2, 1)[:, None] + np.sum(Bc ** 2, 1)[None, :] - 2 * Ac @ Bc.T)
    K = np.exp(-0.5 * sq / (length_scale ** 2))
    for j in cat:
        K = K * (A[:, j][:, None] == B[:, j][None, :]).astype(float)
    return signal_var * K
