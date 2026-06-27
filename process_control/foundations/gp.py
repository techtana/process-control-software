"""A small from-scratch Gaussian-process surrogate (NFR-01).

Used by the controller optimizer (Component 4) as a response surface for
Bayesian optimization and by the experiment planner (Component 2) as the
sequential-infill surrogate whose acquisition criteria (integrated-MSE /
active-learning-Cohn / expected information gain) choose the next experiment.

Deterministic given its inputs; no external ML dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

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

    def __post_init__(self):
        self._X = None
        self._y = None
        self._L = None
        self._alpha = None
        self._y_mean = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        self._X = X
        self._y_mean = float(y.mean())
        yc = y - self._y_mean
        K = rbf_kernel(X, X, self.length_scale, self.signal_var)
        K[np.diag_indices_from(K)] += self.noise_var
        self._L = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._L, yc)
        self._y = y
        return self

    def predict(self, Xs: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        Ks = rbf_kernel(Xs, self._X, self.length_scale, self.signal_var)
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
        kxc = rbf_kernel(ref, candidate, self.length_scale, self.signal_var).reshape(-1)
        kcc = float(rbf_kernel(candidate, candidate, self.length_scale, self.signal_var)
                    [0, 0]) + self.noise_var
        v = cho_solve(self._L, rbf_kernel(candidate, self._X, self.length_scale,
                                          self.signal_var).T).reshape(-1)
        kc_self = rbf_kernel(candidate, self._X, self.length_scale, self.signal_var).reshape(-1)
        denom = kcc - float(kc_self @ v)
        denom = max(denom, 1e-9)
        # reduction in variance at ref points ~ correlation^2 / denom
        kxc_pred = rbf_kernel(ref, self._X, self.length_scale, self.signal_var) @ \
            cho_solve(self._L, rbf_kernel(candidate, self._X, self.length_scale,
                                          self.signal_var).T).reshape(-1)
        corr = kxc - kxc_pred
        return float(np.mean(corr ** 2) / denom)

    def expected_information_gain(self, Xs: np.ndarray) -> np.ndarray:
        """Expected information gain ~ 0.5 log(1 + var/noise) per candidate (EP-04)."""
        var = self.predictive_variance(Xs)
        return 0.5 * np.log1p(var / self.noise_var)
