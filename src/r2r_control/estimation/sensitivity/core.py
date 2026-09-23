"""Shared cross-sectional sensitivity core (§1.3, REG-01).

This is the part Components 1 and 6 genuinely share: the estimate of how outputs
respond to inputs (the gain / cross-sectional sensitivity), with the noise-floor
identifiability report and parameter-uncertainty covariance. It is NOT the whole
estimator — Component 1 wraps it with a dynamics/state-space layer and Component
6 with a high-dimensional regularized-selection layer. Keeping this core single,
and routing the noise floor / counterfactual / data quality through the shared
services, is what makes "the noise floor" and "the counterfactual" mean one
thing for both (§IF-06) without forcing them into one estimator class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ...contracts.provenance import ProvenanceLog
from ...core import noise_floor as nf
from ...core.excitation.spectrum import regressor_spectrum
from ...core.excitation.directions import unidentifiable_directions
from ...core.excitation.vif import vif


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float):
    """Multi-output ridge with parameter covariance (FB-03, REG-03)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, p = X.shape
    XtX = X.T @ X
    A_inv = np.linalg.inv(XtX + lam * np.eye(p))
    B = A_inv @ X.T @ Y
    resid = Y - X @ B
    dof = max(n - np.trace(X @ A_inv @ X.T), 1.0)
    sigma2 = np.sum(resid ** 2, axis=0) / dof
    return B, A_inv @ XtX @ A_inv, sigma2


def svd_truncated_fit(X: np.ndarray, Y: np.ndarray, rank: int):
    """SVD-truncated least squares (FB-02/03; truncation == eigenvalue clipping)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    rank = max(1, min(rank, len(s)))
    s_inv = np.zeros_like(s)
    s_inv[:rank] = 1.0 / s[:rank]
    B = (Vt.T * s_inv) @ (U.T @ Y)
    resid = Y - X @ B
    sigma2 = np.sum(resid ** 2, axis=0) / max(X.shape[0] - rank, 1.0)
    cov = (Vt.T[:, :rank] * (s_inv[:rank] ** 2)) @ Vt[:rank, :]
    return B, cov, sigma2, s


@dataclass
class IdentifiableDirectionReport:
    effective_rank: int
    n_directions: int
    noise_floor: float
    unidentifiable: list = field(default_factory=list)
    vif: Optional[np.ndarray] = None
    names: List[str] = field(default_factory=list)
    signal_rank_mp: Optional[int] = None
    condition_number: float = float("nan")

    def low_confidence_features(self) -> List[str]:
        return sorted({d for u in self.unidentifiable for d in u.dominant})

    def is_deficient(self) -> bool:
        return self.effective_rank < self.n_directions

    def to_dict(self):
        return {"effective_rank": self.effective_rank, "n_directions": self.n_directions,
                "noise_floor": float(self.noise_floor), "signal_rank_mp": self.signal_rank_mp,
                "condition_number": float(self.condition_number),
                "unidentifiable": [d.to_dict() for d in self.unidentifiable],
                "vif": None if self.vif is None else self.vif.tolist(),
                "low_confidence_features": self.low_confidence_features(),
                "names": self.names}


@dataclass
class FittedSensitivity:
    coef: np.ndarray                   # (p, n_out)
    param_cov: np.ndarray
    sigma2: np.ndarray
    identifiable: IdentifiableDirectionReport
    intercept: np.ndarray
    feature_names: List[str]
    output_names: List[str]
    estimator: str
    provenance: ProvenanceLog
    extras: dict = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef + self.intercept

    def relative_uncertainty(self) -> float:
        scale = np.linalg.norm(self.coef) + 1e-12
        return float(np.sqrt(np.trace(self.param_cov) * np.mean(self.sigma2)) / scale)


class SensitivityCore:
    """Shared cross-sectional gain/sensitivity estimator (no dynamics, no selection)."""

    def __init__(self, estimator: str = "ridge", ridge_lambda: float = 1.0,
                 rank_rtol: float = 1e-2, order_by_noise_floor: bool = True,
                 noise_floor_perm: int = 200, noise_floor_quantile: float = 0.99,
                 seed: int = 0):
        if estimator not in ("ridge", "svd"):
            raise ValueError(f"SensitivityCore supports 'ridge'|'svd', got {estimator!r} "
                             f"(PLS/elastic-net live in the Component-6 regularized layer)")
        self.estimator = estimator
        self.ridge_lambda = ridge_lambda
        self.rank_rtol = rank_rtol
        self.order_by_noise_floor = order_by_noise_floor
        self.noise_floor_perm = noise_floor_perm
        self.noise_floor_quantile = noise_floor_quantile
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def identifiability(self, X, feature_names=None, provenance=None):
        X = np.asarray(X, dtype=float)
        names = feature_names or [f"x{j}" for j in range(X.shape[1])]
        floor_res = nf.estimate(X, n_perm=self.noise_floor_perm,
                                quantile=self.noise_floor_quantile, rng=self.rng)
        spec = regressor_spectrum(X, names=names, noise_floor=floor_res.floor,
                                  rtol=self.rank_rtol)
        if provenance is not None:
            provenance.record_noise_floor(floor_res.kind, floor_res.floor, floor_res.reason,
                                          **floor_res.detail)
        return IdentifiableDirectionReport(
            effective_rank=spec.numerical_rank, n_directions=X.shape[1],
            noise_floor=floor_res.floor,
            unidentifiable=unidentifiable_directions(X, rtol=self.rank_rtol, names=names),
            vif=vif(X), names=names, signal_rank_mp=spec.signal_rank_mp,
            condition_number=spec.condition_number)

    def fit(self, X, Y, feature_names=None, output_names=None, provenance=None,
            weights=None) -> FittedSensitivity:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        names = feature_names or [f"x{j}" for j in range(X.shape[1])]
        onames = output_names or [f"y{j}" for j in range(Y.shape[1])]
        prov = provenance or ProvenanceLog(seed=self.seed)

        if weights is not None:
            w = np.sqrt(np.asarray(weights, dtype=float)).reshape(-1, 1)
            X, Y = X * w, Y * w
        x_mu, y_mu = X.mean(0), Y.mean(0)
        Xc, Yc = X - x_mu, Y - y_mu

        report = self.identifiability(Xc, names, prov)
        rank = max(1, report.effective_rank) if self.order_by_noise_floor else min(Xc.shape)
        prov.note(f"retained model order = {rank} (effective rank above noise floor)")

        extras = {}
        if self.estimator == "ridge":
            B, cov, sigma2 = ridge_fit(Xc, Yc, self.ridge_lambda)
        else:
            B, cov, sigma2, s = svd_truncated_fit(Xc, Yc, rank)
            extras["singular_values"] = s
        prov.note(f"sensitivity core fit ({self.estimator}); order={rank}; "
                  f"per-output residual var={np.round(sigma2, 5).tolist()}")
        return FittedSensitivity(coef=B, param_cov=cov, sigma2=sigma2, identifiable=report,
                                 intercept=y_mu - x_mu @ B, feature_names=names,
                                 output_names=onames, estimator=self.estimator,
                                 provenance=prov, extras=extras)
