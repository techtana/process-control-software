"""The shared estimation engine (§REG-01, §IF-06).

The FB-model identifier (Component 1) and the all-data regression machine
(Component 6) are **two configurations of one estimation engine**, not two
unrelated tools.  They share this engine, the data model, the data-quality
layer, the excitation/confounding analysis, the noise-floor estimator, the
counterfactual service, and the validation framework.  Keeping them unified is a
hard requirement: the confounding and noise-floor logic is identical for both.

The engine solves a regularized multi-output linear estimation whose retained
dimensionality is set by where the singular-value spectrum drops into the noise
floor.  Three estimators are provided, the choice encoding a structural belief:

* ``ridge``        — dense-but-shrunk (Tikhonov on the normal equations);
* ``pls``          — supervised latent components maximizing covariance with the
                     target (the directly target-relevant analog of the
                     eigenvalue decomposition), for ``N_features >> T``;
* ``elastic_net``  — sparse-with-correlated-groups.

Every fit returns a :class:`FittedModel` carrying the coefficient map, a
parameter-uncertainty covariance, the identifiable-direction report, and the
provenance log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..provenance import ProvenanceLog
from . import noise_floor as nf
from .excitation import regressor_spectrum, unidentifiable_directions, vif


# --------------------------------------------------------------------------- #
# Low-level estimators (numpy from scratch)
# --------------------------------------------------------------------------- #
def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float):
    """Multi-output ridge with parameter covariance (FB-03, REG-03).

    Returns ``(B, cov_per_output, sigma2)`` where ``B`` is ``(p, n_out)`` and
    ``cov_per_output`` is the ``(p, p)`` parameter covariance scaled per output.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, p = X.shape
    XtX = X.T @ X
    A = XtX + lam * np.eye(p)
    A_inv = np.linalg.inv(A)
    B = A_inv @ X.T @ Y
    resid = Y - X @ B
    dof = max(n - np.trace(X @ A_inv @ X.T), 1.0)
    sigma2 = np.sum(resid ** 2, axis=0) / dof          # per output
    # Cov(beta) = sigma2 * A_inv X^T X A_inv  (ridge sandwich)
    sandwich = A_inv @ XtX @ A_inv
    return B, sandwich, sigma2


def svd_truncated_fit(X: np.ndarray, Y: np.ndarray, rank: int):
    """SVD-truncated least squares (FB-02/03; truncation == eigenvalue clipping).

    Keeps only the top ``rank`` singular directions — the directions above the
    noise gap — discarding noisy small-singular-value directions that would
    inflate parameter variance and corrupt the MPC model.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    rank = max(1, min(rank, len(s)))
    s_inv = np.zeros_like(s)
    s_inv[:rank] = 1.0 / s[:rank]
    B = (Vt.T * s_inv) @ (U.T @ Y)
    resid = Y - X @ B
    dof = max(X.shape[0] - rank, 1.0)
    sigma2 = np.sum(resid ** 2, axis=0) / dof
    # parameter covariance restricted to retained subspace
    cov = (Vt.T[:, :rank] * (s_inv[:rank] ** 2)) @ Vt[:rank, :]
    return B, cov, sigma2, s


def pls_fit(X: np.ndarray, Y: np.ndarray, n_components: int):
    """Partial least squares via NIPALS (REG-03).

    Supervised latent components maximizing covariance with the target.  Returns
    coefficients in the original feature space plus the latent scores/loadings.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(len(X), -1)
    n, p = X.shape
    m = Y.shape[1]
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    Xr, Yr = Xc.copy(), Yc.copy()
    n_components = max(1, min(n_components, min(n - 1, p)))
    W = np.zeros((p, n_components))
    P = np.zeros((p, n_components))
    Q = np.zeros((m, n_components))
    T = np.zeros((n, n_components))
    for a in range(n_components):
        # dominant singular vector of Xr^T Yr
        C = Xr.T @ Yr
        u, s, vt = np.linalg.svd(C, full_matrices=False)
        w = u[:, 0]
        w = w / (np.linalg.norm(w) + 1e-15)
        t = Xr @ w
        tt = float(t @ t) + 1e-15
        pld = Xr.T @ t / tt
        q = Yr.T @ t / tt
        Xr = Xr - np.outer(t, pld)
        Yr = Yr - np.outer(t, q)
        W[:, a], P[:, a], Q[:, a], T[:, a] = w, pld, q, t
    # regression coefficients in original space
    Wstar = W @ np.linalg.pinv(P.T @ W)
    B = Wstar @ Q.T                      # (p, m)
    return B, T, Wstar


def elastic_net_fit(X: np.ndarray, y: np.ndarray, lam: float, l1_ratio: float = 0.5,
                    max_iter: int = 1000, tol: float = 1e-6):
    """Single-output elastic net via coordinate descent (REG-03, REG-06).

    Sparse-with-correlated-groups.  Standardizes internally; returns coefficients
    in standardized space (the caller uses selection support, not raw scale).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    mu = X.mean(0)
    sd = X.std(0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd_safe
    yc = y - y.mean()
    beta = np.zeros(p)
    l1 = lam * l1_ratio
    l2 = lam * (1 - l1_ratio)
    col_sq = np.sum(Z ** 2, axis=0) / n
    r = yc.copy()
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            if col_sq[j] == 0:
                continue
            r = r + Z[:, j] * beta[j]
            rho = float(Z[:, j] @ r) / n
            beta[j] = _soft_threshold(rho, l1) / (col_sq[j] + l2)
            r = r - Z[:, j] * beta[j]
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta


def _soft_threshold(x: float, t: float) -> float:
    if x > t:
        return x - t
    if x < -t:
        return x + t
    return 0.0


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
@dataclass
class EngineConfig:
    """Configuration selecting one of the two engine personalities (REG-01)."""

    mode: str = "fb"                  # 'fb' (structured MIMO) | 'regression' (broad)
    estimator: str = "ridge"          # 'ridge' | 'pls' | 'elastic_net' | 'svd'
    ridge_lambda: float = 1.0
    n_components: Optional[int] = None
    l1_ratio: float = 0.5
    elastic_lambda: float = 0.1
    noise_floor_perm: int = 200
    noise_floor_quantile: float = 0.99
    order_by_noise_floor: bool = True
    rank_rtol: float = 1e-2           # singular-value gap for numerical rank (FB-02)
    seed: int = 0


@dataclass
class IdentifiableDirectionReport:
    """Effective rank and which directions are unresolved (FB-04, REG-07)."""

    effective_rank: int
    n_directions: int
    noise_floor: float
    unidentifiable: list = field(default_factory=list)
    vif: Optional[np.ndarray] = None
    names: List[str] = field(default_factory=list)
    signal_rank_mp: Optional[int] = None
    condition_number: float = float("nan")

    def low_confidence_features(self) -> List[str]:
        out = set()
        for d in self.unidentifiable:
            out.update(d.dominant)
        return sorted(out)

    def is_deficient(self) -> bool:
        return self.effective_rank < self.n_directions

    def to_dict(self):
        return {
            "effective_rank": self.effective_rank,
            "n_directions": self.n_directions,
            "noise_floor": float(self.noise_floor),
            "signal_rank_mp": self.signal_rank_mp,
            "condition_number": float(self.condition_number),
            "unidentifiable": [d.to_dict() for d in self.unidentifiable],
            "vif": None if self.vif is None else self.vif.tolist(),
            "low_confidence_features": self.low_confidence_features(),
            "names": self.names,
        }


@dataclass
class FittedModel:
    coef: np.ndarray                   # (p, n_out) coefficient map
    param_cov: np.ndarray              # (p, p) parameter-uncertainty covariance
    sigma2: np.ndarray                 # per-output residual variance
    identifiable: IdentifiableDirectionReport
    provenance: ProvenanceLog
    feature_names: List[str]
    output_names: List[str]
    estimator: str
    intercept: np.ndarray
    extras: dict = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef + self.intercept

    def relative_uncertainty(self) -> float:
        """A scalar relative-uncertainty estimate on the coefficient map (CF-03/SIM-02)."""
        scale = np.linalg.norm(self.coef) + 1e-12
        return float(np.sqrt(np.trace(self.param_cov) * np.mean(self.sigma2)) / scale)

    def to_dict(self):
        return {
            "estimator": self.estimator,
            "coef_shape": list(self.coef.shape),
            "relative_uncertainty": self.relative_uncertainty(),
            "identifiable": self.identifiable.to_dict(),
            "provenance": self.provenance.to_dict(),
        }


class EstimationEngine:
    """One estimation engine, two configurations (REG-01)."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.rng = np.random.default_rng(self.config.seed)

    # -- identifiability report ------------------------------------------
    def identifiability(self, X: np.ndarray, feature_names: Optional[List[str]] = None,
                        provenance: Optional[ProvenanceLog] = None
                        ) -> IdentifiableDirectionReport:
        """Compute the effective rank and unidentifiable directions (EX-02/03, NF)."""
        X = np.asarray(X, dtype=float)
        names = feature_names or [f"x{j}" for j in range(X.shape[1])]
        # noise floor on the (standardized) regressor block — the MP/empirical
        # spike floor used for the confounding narrative (signal_rank_mp).
        floor_res = nf.estimate(
            X, n_perm=self.config.noise_floor_perm,
            quantile=self.config.noise_floor_quantile, rng=self.rng,
        )
        spec = regressor_spectrum(X, names=names, noise_floor=floor_res.floor,
                                  rtol=self.config.rank_rtol)
        unident = unidentifiable_directions(X, rtol=self.config.rank_rtol, names=names)
        if provenance is not None:
            provenance.record_noise_floor(
                floor_res.kind, floor_res.floor, floor_res.reason, **floor_res.detail)
        return IdentifiableDirectionReport(
            effective_rank=spec.numerical_rank,
            n_directions=X.shape[1],
            noise_floor=floor_res.floor,
            unidentifiable=unident,
            vif=vif(X),
            names=names,
            signal_rank_mp=spec.signal_rank_mp,
            condition_number=spec.condition_number,
        )

    def _select_rank(self, X: np.ndarray, report: IdentifiableDirectionReport) -> int:
        """Model order = numerical rank (singular values above the noise gap, FB-02)."""
        if not self.config.order_by_noise_floor:
            return min(X.shape)
        return max(1, report.effective_rank)

    # -- fit --------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        provenance: Optional[ProvenanceLog] = None,
        weights: Optional[np.ndarray] = None,
    ) -> FittedModel:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        n, p = X.shape
        names = feature_names or [f"x{j}" for j in range(p)]
        onames = output_names or [f"y{j}" for j in range(Y.shape[1])]
        prov = provenance or ProvenanceLog(seed=self.config.seed)
        prov.config.update({"engine_mode": self.config.mode,
                            "estimator": self.config.estimator})

        # apply row weights (provenance-weighted combination, FB-01/DQ-07)
        if weights is not None:
            w = np.sqrt(np.asarray(weights, dtype=float)).reshape(-1, 1)
            Xw, Yw = X * w, Y * w
        else:
            Xw, Yw = X, Y

        # center
        x_mu = Xw.mean(0)
        y_mu = Yw.mean(0)
        Xc = Xw - x_mu
        Yc = Yw - y_mu

        report = self.identifiability(Xc, names, prov)
        rank = self._select_rank(Xc, report)
        prov.note(f"retained model order = {rank} (effective rank above noise floor)")

        est = self.config.estimator
        extras: dict = {}
        if est == "ridge":
            B, cov, sigma2 = ridge_fit(Xc, Yc, self.config.ridge_lambda)
        elif est == "svd":
            B, cov, sigma2, s = svd_truncated_fit(Xc, Yc, rank)
            extras["singular_values"] = s
        elif est == "pls":
            ncomp = self.config.n_components or rank
            B, Tsc, Wstar = pls_fit(Xc, Yc, ncomp)
            resid = Yc - Xc @ B
            sigma2 = np.sum(resid ** 2, axis=0) / max(n - ncomp, 1)
            # crude coefficient covariance from latent pseudo-inverse
            cov = Wstar @ Wstar.T
            extras["n_components"] = ncomp
        elif est == "elastic_net":
            B = np.zeros((p, Yc.shape[1]))
            for j in range(Yc.shape[1]):
                B[:, j] = elastic_net_fit(
                    Xc, Yc[:, j], self.config.elastic_lambda, self.config.l1_ratio)
            resid = Yc - Xc @ B
            sigma2 = np.sum(resid ** 2, axis=0) / max(n - 1, 1)
            cov = np.eye(p)  # elastic-net coef uncertainty handled via stability selection
        else:
            raise ValueError(f"unknown estimator {est!r}")

        intercept = y_mu - x_mu @ B
        prov.note(f"fitted {est}; order={rank}; "
                  f"per-output residual var = {np.round(sigma2, 5).tolist()}")

        return FittedModel(
            coef=B,
            param_cov=cov,
            sigma2=sigma2,
            identifiable=report,
            provenance=prov,
            feature_names=names,
            output_names=onames,
            estimator=est,
            intercept=intercept,
            extras=extras,
        )
