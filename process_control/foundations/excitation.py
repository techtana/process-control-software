"""Excitation and confounding analysis (§4.3, EX-01..04).

Passive closed-loop data reflects the controller's corrections, not the
open-loop process, so the knob and disturbance regressors become collinear and
the data covariance is rank-deficient in exactly the directions one needs to
identify.  This module *quantifies* that confounding (VIF), *locates* the
signal/noise split in the regressor spectrum (effective rank against a noise
floor), *maps* the deficiency to specific unidentifiable knob/FF directions,
and *mines* whatever accidental excitation exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


def _standardize(X: np.ndarray):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd_safe
    return Z, mu, sd


def vif(X: np.ndarray) -> np.ndarray:
    """Variance-inflation factor per regressor (EX-01).

    ``VIF_j = 1 / (1 - R_j^2)`` where ``R_j^2`` is from regressing column j on
    the others.  Large VIF => that knob/FF direction is largely explained by the
    others, i.e. its apparent relationship to the output is confounded by
    feedback.  Returned in the natural units for stakeholder framing: a VIF of
    10 means ~90% of that knob's movement is collinear with the rest.
    """
    Z, _, sd = _standardize(np.asarray(X, dtype=float))
    n, p = Z.shape
    out = np.full(p, np.nan)
    for j in range(p):
        if sd[j] <= 1e-12:
            out[j] = np.inf  # a constant column is perfectly confounded
            continue
        others = np.delete(Z, j, axis=1)
        if others.shape[1] == 0:
            out[j] = 1.0
            continue
        beta, *_ = np.linalg.lstsq(others, Z[:, j], rcond=None)
        resid = Z[:, j] - others @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(Z[:, j] @ Z[:, j])
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2 = min(max(r2, 0.0), 1.0 - 1e-12)
        out[j] = 1.0 / (1.0 - r2)
    return out


@dataclass
class SpectrumReport:
    eigenvalues: np.ndarray            # descending eigenvalues of the corr matrix
    singular_values: np.ndarray        # of the standardized regressor block
    noise_floor: Optional[float]       # eigenvalue floor (NF); None until set
    signal_rank_mp: Optional[int]      # MP spike count: eigenvalues ABOVE the floor
    numerical_rank: int                # singular values above the noise gap (FB-02)
    effective_rank_entropy: float      # participation-ratio / entropy rank
    condition_number: float
    names: List[str] = field(default_factory=list)

    @property
    def effective_rank(self) -> int:
        """The identification-relevant rank: numerical rank above the gap (FB-02)."""
        return self.numerical_rank

    def to_dict(self):
        return {
            "eigenvalues": self.eigenvalues.tolist(),
            "singular_values": self.singular_values.tolist(),
            "noise_floor": self.noise_floor,
            "signal_rank_mp": self.signal_rank_mp,
            "numerical_rank": self.numerical_rank,
            "effective_rank_entropy": self.effective_rank_entropy,
            "condition_number": self.condition_number,
            "names": self.names,
        }


def numerical_rank(singular_values: np.ndarray, rtol: float = 1e-2) -> int:
    """Count singular values above the noise gap (FB-02, subspace-style order).

    Model order = the number of singular values that are a non-negligible
    fraction ``rtol`` of the largest.  For a well-conditioned designed experiment
    every direction is excited and this returns full rank; for confounded passive
    data the feedback-suppressed directions fall below the gap and are dropped.
    This is the correct order-selection criterion — distinct from the MP
    spike-count, which detects low-rank common structure (the "8-vs-392"
    high-dimensional case, §REG-03) rather than excitation rank.
    """
    sv = np.asarray(singular_values, dtype=float)
    if sv.size == 0 or sv[0] <= 0:
        return 0
    return int(np.sum(sv > sv[0] * rtol))


def regressor_spectrum(X: np.ndarray, names: Optional[List[str]] = None,
                       noise_floor: Optional[float] = None,
                       rtol: float = 1e-2) -> SpectrumReport:
    """Spectrum of the regressor (correlation) covariance (EX-02).

    Returns the eigenvalues/singular values, condition number and three notions
    of rank: the MP *signal rank* (eigenvalues above ``noise_floor`` — common
    low-rank structure / the "8-vs-392" split), the *numerical rank* (singular
    values above the noise gap — the identification model order, FB-02), and an
    entropy-based participation ratio that needs no floor.
    """
    Z, _, _ = _standardize(np.asarray(X, dtype=float))
    n = Z.shape[0]
    # correlation matrix (standardized => correlation)
    C = (Z.T @ Z) / max(n - 1, 1)
    eig = np.linalg.eigvalsh(C)[::-1]
    eig = np.clip(eig, 0.0, None)
    sv = np.linalg.svd(Z, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 1e-15 else np.inf
    p = eig / eig.sum() if eig.sum() > 0 else np.ones_like(eig) / len(eig)
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    eff_entropy = float(np.exp(entropy))
    signal_rank = int(np.sum(eig > noise_floor)) if noise_floor is not None else None
    return SpectrumReport(
        eigenvalues=eig,
        singular_values=sv,
        noise_floor=noise_floor,
        signal_rank_mp=signal_rank,
        numerical_rank=numerical_rank(sv, rtol),
        effective_rank_entropy=eff_entropy,
        condition_number=cond,
        names=list(names) if names is not None else [],
    )


@dataclass
class UnidentifiableDirection:
    """One eigen-direction below the noise floor, expressed in regressor terms (EX-03)."""

    eigenvalue: float
    loadings: np.ndarray          # contribution of each regressor to the direction
    dominant: List[str]           # names of the regressors that dominate it

    def to_dict(self):
        return {
            "eigenvalue": float(self.eigenvalue),
            "loadings": self.loadings.tolist(),
            "dominant": self.dominant,
        }


def unidentifiable_directions(X: np.ndarray, rtol: float = 1e-2,
                              names: Optional[List[str]] = None,
                              top_loadings: int = 3) -> List[UnidentifiableDirection]:
    """Map excitation deficiency to specific knob/FF directions (EX-03).

    Returns the right-singular directions whose singular value falls below the
    noise gap (``rtol`` of the largest) — the near-null space of the regressor
    block — each annotated with the regressors that dominate it, so the
    experiment planner can target *which* directions to excite, not merely a
    scalar rank.  These are the feedback-suppressed / collinear directions that
    passive data cannot resolve.
    """
    Z, _, _ = _standardize(np.asarray(X, dtype=float))
    n, p = Z.shape
    # SVD of the standardized block: small singular values = deficient directions
    _, s, Vt = np.linalg.svd(Z / np.sqrt(max(n - 1, 1)), full_matrices=True)
    names = list(names) if names is not None else [f"x{j}" for j in range(p)]
    out: List[UnidentifiableDirection] = []
    smax = s[0] if s.size else 0.0
    # pad singular values to p (SVD returns min(n,p) of them)
    sv_full = np.zeros(p)
    sv_full[:len(s)] = s
    for k in range(p):
        if smax <= 0 or sv_full[k] <= smax * rtol:
            vec = Vt[k]
            loadings = vec ** 2  # share of the direction per regressor
            idx = np.argsort(loadings)[::-1][:top_loadings]
            out.append(UnidentifiableDirection(
                eigenvalue=float(sv_full[k] ** 2),
                loadings=loadings,
                dominant=[names[i] for i in idx],
            ))
    return out


@dataclass
class AccidentalExcitation:
    """Catalogue of 'free' excitation episodes in otherwise deficient records (EX-04)."""

    override_mask: np.ndarray         # events where used diverged from recommended
    control_off_mask: np.ndarray      # events where control was off
    override_episodes: List[tuple]    # (start, end) contiguous runs
    control_off_episodes: List[tuple]
    injected_variance: dict           # per-knob extra variance from overrides

    def to_dict(self):
        return {
            "n_override_events": int(self.override_mask.sum()),
            "n_control_off_events": int(self.control_off_mask.sum()),
            "override_episodes": self.override_episodes,
            "control_off_episodes": self.control_off_episodes,
            "injected_variance": self.injected_variance,
        }


def _runs(mask: np.ndarray) -> List[tuple]:
    runs = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def mine_accidental_excitation(
    recommended: np.ndarray,
    used: np.ndarray,
    control_off: Optional[np.ndarray] = None,
    knob_names: Optional[List[str]] = None,
    rel_tol: float = 1e-6,
) -> AccidentalExcitation:
    """Detect and catalogue accidental excitation (EX-04).

    Manual overrides (used != recommended) and control-off episodes inject
    variance into directions the controller normally suppresses.  These are
    detected, grouped into contiguous episodes, and offered as identification
    data and validation gold.
    """
    recommended = np.asarray(recommended, dtype=float)
    used = np.asarray(used, dtype=float)
    diff = used - recommended
    scale = np.maximum(np.nanstd(used, axis=0), 1e-9)
    override_event = np.any(np.abs(diff) > rel_tol * (1.0 + scale), axis=1)
    if control_off is None:
        control_off = np.zeros(len(used), dtype=bool)
    control_off = np.asarray(control_off, dtype=bool)
    names = knob_names or [f"u{j}" for j in range(used.shape[1])]
    injected = {}
    if override_event.any():
        for j, nm in enumerate(names):
            injected[nm] = float(np.var(diff[override_event, j]))
    else:
        injected = {nm: 0.0 for nm in names}
    return AccidentalExcitation(
        override_mask=override_event,
        control_off_mask=control_off,
        override_episodes=_runs(override_event),
        control_off_episodes=_runs(control_off),
        injected_variance=injected,
    )
