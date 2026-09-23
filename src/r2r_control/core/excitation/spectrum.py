"""Regressor-covariance spectrum and the two notions of rank (EX-02, FB-02).

* **numerical rank** (singular values above a relative gap) — how many directions
  are *excited enough to identify*? Used for model order (FB-02) and gap
  localization (EP-01).
* **MP signal rank** (eigenvalues above the noise floor) — how many directions
  carry *common low-rank structure*? Used for the high-dimensional split (REG-03).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .vif import _standardize


@dataclass
class SpectrumReport:
    eigenvalues: np.ndarray
    singular_values: np.ndarray
    noise_floor: Optional[float]
    signal_rank_mp: Optional[int]
    numerical_rank: int
    effective_rank_entropy: float
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
    """Count singular values above the noise gap (FB-02, subspace-style order)."""
    sv = np.asarray(singular_values, dtype=float)
    if sv.size == 0 or sv[0] <= 0:
        return 0
    return int(np.sum(sv > sv[0] * rtol))


def regressor_spectrum(X: np.ndarray, names: Optional[List[str]] = None,
                       noise_floor: Optional[float] = None,
                       rtol: float = 1e-2) -> SpectrumReport:
    """Spectrum of the regressor (correlation) covariance (EX-02)."""
    Z, _, _ = _standardize(np.asarray(X, dtype=float))
    n = Z.shape[0]
    C = (Z.T @ Z) / max(n - 1, 1)
    eig = np.clip(np.linalg.eigvalsh(C)[::-1], 0.0, None)
    sv = np.linalg.svd(Z, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 1e-15 else np.inf
    p = eig / eig.sum() if eig.sum() > 0 else np.ones_like(eig) / len(eig)
    p = p[p > 0]
    eff_entropy = float(np.exp(-np.sum(p * np.log(p))))
    signal_rank = int(np.sum(eig > noise_floor)) if noise_floor is not None else None
    return SpectrumReport(
        eigenvalues=eig, singular_values=sv, noise_floor=noise_floor,
        signal_rank_mp=signal_rank, numerical_rank=numerical_rank(sv, rtol),
        effective_rank_entropy=eff_entropy, condition_number=cond,
        names=list(names) if names is not None else [],
    )
