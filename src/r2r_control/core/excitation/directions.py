"""Map excitation deficiency to specific knob/FF directions (EX-03)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .vif import _standardize


@dataclass
class UnidentifiableDirection:
    """One near-null direction of the regressor block, expressed in regressor terms."""

    eigenvalue: float
    loadings: np.ndarray
    dominant: List[str]

    def to_dict(self):
        return {"eigenvalue": float(self.eigenvalue), "loadings": self.loadings.tolist(),
                "dominant": self.dominant}


def unidentifiable_directions(X: np.ndarray, rtol: float = 1e-2,
                              names: Optional[List[str]] = None,
                              top_loadings: int = 3) -> List[UnidentifiableDirection]:
    """Right-singular directions below the noise gap — the directions to excite (EX-03)."""
    Z, _, _ = _standardize(np.asarray(X, dtype=float))
    n, p = Z.shape
    _, s, Vt = np.linalg.svd(Z / np.sqrt(max(n - 1, 1)), full_matrices=True)
    names = list(names) if names is not None else [f"x{j}" for j in range(p)]
    out: List[UnidentifiableDirection] = []
    smax = s[0] if s.size else 0.0
    sv_full = np.zeros(p)
    sv_full[:len(s)] = s
    for k in range(p):
        if smax <= 0 or sv_full[k] <= smax * rtol:
            loadings = Vt[k] ** 2
            idx = np.argsort(loadings)[::-1][:top_loadings]
            out.append(UnidentifiableDirection(eigenvalue=float(sv_full[k] ** 2),
                                               loadings=loadings,
                                               dominant=[names[i] for i in idx]))
    return out
