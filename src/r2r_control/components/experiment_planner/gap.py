"""EP-01/02 — quantify and localize the identifiability gap.

Besides naming the weakly excited adjustable knobs, the report carries the gap
as signed **directions** over the adjustable knobs, so the infill step can aim
experiments at them (KI-6). Knobs that Component 1 flagged as low-confidence
(FB-04) can be passed in as ``target_features`` and become gap directions too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ...core.excitation.vif import vif, _standardize
from ...core.excitation.spectrum import regressor_spectrum


@dataclass
class GapReport:
    unidentifiable_features: List[str]
    vif: Dict[str, float]
    effective_rank: int
    n_directions: int
    confounding_summary: str
    directions: List[Dict[str, float]] = field(default_factory=list)   # over adjustable knobs

    def direction_matrix(self, adjustable: List[str]) -> np.ndarray:
        """Gap directions as unit rows over ``adjustable`` (shape m x d)."""
        if not self.directions:
            return np.zeros((0, len(adjustable)))
        V = np.array([[d.get(a, 0.0) for a in adjustable] for d in self.directions])
        n = np.linalg.norm(V, axis=1, keepdims=True)
        return V / np.where(n > 0, n, 1.0)

    def to_dict(self):
        return dict(self.__dict__)


def diagnose_gap(X_passive: np.ndarray, feature_names: List[str], adjustable: List[str],
                 rtol: float = 1e-2, target_features: Optional[List[str]] = None,
                 min_adjustable_share: float = 0.25) -> GapReport:
    """Localize the gap; only adjustable directions are actionable (EP-01/02).

    A right-singular direction of the standardized passive regressors whose
    singular value is at most ``rtol`` times the largest is unidentifiable. It
    is kept as a gap direction if at least ``min_adjustable_share`` of its
    squared loading falls on adjustable knobs (otherwise no experiment can
    excite it), restricted to those knobs and renormalized.
    """
    X = np.asarray(X_passive, dtype=float)
    spec = regressor_spectrum(X, names=feature_names)
    adj = [a for a in adjustable if a in feature_names]
    adj_idx = [feature_names.index(a) for a in adj]
    Z, _, _ = _standardize(X)
    n, p = Z.shape
    _, s, Vt = np.linalg.svd(Z / np.sqrt(max(n - 1, 1)), full_matrices=True)
    sv = np.zeros(p)
    sv[:len(s)] = s
    smax = sv[0] if sv.size else 0.0

    directions: List[Dict[str, float]] = []
    names_in_gap = set()
    for k in range(p):
        if smax > 0 and sv[k] > smax * rtol:
            continue
        v = Vt[k]
        share = float(np.sum(v[adj_idx] ** 2))
        if share < min_adjustable_share:
            continue
        va = v[adj_idx] / np.sqrt(share)
        directions.append({a: float(x) for a, x in zip(adj, va)})
        names_in_gap.update(a for a, x in zip(adj, va) if x ** 2 >= 0.1)
    for tf in target_features or []:
        if tf in adj:
            directions.append({a: (1.0 if a == tf else 0.0) for a in adj})
            names_in_gap.add(tf)
    unident = sorted(names_in_gap)
    return GapReport(
        unidentifiable_features=unident,
        vif={nm: float(x) for nm, x in zip(feature_names, vif(X))},
        effective_rank=spec.numerical_rank, n_directions=len(feature_names),
        confounding_summary=(f"effective rank {spec.numerical_rank}/{len(feature_names)}, "
                             f"condition number {spec.condition_number:.1f}; "
                             f"{len(unident)} adjustable directions unidentifiable"),
        directions=directions)
