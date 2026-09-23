"""EP-01/02 — quantify and localize the identifiability gap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ...core.excitation.vif import vif
from ...core.excitation.spectrum import regressor_spectrum
from ...core.excitation.directions import unidentifiable_directions


@dataclass
class GapReport:
    unidentifiable_features: List[str]
    vif: Dict[str, float]
    effective_rank: int
    n_directions: int
    confounding_summary: str

    def to_dict(self):
        return dict(self.__dict__)


def diagnose_gap(X_passive: np.ndarray, feature_names: List[str],
                 adjustable: List[str]) -> GapReport:
    """Localize the gap; only adjustable directions are actionable (EP-01/02)."""
    spec = regressor_spectrum(X_passive, names=feature_names)
    adj = set(adjustable)
    unident = sorted({d for u in unidentifiable_directions(X_passive, names=feature_names)
                      for d in u.dominant if d in adj})
    return GapReport(
        unidentifiable_features=unident,
        vif={n: float(x) for n, x in zip(feature_names, vif(X_passive))},
        effective_rank=spec.numerical_rank, n_directions=len(feature_names),
        confounding_summary=(f"effective rank {spec.numerical_rank}/{len(feature_names)}, "
                             f"condition number {spec.condition_number:.1f}; "
                             f"{len(unident)} adjustable directions unidentifiable"))
