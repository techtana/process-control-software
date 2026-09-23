"""Model-artifact container (IF-01, NFR-03).

The single schema Components 1 and 6 publish and Components 2, 3, 5 consume: the
model, its uncertainty covariance, the identifiable-direction report, and the
data-usage log. Carrying them together in one provenance-tracked object is what
keeps the components decoupled (§11) — they communicate only through artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .provenance import ProvenanceLog


# The load-bearing assumptions of §1.4. Each artifact records which ones it
# relies on, so a silent violation cannot propagate through the shared
# services unnoticed (§1.4, NFR-06).
LOAD_BEARING_ASSUMPTIONS = {
    "A1": "gain time-invariance (additive drift, fixed M); guard: realized-gain monitoring",
    "A2": "u0 validity over the analysis horizon; guard: u0 validity horizon + re-baseline",
    "A3": "M accurate enough for decoupling; guard: propagate M uncertainty + phantom-FF quarantine",
    "A4": "measurement delay < MHE horizon; guard: out-of-sequence / slow-bias routing",
    "A5": "control-off episodes representative; guard: tool-state representativeness check",
    "A6": "injected excitation actually lands; guard: open-loop / residual-excitation accounting",
}


@dataclass
class ModelArtifact:
    """A published model + everything needed to audit and consume it (IF-01)."""

    kind: str                                  # 'fb' | 'regression'
    coef: np.ndarray                           # M (n_out x n_knob) or coef map
    param_cov: Optional[np.ndarray]            # parameter-uncertainty covariance (FB-07)
    sigma2: Optional[np.ndarray]               # per-output residual variance
    identifiable: Any                          # identifiable-direction report
    provenance: ProvenanceLog                  # data-usage decisions, floors, validation
    feature_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    assumptions_relied_on: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    def relative_uncertainty(self) -> float:
        """Scalar relative-uncertainty estimate on the coefficient map (CF-03/SIM-02)."""
        if self.param_cov is None or self.sigma2 is None:
            return float("nan")
        scale = np.linalg.norm(self.coef) + 1e-12
        return float(np.sqrt(np.trace(self.param_cov) * np.mean(self.sigma2)) / scale)

    def assumption_ledger(self) -> Dict[str, str]:
        return {a: LOAD_BEARING_ASSUMPTIONS[a] for a in self.assumptions_relied_on
                if a in LOAD_BEARING_ASSUMPTIONS}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "coef_shape": list(np.asarray(self.coef).shape),
            "relative_uncertainty": self.relative_uncertainty(),
            "identifiable": self.identifiable.to_dict()
            if hasattr(self.identifiable, "to_dict") else self.identifiable,
            "assumptions_relied_on": self.assumption_ledger(),
            "extras": self.extras,
            "provenance": self.provenance.to_dict(),
        }
