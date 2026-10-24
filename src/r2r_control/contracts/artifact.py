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


def expected_frobenius_error2(param_cov: np.ndarray, sigma2: np.ndarray) -> float:
    """``E||B_hat - B||_F^2 = trace(cov) * sum_j sigma2_j`` for a multi-output linear fit."""
    return float(np.trace(np.asarray(param_cov, dtype=float)) *
                 np.sum(np.asarray(sigma2, dtype=float)))


def variance_rel_uncertainty(param_cov: Optional[np.ndarray], sigma2: Optional[np.ndarray],
                             coef: np.ndarray) -> float:
    """Relative Frobenius error expected from estimation variance alone (KI-2).

    Same scale as the simulator's multiplicative ``rel_uncertainty_M``
    (``E||M*(1+r*N)-M||_F = r*||M||_F``). Summing ``sigma2`` over outputs, not
    averaging it, is what makes the two comparable.
    """
    if param_cov is None or sigma2 is None:
        return float("nan")
    return float(np.sqrt(expected_frobenius_error2(param_cov, sigma2)) /
                 (np.linalg.norm(coef) + 1e-12))


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
    bias_rel_uncertainty: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)

    def relative_uncertainty(self) -> float:
        """Relative uncertainty on the coefficient map: variance and bias in quadrature
        (CF-03/SIM-02, KI-2)."""
        var = variance_rel_uncertainty(self.param_cov, self.sigma2, self.coef)
        return float(np.hypot(var, self.bias_rel_uncertainty))

    def assumption_ledger(self) -> Dict[str, str]:
        return {a: LOAD_BEARING_ASSUMPTIONS[a] for a in self.assumptions_relied_on
                if a in LOAD_BEARING_ASSUMPTIONS}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "coef_shape": list(np.asarray(self.coef).shape),
            "relative_uncertainty": self.relative_uncertainty(),
            "bias_rel_uncertainty": self.bias_rel_uncertainty,
            "identifiable": self.identifiable.to_dict()
            if hasattr(self.identifiable, "to_dict") else self.identifiable,
            "assumptions_relied_on": self.assumption_ledger(),
            "extras": self.extras,
            "provenance": self.provenance.to_dict(),
        }
