"""Joint-use contract: provenance-weighted combination guard (DQ-07)."""

from __future__ import annotations

from typing import Dict

from ...contracts.provenance import ProvenanceLog, Verdict
from .decisions import QualityReport


def combine(doe: QualityReport, inline: QualityReport, provenance: ProvenanceLog) -> Dict:
    """Record provenance + weight of each contribution; keep stale DOE from dominating (DQ-07)."""
    doe_w = doe.weight if doe.decision != Verdict.REJECT else 0.0
    contract = {
        "doe": {"decision": doe.decision.value, "weight": doe_w,
                "role": "wide-range gain/curvature"},
        "inline": {"decision": inline.decision.value, "weight": inline.weight,
                   "role": "operating-point-local correction + noise"},
        "policy": "weighted/hierarchical, never unlabeled concatenation",
    }
    provenance.note(f"joint-use contract: DOE w={doe_w:.2f} (gain), "
                    f"inline w={inline.weight:.2f} (local+noise)")
    return contract
