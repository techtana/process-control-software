"""The use / down-weight / reject decision record (DQ-03/06/08)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from ...contracts.provenance import Verdict


@dataclass
class QualityReport:
    source: str
    decision: Verdict
    rationale: str
    diagnostics: Dict = field(default_factory=dict)
    weight: float = 1.0

    def to_dict(self):
        return {"source": self.source, "decision": self.decision.value,
                "rationale": self.rationale, "diagnostics": self.diagnostics,
                "weight": self.weight}
