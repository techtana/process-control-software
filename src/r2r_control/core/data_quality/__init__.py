"""Data-quality and assumption validation layer (§4.2, DQ-01..08).

The gatekeeper: for every data source it makes the source's assumptions
explicit, detects critical violations, and emits a use/down-weight/reject
decision with a recorded rationale.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ...contracts.provenance import ProvenanceLog
from ...contracts.schema import EventTable
from .decisions import QualityReport
from .doe import AliasReport, analyze_alias_structure, covariate_shift, assess_doe
from .inline import assess_inline
from .joint import combine


class DataQualityLayer:
    """Holds thresholds + a provenance log and delegates to the per-source checks."""

    def __init__(self, staleness_horizon_days: float = 180.0,
                 covariate_shift_reject: float = 3.0,
                 covariate_shift_downweight: float = 1.0, vif_warn: float = 10.0,
                 min_effective_samples: int = 8,
                 provenance: Optional[ProvenanceLog] = None):
        self.staleness_horizon_days = staleness_horizon_days
        self.covariate_shift_reject = covariate_shift_reject
        self.covariate_shift_downweight = covariate_shift_downweight
        self.vif_warn = vif_warn
        self.min_effective_samples = min_effective_samples
        self.provenance = provenance or ProvenanceLog()

    def assess_doe(self, design: np.ndarray, factor_names: List[str], age_days: float,
                   inline_operating_point: Optional[np.ndarray] = None) -> QualityReport:
        return assess_doe(design, factor_names, age_days, self.provenance,
                          inline_operating_point=inline_operating_point,
                          staleness_horizon_days=self.staleness_horizon_days,
                          covariate_shift_reject=self.covariate_shift_reject,
                          covariate_shift_downweight=self.covariate_shift_downweight)

    def assess_inline(self, table: EventTable) -> QualityReport:
        return assess_inline(table, self.provenance, vif_warn=self.vif_warn,
                             min_effective_samples=self.min_effective_samples)

    def combine(self, doe: QualityReport, inline: QualityReport) -> Dict:
        return combine(doe, inline, self.provenance)


__all__ = ["DataQualityLayer", "QualityReport", "AliasReport", "analyze_alias_structure",
           "covariate_shift", "assess_doe", "assess_inline", "combine"]
