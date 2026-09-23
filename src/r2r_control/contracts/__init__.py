"""Contracts: schema and invariants depended on by everything (§DM, §IF).

This package depends on nothing else in r2r_control.
"""

from .measured import _is_measured, is_measured
from .shapes import check_shape_contract, ShapeContractError
from .schema import EventTable
from .provenance import ProvenanceLog, Decision, Assumption, Verdict
from .artifact import ModelArtifact, LOAD_BEARING_ASSUMPTIONS

__all__ = [
    "_is_measured", "is_measured", "check_shape_contract", "ShapeContractError",
    "EventTable", "ProvenanceLog", "Decision", "Assumption", "Verdict",
    "ModelArtifact", "LOAD_BEARING_ASSUMPTIONS",
]
