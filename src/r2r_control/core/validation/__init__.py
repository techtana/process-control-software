"""Validation framework (§4.6, VAL-01..06)."""

from .forward_chaining import (Split, NaiveKFoldError, naive_kfold_guard,
                               forward_chaining_splits, ForwardValidationResult,
                               forward_validate)
from .purge_embargo import apply_embargo
from .block_bootstrap import block_bootstrap, contiguous_block_permutation
from .control_off import GoldHoldout, control_off_gold
from .concept_drift import ConceptDriftResult, concept_drift_test

__all__ = [
    "Split", "NaiveKFoldError", "naive_kfold_guard", "forward_chaining_splits",
    "ForwardValidationResult", "forward_validate", "apply_embargo",
    "block_bootstrap", "contiguous_block_permutation", "GoldHoldout",
    "control_off_gold", "ConceptDriftResult", "concept_drift_test",
]
