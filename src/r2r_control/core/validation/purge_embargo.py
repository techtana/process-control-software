"""Purge/embargo gap between train and test folds (VAL-03)."""

from __future__ import annotations

import numpy as np


def apply_embargo(train_idx: np.ndarray, test_idx: np.ndarray, embargo: int) -> np.ndarray:
    """Remove training indices within ``embargo`` samples of the test block (VAL-03)."""
    if embargo <= 0 or len(test_idx) == 0:
        return train_idx
    lo = test_idx.min() - embargo
    hi = test_idx.max() + embargo
    return train_idx[(train_idx < lo) | (train_idx > hi)]
