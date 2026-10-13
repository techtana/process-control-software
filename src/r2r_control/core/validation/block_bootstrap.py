"""Block bootstrap and contiguous-block surrogates (VAL-05, shared with NF-03).

The contiguous-block logic is shared between resampling-based selection
(stability selection, VAL-05) and the structure-preserving empirical noise floor
(NF-03), so "block" means one thing system-wide.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def block_bootstrap(n: int, block_size: int, rng: Optional[np.random.Generator] = None,
                    n_blocks: Optional[int] = None) -> np.ndarray:
    """Contiguous-block bootstrap resample of indices (VAL-05)."""
    if rng is None:
        rng = np.random.default_rng(0)
    if n_blocks is None:
        n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, max(n - block_size + 1, 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, min(s + block_size, n)) for s in starts])
    return idx[:n]


def contiguous_block_permutation(vals: np.ndarray, block_size: int,
                                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Reassemble a series from a random ordering of its contiguous blocks (NF-03).

    Preserves within-block autocorrelation and the exact marginal (it permutes
    the actual values), while destroying long-range and cross-series structure.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    vals = np.asarray(vals, dtype=float)
    m = len(vals)
    if m <= block_size or block_size < 1:
        # too short to block-permute: a circular shift still keeps autocorrelation
        return np.roll(vals, int(rng.integers(0, max(m, 1))))
    blocks = [vals[s:s + block_size] for s in range(0, m, block_size)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])[:m]
