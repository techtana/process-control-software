"""Harris index with a finite-sample confidence interval (SIM-06, E10)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core.noise_floor.analytic import harris_index
from ..core.validation.block_bootstrap import block_bootstrap


def harris_with_ci(output: np.ndarray, delay: int = 1, n_boot: int = 200,
                   block_size: int = 10, alpha: float = 0.05,
                   rng: Optional[np.random.Generator] = None) -> Tuple[float, float, float]:
    """Harris index and a block-bootstrap (1-alpha) interval."""
    if rng is None:
        rng = np.random.default_rng(0)
    y = np.asarray(output, dtype=float)
    y = y[np.isfinite(y)]
    point = harris_index(y, delay=delay)
    n = len(y)
    if n < 4 * (delay + 2):
        return point, float("nan"), float("nan")
    boot = np.array([harris_index(y[block_bootstrap(n, min(block_size, max(n // 3, 1)), rng)],
                                  delay=delay) for _ in range(n_boot)])
    return (point, float(np.nanpercentile(boot, 100 * alpha / 2)),
            float(np.nanpercentile(boot, 100 * (1 - alpha / 2))))
