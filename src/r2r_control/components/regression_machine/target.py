"""REG-04 — raw-output vs post-control innovation target."""

from __future__ import annotations

from typing import Optional

import numpy as np


def innovation_target(y_raw: np.ndarray, innovation_series: Optional[np.ndarray] = None,
                      provenance=None) -> np.ndarray:
    """Post-control innovation/residual target (REG-04).

    Separates what the loop already tracks (drift, slow tool state) from what
    sensors can additionally explain; the target is closer to stationary, which
    eases temporal validation. Sensors that explain it are feedforward candidates
    (the same object as DIAG-01).
    """
    if innovation_series is not None:
        if provenance is not None:
            provenance.note("using provided post-control innovation sequence as target (REG-04)")
        return np.asarray(innovation_series, dtype=float)
    y = np.asarray(y_raw, dtype=float)
    innov = np.full_like(y, np.nan)
    innov[1:] = y[1:] - y[:-1]
    if provenance is not None:
        provenance.note("innovation target proxied by one-step output difference "
                        "(closer to stationary; eases temporal validation, REG-04)")
    return innov
