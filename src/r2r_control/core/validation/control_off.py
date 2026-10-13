"""Control-off gold holdout with representativeness check (VAL-04, A5/E8).

Control-off episodes are used as validation gold. But they occur during
outages/maintenance — exactly when the tool may be in an anomalous, freshly
serviced, or warming state — so the "gold" can be a biased subsample whose
open-loop behavior does not match normal operation. This module compares the
tool state during each control-off episode against the normal-operation region
and excludes anomalous episodes from gold (A5/E8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..excitation.accidental import runs


@dataclass
class GoldHoldout:
    fit_idx: np.ndarray
    gold_idx: np.ndarray
    n_gold_episodes: int
    excluded_episodes: List[tuple] = field(default_factory=list)
    representativeness: List[dict] = field(default_factory=list)

    def to_dict(self):
        return {"n_fit": int(len(self.fit_idx)), "n_gold": int(len(self.gold_idx)),
                "n_gold_episodes": self.n_gold_episodes,
                "n_excluded_anomalous": len(self.excluded_episodes),
                "excluded_episodes": self.excluded_episodes,
                "representativeness": self.representativeness}


def control_off_gold(control_off_mask: np.ndarray, tool_state: Optional[np.ndarray] = None,
                     anomaly_z: float = 3.0) -> GoldHoldout:
    """Set aside control-off episodes as validation gold, excluding anomalous ones.

    ``tool_state`` is a per-event feature (or feature matrix) describing tool
    state. An episode whose mean tool state lies more than ``anomaly_z``
    standardized units from the normal (control-on) region is excluded from gold
    and reported (A5/E8). Fit and gold index sets are disjoint (VAL-04).
    """
    mask = np.asarray(control_off_mask, dtype=bool)
    episodes = runs(mask)
    kept, excluded, log = [], [], []
    if tool_state is None:
        for a, b in episodes:
            kept.extend(range(a, b + 1))
    else:
        ts = np.asarray(tool_state, dtype=float)
        if ts.ndim == 1:
            ts = ts[:, None]
        normal = ts[~mask]
        mu = np.nanmean(normal, axis=0)
        sd = np.nanstd(normal, axis=0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        for a, b in episodes:
            z = float(np.linalg.norm((np.nanmean(ts[a:b + 1], axis=0) - mu) / sd))
            ok = z <= anomaly_z
            log.append({"episode": (a, b), "z": z, "representative": bool(ok)})
            if ok:
                kept.extend(range(a, b + 1))
            else:
                excluded.append((a, b))
    return GoldHoldout(fit_idx=np.where(~mask)[0], gold_idx=np.array(sorted(kept), dtype=int),
                       n_gold_episodes=len(episodes) - len(excluded),
                       excluded_episodes=excluded, representativeness=log)
