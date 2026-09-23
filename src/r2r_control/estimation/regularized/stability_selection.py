"""Stability selection over block-bootstrap resamples (REG-06, E12).

Reports selection frequency rather than a single fit. E12: correlated sensor
groups split their selection frequency — each member can look unstable though
the group is stably important — so we also report **group-wise** selection
stability (how often *any* member of the group is selected).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ...core.validation.block_bootstrap import block_bootstrap
from .elastic_net import elastic_net_fit


@dataclass
class StabilityResult:
    names: List[str]
    selection_frequency: np.ndarray
    mean_coef: np.ndarray
    group_frequency: Dict[str, float] = field(default_factory=dict)
    groups: Dict[str, List[str]] = field(default_factory=dict)

    def ranked(self, top: Optional[int] = None):
        order = np.argsort(self.selection_frequency)[::-1]
        if top:
            order = order[:top]
        return [(self.names[i], float(self.selection_frequency[i]), float(self.mean_coef[i]))
                for i in order]

    def to_dict(self):
        return {"names": self.names, "selection_frequency": self.selection_frequency.tolist(),
                "mean_coef": self.mean_coef.tolist(), "ranked": self.ranked(),
                "group_frequency": self.group_frequency, "groups": self.groups}


def infer_groups(X: np.ndarray, names: List[str], corr_threshold: float = 0.8
                 ) -> Dict[str, List[str]]:
    """Group features by high pairwise |correlation| (E12)."""
    X = np.asarray(X, dtype=float)
    sd = X.std(0)
    Z = (X - X.mean(0)) / np.where(sd > 1e-12, sd, 1.0)
    C = np.abs((Z.T @ Z) / max(len(X) - 1, 1))
    assigned, groups = set(), {}
    for j in range(len(names)):
        if names[j] in assigned:
            continue
        members = [names[j]] + [names[k] for k in range(j + 1, len(names))
                                if names[k] not in assigned and C[j, k] >= corr_threshold]
        assigned.update(members)
        if len(members) > 1:
            groups[f"group{len(groups)}"] = members
    return groups


def stability_selection(X: np.ndarray, y: np.ndarray, names: List[str],
                        n_resamples: int = 100, block_size: int = 10, lam: float = 0.1,
                        l1_ratio: float = 0.7, rng: Optional[np.random.Generator] = None,
                        groups: Optional[Dict[str, List[str]]] = None,
                        group_corr_threshold: float = 0.8) -> StabilityResult:
    """Elastic net over block-bootstrap resamples; per-feature + group frequencies."""
    if rng is None:
        rng = np.random.default_rng(0)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    if groups is None:
        groups = infer_groups(X, names, group_corr_threshold)
    idx_of = {nm: i for i, nm in enumerate(names)}
    counts, coef_sum = np.zeros(p), np.zeros(p)
    group_hits = {g: 0 for g in groups}
    n_eff = 0
    for _ in range(n_resamples):
        idx = block_bootstrap(n, min(block_size, max(n // 3, 1)), rng)
        if len(np.unique(idx)) < 3:
            continue
        beta = elastic_net_fit(X[idx], y[idx], lam=lam, l1_ratio=l1_ratio)
        selected = np.abs(beta) > 1e-6
        counts += selected
        coef_sum += beta
        n_eff += 1
        for g, members in groups.items():
            if any(selected[idx_of[m]] for m in members if m in idx_of):
                group_hits[g] += 1
    n_eff = max(n_eff, 1)
    return StabilityResult(names=list(names), selection_frequency=counts / n_eff,
                           mean_coef=coef_sum / n_eff,
                           group_frequency={g: h / n_eff for g, h in group_hits.items()},
                           groups=groups)
