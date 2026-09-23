"""DOE assumptions, alias/resolution parsing, covariate shift, staleness (DQ-01..03, E13)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ...contracts.provenance import ProvenanceLog, Verdict
from .decisions import QualityReport


@dataclass
class AliasReport:
    resolution: Optional[int]
    alias_chains: List[List[str]]
    aliased_terms: List[str]

    def to_dict(self):
        return {"resolution": self.resolution, "alias_chains": self.alias_chains,
                "aliased_terms": self.aliased_terms}


def analyze_alias_structure(design: np.ndarray, names: List[str],
                            corr_tol: float = 0.999) -> AliasReport:
    """Parse a (coded +/-1) design and report its confounding structure (DQ-02, E13)."""
    D = np.asarray(design, dtype=float)
    p = D.shape[1]
    cols = {names[j]: D[:, j] - D[:, j].mean() for j in range(p)}
    for i in range(p):
        for j in range(i + 1, p):
            v = D[:, i] * D[:, j]
            cols[f"{names[i]}*{names[j]}"] = v - v.mean()
    keys = list(cols)
    mat = np.column_stack([cols[k] for k in keys])
    norms = np.linalg.norm(mat, axis=0)
    matn = mat / np.where(norms > 1e-12, norms, 1.0)
    corr = matn.T @ matn
    visited, chains = set(), []
    for a in range(len(keys)):
        if a in visited or norms[a] < 1e-12:
            continue
        group = [a] + [b for b in range(a + 1, len(keys)) if abs(corr[a, b]) >= corr_tol]
        visited.update(group)
        if len(group) > 1:
            chains.append([keys[g] for g in group])
    aliased = sorted({t for ch in chains for t in ch})
    resolution = None
    for ch in chains:
        if any("*" not in t for t in ch):
            for t in ch:
                order = t.count("*") + 1
                if order >= 2:
                    res = order + 1
                    resolution = res if resolution is None else min(resolution, res)
    return AliasReport(resolution=resolution, alias_chains=chains, aliased_terms=aliased)


def covariate_shift(reference: np.ndarray, current: np.ndarray) -> float:
    """Standardized distributional distance between operating points (DQ-02)."""
    R = np.asarray(reference, dtype=float)
    C = np.asarray(current, dtype=float)
    sd = np.sqrt(0.5 * (np.nanvar(R, 0) + np.nanvar(C, 0)))
    d = (np.nanmean(R, 0) - np.nanmean(C, 0)) / np.where(sd > 1e-12, sd, 1.0)
    return float(np.sqrt(np.mean(d ** 2)))


def assess_doe(design: np.ndarray, factor_names: List[str], age_days: float,
               provenance: ProvenanceLog, inline_operating_point: Optional[np.ndarray] = None,
               staleness_horizon_days: float = 180.0, covariate_shift_reject: float = 3.0,
               covariate_shift_downweight: float = 1.0) -> QualityReport:
    """Assess DOE: alias/resolution, covariate shift, staleness => decision (DQ-01..03)."""
    alias = analyze_alias_structure(design, factor_names)
    diagnostics: Dict = {"alias": alias.to_dict(), "age_days": age_days}
    provenance.assume("DOE factor levels span operating range", True,
                      "assumed; verify against process limits")
    res_detail = ("no aliasing detected (full resolution)" if alias.resolution is None
                  else f"resolution {alias.resolution}, {len(alias.aliased_terms)} aliased terms")
    provenance.assume("DOE alias/resolution known", True, res_detail, alias.to_dict())

    verdict, reasons, weight = Verdict.USE, [], 1.0
    if age_days > staleness_horizon_days:
        verdict = Verdict.DOWN_WEIGHT
        weight = float(np.exp(-(age_days - staleness_horizon_days) / staleness_horizon_days))
        reasons.append(f"stale: {age_days:.0f}d > {staleness_horizon_days:.0f}d horizon "
                       f"=> down-weight gain (w={weight:.2f})")
        diagnostics["staleness_weight"] = weight
    if inline_operating_point is not None:
        shift = covariate_shift(design, inline_operating_point)
        diagnostics["covariate_shift"] = shift
        if shift > covariate_shift_reject:
            verdict = Verdict.REJECT
            reasons.append(f"covariate shift {shift:.2f} > reject threshold; DOE operating "
                           f"point too far from current epoch")
        elif shift > covariate_shift_downweight:
            verdict = Verdict.DOWN_WEIGHT
            weight = min(weight, 1.0 / (1.0 + shift))
            reasons.append(f"covariate shift {shift:.2f} => down-weight")
    if alias.aliased_terms:
        reasons.append(f"aliased terms excluded from identifiability claims: "
                       f"{alias.aliased_terms} (resolution {alias.resolution})")
    if not reasons:
        reasons.append("DOE within range, fresh, aligned => use for wide-range gain/curvature")
    rationale = "; ".join(reasons)
    provenance.decide("DOE", verdict, rationale, diagnostics, weight)
    return QualityReport("DOE", verdict, rationale, diagnostics, weight)
