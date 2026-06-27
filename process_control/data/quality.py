"""Data-quality and assumption validation layer (§4.2, DQ-01..08).

This layer is the system's gatekeeper.  For every data source it makes the
source's assumptions explicit, detects critical violations, and emits a
*use / down-weight / reject* decision with a recorded rationale.  No downstream
component may consume data that has not passed through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..provenance import ProvenanceLog, Verdict
from ..foundations import stats
from ..foundations.excitation import vif, regressor_spectrum
from .schema import EventTable


# --------------------------------------------------------------------------- #
# DOE alias structure (DQ-02/03)
# --------------------------------------------------------------------------- #
@dataclass
class AliasReport:
    resolution: Optional[int]
    alias_chains: List[List[str]]
    aliased_terms: List[str]

    def to_dict(self):
        return {
            "resolution": self.resolution,
            "alias_chains": self.alias_chains,
            "aliased_terms": self.aliased_terms,
        }


def analyze_alias_structure(design: np.ndarray, names: List[str],
                            corr_tol: float = 0.999) -> AliasReport:
    """Parse a (coded +/-1) design and report its confounding structure (DQ-02).

    Two columns/effects are aliased when their coded contrast vectors are
    (anti-)collinear.  We build main-effect and two-factor-interaction columns
    and group those that are perfectly correlated into alias chains.  Resolution
    is inferred from the shortest word in the defining relation we can detect
    (main aliased with k-factor interaction => resolution k+1).
    """
    D = np.asarray(design, dtype=float)
    n, p = D.shape
    # center to +/- contrasts
    cols = {names[j]: D[:, j] - D[:, j].mean() for j in range(p)}
    # add two-factor interactions
    for i in range(p):
        for j in range(i + 1, p):
            key = f"{names[i]}*{names[j]}"
            v = D[:, i] * D[:, j]
            cols[key] = v - v.mean()
    keys = list(cols.keys())
    mat = np.column_stack([cols[k] for k in keys])
    norms = np.linalg.norm(mat, axis=0)
    norms_safe = np.where(norms > 1e-12, norms, 1.0)
    matn = mat / norms_safe
    corr = matn.T @ matn
    visited = set()
    chains: List[List[str]] = []
    for a in range(len(keys)):
        if a in visited or norms[a] < 1e-12:
            continue
        group = [a]
        for b in range(a + 1, len(keys)):
            if abs(corr[a, b]) >= corr_tol:
                group.append(b)
                visited.add(b)
        if len(group) > 1:
            chains.append([keys[g] for g in group])
        visited.add(a)
    aliased = sorted({t for ch in chains for t in ch})
    # resolution: shortest interaction length aliased with a main effect
    resolution = None
    for ch in chains:
        has_main = any("*" not in t for t in ch)
        if has_main:
            for t in ch:
                order = t.count("*") + 1
                if order >= 2:
                    res = order + 1
                    resolution = res if resolution is None else min(resolution, res)
    return AliasReport(resolution=resolution, alias_chains=chains, aliased_terms=aliased)


# --------------------------------------------------------------------------- #
# Covariate shift / staleness (DQ-02)
# --------------------------------------------------------------------------- #
def covariate_shift(reference: np.ndarray, current: np.ndarray) -> float:
    """Standardized distributional distance between operating points (DQ-02).

    A normalized Euclidean distance between the per-feature means scaled by the
    pooled std (a multivariate standardized mean difference).  Large values mean
    the DOE operating points sit far from the current inline operating point.
    """
    R = np.asarray(reference, dtype=float)
    C = np.asarray(current, dtype=float)
    mu_r, mu_c = np.nanmean(R, 0), np.nanmean(C, 0)
    sd = np.sqrt(0.5 * (np.nanvar(R, 0) + np.nanvar(C, 0)))
    sd = np.where(sd > 1e-12, sd, 1.0)
    d = (mu_r - mu_c) / sd
    return float(np.sqrt(np.mean(d ** 2)))


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
@dataclass
class QualityReport:
    source: str
    decision: Verdict
    rationale: str
    diagnostics: Dict = field(default_factory=dict)
    weight: float = 1.0

    def to_dict(self):
        return {
            "source": self.source,
            "decision": self.decision.value,
            "rationale": self.rationale,
            "diagnostics": self.diagnostics,
            "weight": self.weight,
        }


class DataQualityLayer:
    """The gatekeeper.  Assess each source; emit use/down-weight/reject (DQ)."""

    def __init__(
        self,
        staleness_horizon_days: float = 180.0,
        covariate_shift_reject: float = 3.0,
        covariate_shift_downweight: float = 1.0,
        vif_warn: float = 10.0,
        min_effective_samples: int = 8,
        provenance: Optional[ProvenanceLog] = None,
    ):
        self.staleness_horizon_days = staleness_horizon_days
        self.covariate_shift_reject = covariate_shift_reject
        self.covariate_shift_downweight = covariate_shift_downweight
        self.vif_warn = vif_warn
        self.min_effective_samples = min_effective_samples
        self.provenance = provenance or ProvenanceLog()

    # -- DOE (DQ-01/02/03) ----------------------------------------------
    def assess_doe(
        self,
        design: np.ndarray,
        factor_names: List[str],
        age_days: float,
        inline_operating_point: Optional[np.ndarray] = None,
    ) -> QualityReport:
        alias = analyze_alias_structure(design, factor_names)
        diagnostics: Dict = {"alias": alias.to_dict(), "age_days": age_days}
        self.provenance.assume("DOE factor levels span operating range", True,
                               "assumed; verify against process limits")
        # The alias structure is always *known* once parsed; resolution=None simply
        # means no main-effect aliasing was detected (the high-resolution case).
        res_detail = (f"no aliasing detected (full resolution)" if alias.resolution is None
                      else f"resolution {alias.resolution}, "
                           f"{len(alias.aliased_terms)} aliased terms")
        self.provenance.assume("DOE alias/resolution known", True, res_detail,
                               alias.to_dict())

        verdict = Verdict.USE
        reasons = []
        weight = 1.0

        # staleness => gain may be invalid (DQ-02/03)
        if age_days > self.staleness_horizon_days:
            verdict = Verdict.DOWN_WEIGHT
            weight = float(np.exp(-(age_days - self.staleness_horizon_days)
                                  / self.staleness_horizon_days))
            reasons.append(
                f"stale: {age_days:.0f}d > {self.staleness_horizon_days:.0f}d horizon "
                f"=> down-weight gain (w={weight:.2f})")
            diagnostics["staleness_weight"] = weight

        # covariate shift vs current operating point (DQ-02)
        if inline_operating_point is not None:
            shift = covariate_shift(design, inline_operating_point)
            diagnostics["covariate_shift"] = shift
            if shift > self.covariate_shift_reject:
                verdict = Verdict.REJECT
                reasons.append(f"covariate shift {shift:.2f} > reject threshold; "
                               f"DOE operating point too far from current epoch")
            elif shift > self.covariate_shift_downweight:
                if verdict != Verdict.REJECT:
                    verdict = Verdict.DOWN_WEIGHT
                weight = min(weight, 1.0 / (1.0 + shift))
                reasons.append(f"covariate shift {shift:.2f} => down-weight")

        if alias.aliased_terms:
            reasons.append(f"aliased terms excluded from identifiability claims: "
                           f"{alias.aliased_terms} (resolution {alias.resolution})")

        if not reasons:
            reasons.append("DOE within range, fresh, aligned => use for wide-range "
                           "gain/curvature")
        rationale = "; ".join(reasons)
        self.provenance.decide("DOE", verdict, rationale, diagnostics, weight)
        return QualityReport("DOE", verdict, rationale, diagnostics, weight)

    # -- Inline (DQ-04/05/06) -------------------------------------------
    def assess_inline(self, table: EventTable) -> QualityReport:
        diagnostics: Dict = {}
        R = table.regressors()
        names = table.regressor_names
        # closed-loop confounding (EX)
        v = vif(R)
        spec = regressor_spectrum(R, names=names)
        diagnostics["vif"] = {n: float(x) for n, x in zip(names, v)}
        diagnostics["condition_number"] = spec.condition_number
        diagnostics["effective_rank_entropy"] = spec.effective_rank_entropy

        # recommended vs used divergence (EX-04, DM-05)
        div = table.recommended_used_divergence()
        diagnostics["rec_used_divergence_var"] = {
            n: float(np.nanvar(div[:, j])) for j, n in enumerate(table.knob_cols_used)}

        # metrology alignment / effective measured count (DM-02/03)
        eff_counts = {c: int(table.effective_measured_count(c)) for c in table.y_cols}
        diagnostics["effective_measured_count"] = eff_counts

        # stationarity diagnostics on raw output to confirm drift (DQ-05)
        Y = table.y_matrix()
        drift_flags = {}
        for j, c in enumerate(table.y_cols):
            col = Y[:, j]
            col = col[np.isfinite(col)]
            if len(col) >= 12:
                adf = stats.adf_test(col)
                kpss = stats.kpss_test(col)
                vr = stats.variance_ratio(col)
                drift = (adf.reject_null is False) or (kpss.reject_null is True)
                drift_flags[c] = {
                    "adf": adf.to_dict(), "kpss": kpss.to_dict(),
                    "variance_ratio": vr.to_dict(), "drift_present": bool(drift)}
        diagnostics["stationarity"] = drift_flags

        # decision (DQ-06): inline alone not for gain when excitation deficient
        max_vif = float(np.nanmax(v)) if len(v) else 1.0
        excitation_deficient = (max_vif > self.vif_warn or
                                spec.effective_rank_entropy < 0.5 * len(names))
        min_eff = min(eff_counts.values()) if eff_counts else 0
        reasons = []
        verdict = Verdict.USE
        if excitation_deficient:
            verdict = Verdict.DOWN_WEIGHT
            reasons.append(
                f"closed-loop confounding: max VIF={max_vif:.1f}, eff-rank≈"
                f"{spec.effective_rank_entropy:.1f}/{len(names)} => NOT for gain "
                f"identification; use for noise floor, drift, validation, innovation")
        if min_eff < self.min_effective_samples:
            verdict = Verdict.DOWN_WEIGHT
            reasons.append(f"sparse metrology: min effective measured count {min_eff} "
                           f"< {self.min_effective_samples}")
        if not reasons:
            reasons.append("inline excitation adequate in measured directions")
        rationale = "; ".join(reasons)

        self.provenance.assume("inline gain directly identifiable",
                               not excitation_deficient,
                               "closed-loop confounding violates direct gain ID (DQ-04)")
        self.provenance.decide("inline", verdict, rationale, diagnostics)
        return QualityReport("inline", verdict, rationale, diagnostics)

    # -- joint use (DQ-07) ----------------------------------------------
    def combine(self, doe: QualityReport, inline: QualityReport) -> Dict:
        """Provenance- and weight-aware combination contract (DQ-07).

        Records the provenance and weight of every contribution and prevents a
        stale or covariate-shifted DOE point from dominating an estimate at the
        current operating point.  Combination is explicit, never an unlabeled
        concatenation.
        """
        doe_w = doe.weight if doe.decision != Verdict.REJECT else 0.0
        contract = {
            "doe": {"decision": doe.decision.value, "weight": doe_w,
                    "role": "wide-range gain/curvature"},
            "inline": {"decision": inline.decision.value, "weight": inline.weight,
                       "role": "operating-point-local correction + noise"},
            "policy": "weighted/hierarchical, never unlabeled concatenation",
        }
        self.provenance.note(f"joint-use contract: DOE w={doe_w:.2f} (gain), "
                             f"inline w={inline.weight:.2f} (local+noise)")
        return contract
