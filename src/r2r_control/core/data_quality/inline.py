"""Inline closed-loop data assessment (DQ-04..06)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ...contracts.provenance import ProvenanceLog, Verdict
from ...contracts.schema import EventTable
from ..excitation.vif import vif
from ..excitation.spectrum import regressor_spectrum
from ..excitation.feedback import feedback_test
from .. import stats
from .decisions import QualityReport


def assess_inline(table: EventTable, provenance: ProvenanceLog, vif_warn: float = 10.0,
                  min_effective_samples: int = 8, downweight: float = 0.5,
                  feedback_alpha: float = 0.01) -> QualityReport:
    """Detect confounding, feedback, divergence, sparsity, drift => decision (DQ-04..06).

    A ``DOWN_WEIGHT`` verdict carries ``weight = downweight`` (DQ-07). Separately,
    ``gain_usable`` is False when the knob settings respond to past outputs
    (closed-loop feedback, M3): such rows are endogenous and must not enter gain
    identification, whatever their conditioning (DQ-06).
    """
    diagnostics: Dict = {}
    R = table.regressors()
    names = table.regressor_names
    v = vif(R)
    spec = regressor_spectrum(R, names=names)
    diagnostics["vif"] = {n: float(x) for n, x in zip(names, v)}
    diagnostics["condition_number"] = spec.condition_number
    diagnostics["effective_rank_entropy"] = spec.effective_rank_entropy

    fb = feedback_test(table.used_knobs(), table.y_matrix(),
                       measured=np.all(table.measured_mask().reshape(table.n_events, -1), axis=1),
                       exclude=table.control_off_mask(), knob_names=table.knob_cols_used,
                       alpha=feedback_alpha)
    diagnostics["feedback"] = fb.to_dict()

    div = table.recommended_used_divergence()
    diagnostics["rec_used_divergence_var"] = {
        n: float(np.nanvar(div[:, j])) for j, n in enumerate(table.knob_cols_used)}
    eff_counts = {c: int(table.effective_measured_count(c)) for c in table.y_cols}
    diagnostics["effective_measured_count"] = eff_counts

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
            drift_flags[c] = {"adf": adf.to_dict(), "kpss": kpss.to_dict(),
                              "variance_ratio": vr.to_dict(), "drift_present": bool(drift)}
    diagnostics["stationarity"] = drift_flags

    max_vif = float(np.nanmax(v)) if len(v) else 1.0
    excitation_deficient = (max_vif > vif_warn or
                            spec.effective_rank_entropy < 0.5 * len(names))
    min_eff = min(eff_counts.values()) if eff_counts else 0
    reasons, verdict = [], Verdict.USE
    if excitation_deficient:
        verdict = Verdict.DOWN_WEIGHT
        reasons.append(f"closed-loop confounding: max VIF={max_vif:.1f}, eff-rank≈"
                       f"{spec.effective_rank_entropy:.1f}/{len(names)} => NOT for gain "
                       f"identification; use for noise floor, drift, validation, innovation")
    if min_eff < min_effective_samples:
        verdict = Verdict.DOWN_WEIGHT
        reasons.append(f"sparse metrology: min effective measured count {min_eff} "
                       f"< {min_effective_samples}")
    if fb.detected:
        reasons.append(f"closed-loop feedback: knob settings respond to past outputs "
                       f"(Granger F-test, adjusted p={fb.pvalue:.1e}) => rows endogenous, "
                       f"excluded from gain identification (DQ-06)")
    if not reasons:
        reasons.append("inline excitation adequate in measured directions")
    rationale = "; ".join(reasons)
    weight = downweight if verdict == Verdict.DOWN_WEIGHT else 1.0
    gain_usable = not fb.detected
    provenance.assume("inline gain directly identifiable",
                      gain_usable and not excitation_deficient,
                      "closed-loop feedback and/or confounding violate direct gain ID (DQ-04)",
                      {"feedback_detected": fb.detected,
                       "excitation_deficient": bool(excitation_deficient)})
    provenance.decide("inline", verdict, rationale, diagnostics, weight)
    return QualityReport("inline", verdict, rationale, diagnostics, weight,
                         gain_usable=gain_usable)
