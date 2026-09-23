"""Component 1 — FB / Process-Dynamic Model Identification (§5, FB-01..08).

Wraps the shared `SensitivityCore` (gain) with the Component-1-only dynamics
layer. Fuses DOE and inline under the joint-use contract, sets order by the
singular-value noise gap, reports identifiability limits, mines accidental
excitation, sources ``u0`` through the baseline service (A2/E3), and refuses to
emit attributions the data cannot support (FB-08).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ...contracts.schema import EventTable
from ...contracts.shapes import check_shape_contract
from ...contracts.provenance import ProvenanceLog, Verdict
from ...contracts.artifact import ModelArtifact
from ...core import stats
from ...core.counterfactual.reconstruct import reconstruct
from ...core.counterfactual.baseline import estimate_baseline
from ...core.data_quality import DataQualityLayer, QualityReport
from ...core.excitation.accidental import mine_accidental_excitation, AccidentalExcitation
from ...estimation.sensitivity.core import SensitivityCore
from ...estimation.dynamics.state_space import identify_first_order_dynamics


@dataclass
class FBModel:
    M: np.ndarray
    A: Optional[np.ndarray]
    param_cov: np.ndarray
    sigma2: np.ndarray
    identifiable: object
    innovation_diag: Dict
    accidental: AccidentalExcitation
    provenance: ProvenanceLog
    knob_names: List[str]
    out_names: List[str]
    u0: np.ndarray
    baseline_info: Dict = field(default_factory=dict)
    low_confidence_entries: List[tuple] = field(default_factory=list)
    assumptions_relied_on: List[str] = field(default_factory=list)

    @property
    def n_out(self) -> int:
        return self.M.shape[0]

    @property
    def n_knob(self) -> int:
        return self.M.shape[1]

    def u0_static(self) -> np.ndarray:
        return self.u0 if np.ndim(self.u0) == 1 else np.mean(self.u0, axis=0)

    def relative_uncertainty(self) -> float:
        scale = np.linalg.norm(self.M) + 1e-12
        return float(np.sqrt(np.trace(self.param_cov) * np.mean(self.sigma2)) / scale)

    def as_artifact(self) -> ModelArtifact:
        return ModelArtifact(
            kind="fb", coef=self.M, param_cov=self.param_cov, sigma2=self.sigma2,
            identifiable=self.identifiable, provenance=self.provenance,
            feature_names=self.knob_names, output_names=self.out_names,
            assumptions_relied_on=self.assumptions_relied_on,
            extras={"A": None if self.A is None else self.A.tolist(),
                    "baseline": self.baseline_info,
                    "low_confidence_entries": self.low_confidence_entries,
                    "innovation_diagnostics": self.innovation_diag})

    def to_dict(self):
        return self.as_artifact().to_dict()


class FBIdentifier:
    """Component 1. Structured MIMO gain-plus-dynamics estimate."""

    def __init__(self, estimator: str = "svd", ridge_lambda: float = 1.0,
                 identify_dynamics: bool = True,
                 quality_layer: Optional[DataQualityLayer] = None, seed: int = 0):
        self.estimator = estimator
        self.ridge_lambda = ridge_lambda
        self.identify_dynamics = identify_dynamics
        self.dq = quality_layer or DataQualityLayer()
        self.seed = seed

    def identify(self, inline: EventTable, doe: Optional[EventTable] = None,
                 u0: Optional[np.ndarray] = None, doe_age_days: float = 0.0,
                 doe_factor_names: Optional[List[str]] = None,
                 allow_time_varying_u0: bool = False) -> FBModel:
        prov = ProvenanceLog(seed=self.seed)
        prov.config["component"] = "1-FB-identification"
        n_knob, n_out = inline.n_knob, inline.n_out
        knob_names, out_names = inline.knob_cols_used, inline.y_cols
        assumptions = ["A1", "A3"]   # fixed-gain premise; M accuracy for downstream use

        # ---- baseline u0 via the shared service (CF-02, A2/E3) --------
        baseline_info: Dict = {}
        if u0 is None:
            base = estimate_baseline(
                inline.used_knobs(), inline.control_off_mask(),
                times=inline.frame[inline.time_col].to_numpy(dtype=float),
                allow_time_varying=allow_time_varying_u0)
            u0 = base.u0
            baseline_info = base.to_dict()
            prov.note(f"u0 source: {base.source}")
            if base.warning:
                prov.note(base.warning)
            if base.drift_detected:
                assumptions.append("A2")
            prov.assume("u0 valid over analysis horizon", not base.drift_detected,
                        base.warning or "u0 stable across control-off episodes (A2)")
        u0 = np.asarray(u0, dtype=float)
        u0_static = u0 if u0.ndim == 1 else u0.mean(axis=0)
        u0_row = u0 if u0.ndim == 2 else u0_static[None, :]

        # ---- data-quality gate (DQ) -----------------------------------
        inline_rep = self.dq.assess_inline(inline)
        doe_rep: Optional[QualityReport] = None
        if doe is not None:
            doe_rep = self.dq.assess_doe(
                doe.used_knobs() - u0_static, doe_factor_names or doe.knob_cols_used,
                age_days=doe_age_days, inline_operating_point=inline.used_knobs() - u0_static)
            self.dq.combine(doe_rep, inline_rep)
        prov.merge(self.dq.provenance, prefix="dq:")

        # ---- accidental excitation (FB-05, EX-04) ---------------------
        accidental = mine_accidental_excitation(
            inline.recommended_knobs(), inline.used_knobs(),
            control_off=inline.control_off_mask(), knob_names=knob_names)
        prov.note(f"accidental excitation: {int(accidental.override_mask.sum())} override "
                  f"events, {int(accidental.control_off_mask.sum())} control-off events")

        # ---- fused gain-regression data (FB-01) -----------------------
        X_parts, Y_parts, w_parts = [], [], []
        if doe_rep is not None and doe_rep.decision != Verdict.REJECT:
            Xd = doe.used_knobs() - u0_static
            Yd = doe.y_matrix()
            good = np.all(np.isfinite(Yd), axis=1)
            X_parts.append(Xd[good]); Y_parts.append(Yd[good])
            w_parts.append(np.full(good.sum(), doe_rep.weight))
            prov.note(f"DOE contributes {good.sum()} runs at weight {doe_rep.weight:.2f}")
        Xi = inline.used_knobs() - u0_row
        Yi = inline.y_matrix()
        meas = np.all(np.isfinite(Yi), axis=1)
        inline_w = np.where(accidental.override_mask | accidental.control_off_mask, 1.0,
                            inline_rep.weight)
        if meas.any():
            X_parts.append(Xi[meas]); Y_parts.append(Yi[meas]); w_parts.append(inline_w[meas])
            prov.note(f"inline contributes {int(meas.sum())} measured events")
        X, Y, weights = np.vstack(X_parts), np.vstack(Y_parts), np.concatenate(w_parts)

        # ---- regularized gain via the shared sensitivity core ---------
        fit = SensitivityCore(estimator=self.estimator, ridge_lambda=self.ridge_lambda,
                              seed=self.seed).fit(X, Y, feature_names=knob_names,
                                                  output_names=out_names, provenance=prov,
                                                  weights=weights)
        M = check_shape_contract(fit.coef.T, n_out, n_knob, where="identified M")

        # ---- closed-loop identifiability limits (FB-04) ----------------
        low_conf = set(fit.identifiable.low_confidence_features())
        low_confidence_entries = [(on, kn) for kn in knob_names if kn in low_conf
                                  for on in out_names]
        if low_conf:
            prov.note(f"low-confidence M entries (excitation-deficient knobs "
                      f"{sorted(low_conf)}) routed to experiment planner (FB-04)")

        # ---- dynamics (Component-1-only layer) + innovation diagnostics
        cf = reconstruct(np.nan_to_num(Yi), M, inline.used_knobs(), u0)
        A = identify_first_order_dynamics(cf.y_nocontrol[meas], prov) \
            if self.identify_dynamics else None
        innovation_diag = self._innovation_diagnostics(cf.y_nocontrol[meas], out_names,
                                                       fit.sigma2, prov)

        # ---- FB-08 identifiability boundary ---------------------------
        prov.note("FB-08: passive closed-loop data reveals only the PRODUCT of gain error "
                  "and estimator tuning; this artifact does NOT attribute residual to one "
                  "or the other without active excitation or the estimator spec.")
        prov.assume("gain error separable from estimator tuning", False,
                    "structurally unidentifiable from passive data (FB-08)")

        return FBModel(M=M, A=A, param_cov=fit.param_cov, sigma2=fit.sigma2,
                       identifiable=fit.identifiable, innovation_diag=innovation_diag,
                       accidental=accidental, provenance=prov, knob_names=knob_names,
                       out_names=out_names, u0=u0, baseline_info=baseline_info,
                       low_confidence_entries=low_confidence_entries,
                       assumptions_relied_on=assumptions)

    @staticmethod
    def _innovation_diagnostics(disturbance, out_names, sigma2, prov) -> Dict:
        diag = {}
        for j, on in enumerate(out_names):
            series = disturbance[:, j]
            if len(series) < 12:
                continue
            innov = np.diff(series)
            lb = stats.ljung_box(innov, lags=min(10, len(innov) // 3))
            resid_var = float(np.var(innov))
            colored = bool(lb.reject_null)
            diag[on] = {"residual_variance": resid_var, "ljung_box": lb.to_dict(),
                        "colored_residual": colored,
                        "consistent_with_noise_floor": resid_var <= 4.0 * float(sigma2[j])}
            if colored:
                prov.note(f"{on}: colored residual (Ljung-Box p={lb.pvalue:.3f}) => model "
                          f"mis-specification, not mere noise (FB-06)")
        return diag
