"""Component 1 — FB / Process-Dynamic Model Identification (§5, FB-01..08).

Wraps the shared `SensitivityCore` (gain) with the Component-1-only dynamics
layer. Fuses DOE and inline under the joint-use contract, sets order by the
singular-value noise gap, reports identifiability limits, mines accidental
excitation, sources ``u0`` through the baseline service (A2/E3), and refuses to
emit attributions the data cannot support (FB-08).

Gain rows must be exogenous (M3). Closed-loop inline rows, where the knob
settings respond to past outputs, are excluded from the gain regression and used
only for ``u0``, noise and drift (DQ-06). They are used for the gain only as a
flagged fallback when no usable DOE exists, and the reported uncertainty then
carries an explicit bias floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ...contracts.schema import EventTable
from ...contracts.shapes import check_shape_contract
from ...contracts.provenance import ProvenanceLog, Verdict
from ...contracts.artifact import (ModelArtifact, variance_rel_uncertainty,
                                   expected_frobenius_error2)
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
    bias_rel_uncertainty: float = 0.0     # KI-2: bias term added to the variance term
    gain_source: str = ""
    FF_gain: Optional[np.ndarray] = None  # (n_out, n_ff) from the counterfactual (KI-5)
    ff_names: List[str] = field(default_factory=list)
    ff_rel_uncertainty: float = float("nan")

    @property
    def n_out(self) -> int:
        return self.M.shape[0]

    @property
    def n_knob(self) -> int:
        return self.M.shape[1]

    def u0_static(self) -> np.ndarray:
        return self.u0 if np.ndim(self.u0) == 1 else np.mean(self.u0, axis=0)

    def variance_rel_uncertainty(self) -> float:
        """Expected relative Frobenius error from estimation variance alone."""
        return variance_rel_uncertainty(self.param_cov, self.sigma2, self.M)

    def relative_uncertainty(self) -> float:
        """Relative uncertainty on ``M``: variance and bias terms in quadrature (KI-2).

        Defined as an expected ``||M_hat - M||_F / ||M||``, the same scale the
        simulator and counterfactual bands use for ``rel_uncertainty_M``.
        """
        return float(np.hypot(self.variance_rel_uncertainty(), self.bias_rel_uncertainty))

    def as_artifact(self) -> ModelArtifact:
        return ModelArtifact(
            kind="fb", coef=self.M, param_cov=self.param_cov, sigma2=self.sigma2,
            identifiable=self.identifiable, provenance=self.provenance,
            feature_names=self.knob_names, output_names=self.out_names,
            assumptions_relied_on=self.assumptions_relied_on,
            bias_rel_uncertainty=self.bias_rel_uncertainty,
            extras={"A": None if self.A is None else self.A.tolist(),
                    "FF_gain": None if self.FF_gain is None else self.FF_gain.tolist(),
                    "gain_source": self.gain_source,
                    "baseline": self.baseline_info,
                    "low_confidence_entries": self.low_confidence_entries,
                    "innovation_diagnostics": self.innovation_diag})

    def to_dict(self):
        return self.as_artifact().to_dict()


class FBIdentifier:
    """Component 1. Structured MIMO gain-plus-dynamics estimate."""

    def __init__(self, estimator: str = "svd", ridge_lambda: float = 1.0,
                 identify_dynamics: bool = True,
                 quality_layer: Optional[DataQualityLayer] = None,
                 endogenous_bias_floor: float = 1.0, ff_ridge_lambda: float = 1.0,
                 covariance_cleaning: Optional[str] = None, seed: int = 0):
        self.estimator = estimator
        self.ridge_lambda = ridge_lambda
        self.identify_dynamics = identify_dynamics
        self.dq = quality_layer or DataQualityLayer()
        # relative bias assumed when the gain must come from endogenous closed-loop
        # rows: the bias is unknowable from those rows, so downstream bands are wide
        self.endogenous_bias_floor = endogenous_bias_floor
        self.ff_ridge_lambda = ff_ridge_lambda
        self.covariance_cleaning = covariance_cleaning   # NF-05, ridge estimator only
        self.seed = seed

    def _core(self) -> SensitivityCore:
        return SensitivityCore(estimator=self.estimator, ridge_lambda=self.ridge_lambda,
                               covariance_cleaning=self.covariance_cleaning, seed=self.seed)

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

        # ---- gain-regression rows (FB-01, DQ-06/07, M3) -----------------
        doe_rows = None
        if doe_rep is not None and doe_rep.gain_usable:
            Xd = doe.used_knobs() - u0_static
            Yd = doe.y_matrix()
            good = np.all(np.isfinite(Yd), axis=1)
            if good.sum() >= n_knob + 2:
                doe_rows = (Xd[good], Yd[good], np.full(good.sum(), doe_rep.weight))
                prov.note(f"DOE contributes {good.sum()} runs at weight {doe_rep.weight:.2f}")
            else:
                prov.note(f"DOE has only {good.sum()} complete runs (< n_knob+2); not used "
                          f"for gain")
        Xi = inline.used_knobs() - u0_row
        Yi = inline.y_matrix()
        meas = np.all(np.isfinite(Yi), axis=1)
        inline_rows = None
        endogenous_fallback = False
        if inline_rep.gain_usable and meas.any():
            inline_w = np.where(accidental.override_mask | accidental.control_off_mask, 1.0,
                                inline_rep.weight)
            inline_rows = (Xi[meas], Yi[meas], inline_w[meas])
            prov.note(f"inline contributes {int(meas.sum())} measured events (no feedback "
                      f"detected; weight {inline_rep.weight:.2f}, overrides/control-off 1.0)")
        elif meas.any() and doe_rows is None:
            endogenous_fallback = True
            inline_rows = (Xi[meas], Yi[meas], np.ones(int(meas.sum())))
            prov.note("NO usable DOE: gain falls back to closed-loop inline rows whose knob "
                      "settings respond to past outputs. The estimate is biased by an unknown "
                      f"amount; relative bias floor {self.endogenous_bias_floor:.2f} applied "
                      "and every M entry routed to the experiment planner (DQ-06, M3)")
        elif meas.any():
            prov.note(f"closed-loop inline rows ({int(meas.sum())} measured events) EXCLUDED "
                      "from gain identification: knob settings respond to past outputs, so "
                      "the rows are endogenous. Inline data is used for u0, noise and drift "
                      "only (DQ-06, M3)")
        prov.assume("gain-identification rows exogenous (M3)", not endogenous_fallback,
                    "closed-loop inline rows used as a fallback" if endogenous_fallback
                    else "DOE and/or open-loop inline rows only")
        parts = [r for r in (doe_rows, inline_rows) if r is not None]
        if not parts:
            raise ValueError("no measured data usable for gain identification")
        X = np.vstack([p[0] for p in parts])
        Y = np.vstack([p[1] for p in parts])
        weights = np.concatenate([p[2] for p in parts])
        gain_source = ("DOE + open-loop inline" if doe_rows is not None and inline_rows is not None
                       else "DOE only" if doe_rows is not None
                       else "closed-loop inline (endogenous fallback)" if endogenous_fallback
                       else "open-loop inline only")

        # ---- regularized gain via the shared sensitivity core ---------
        fit = self._core().fit(X, Y, feature_names=knob_names, output_names=out_names,
                               provenance=prov, weights=weights)
        M = check_shape_contract(fit.coef.T, n_out, n_knob, where="identified M")

        # ---- bias term for the uncertainty (KI-2) ----------------------
        bias_rel = 0.0
        if endogenous_fallback:
            bias_rel = float(self.endogenous_bias_floor)
        elif doe_rows is not None and inline_rows is not None:
            fd = self._core().fit(doe_rows[0], doe_rows[1], weights=doe_rows[2])
            fi = self._core().fit(inline_rows[0], inline_rows[1], weights=inline_rows[2])
            diff2 = float(np.sum((fd.coef - fi.coef) ** 2))
            var2 = sum(expected_frobenius_error2(f.param_cov, f.sigma2) for f in (fd, fi))
            bias_rel = float(np.sqrt(max(diff2 - var2, 0.0)) / (np.linalg.norm(M) + 1e-12))
            prov.note(f"DOE vs inline gain consistency: excess discrepancy => relative bias "
                      f"term {bias_rel:.3f} (KI-2)")
        var_rel = variance_rel_uncertainty(fit.param_cov, fit.sigma2, M)
        prov.note(f"gain source: {gain_source}; relative uncertainty: variance "
                  f"{var_rel:.3f}, bias {bias_rel:.3f}")

        # ---- closed-loop identifiability limits (FB-04) ----------------
        low_conf = set(fit.identifiable.low_confidence_features())
        if endogenous_fallback:
            low_conf = set(knob_names)
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

        # ---- measured-disturbance (FF) gains from the counterfactual (KI-5)
        FF_gain, ff_rel = None, float("nan")
        ff = inline.ff()
        ff_ok = meas & np.all(np.isfinite(ff), axis=1) if ff.shape[1] else meas & False
        if ff.shape[1] and ff_ok.sum() > ff.shape[1] + 5:
            from ...estimation.sensitivity.core import ridge_fit
            F = ff[ff_ok] - ff[ff_ok].mean(0)
            D = cf.y_nocontrol[ff_ok] - cf.y_nocontrol[ff_ok].mean(0)
            B, cov_ff, s2_ff = ridge_fit(F, D, self.ff_ridge_lambda)
            FF_gain = check_shape_contract(B.T, n_out, ff.shape[1], where="identified FF gain")
            ff_rel = variance_rel_uncertainty(cov_ff, s2_ff, FF_gain)
            prov.note(f"FF gains estimated from the counterfactual disturbance on "
                      f"{int(ff_ok.sum())} events (ridge lambda={self.ff_ridge_lambda}); "
                      f"relative uncertainty {ff_rel:.2f}")

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
                       assumptions_relied_on=assumptions, bias_rel_uncertainty=bias_rel,
                       gain_source=gain_source, FF_gain=FF_gain, ff_names=list(inline.ff_cols),
                       ff_rel_uncertainty=ff_rel)

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
