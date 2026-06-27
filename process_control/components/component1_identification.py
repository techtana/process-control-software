"""Component 1 — FB / Process-Dynamic Model Identification (§5, FB-01..08).

Identify the MIMO process gain ``M`` (and first-order dynamics for the
state-space MPC) from partial-factorial DOE and noisy inline data, while
honestly representing what the data can and cannot identify.  This is the
``mode='fb'`` configuration of the shared estimation engine (§REG-01); it fuses
DOE (wide-range gain/curvature) and inline (operating-point-local correction,
noise) under the joint-use contract, sets model order by the singular-value
noise gap, reports closed-loop identifiability limits explicitly, mines
accidental excitation, and refuses to emit attributions the data cannot support
(FB-08).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..data.schema import EventTable, check_shape_contract
from ..data.quality import DataQualityLayer, QualityReport
from ..provenance import ProvenanceLog, Verdict
from ..foundations import stats
from ..foundations.engine import EstimationEngine, EngineConfig, FittedModel
from ..foundations.excitation import mine_accidental_excitation, AccidentalExcitation


@dataclass
class FBModel:
    """The identified FB model artifact published to the rest of the system (§IF-01)."""

    M: np.ndarray                       # process gain (n_out x n_knob)
    A: Optional[np.ndarray]             # first-order dynamics (n_out x n_out) or None
    param_cov: np.ndarray               # parameter-uncertainty covariance (FB-07)
    sigma2: np.ndarray                  # per-output one-step residual variance
    identifiable: object                # IdentifiableDirectionReport
    innovation_diag: Dict               # FB-06 diagnostics
    accidental: AccidentalExcitation
    provenance: ProvenanceLog
    knob_names: List[str]
    out_names: List[str]
    low_confidence_entries: List[tuple] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def n_out(self) -> int:
        return self.M.shape[0]

    @property
    def n_knob(self) -> int:
        return self.M.shape[1]

    def relative_uncertainty(self) -> float:
        scale = np.linalg.norm(self.M) + 1e-12
        return float(np.sqrt(np.trace(self.param_cov) * np.mean(self.sigma2)) / scale)

    def to_dict(self):
        return {
            "M": self.M.tolist(),
            "A": None if self.A is None else self.A.tolist(),
            "sigma2": self.sigma2.tolist(),
            "relative_uncertainty": self.relative_uncertainty(),
            "identifiable": self.identifiable.to_dict(),
            "innovation_diag": self.innovation_diag,
            "accidental_excitation": self.accidental.to_dict(),
            "low_confidence_entries": self.low_confidence_entries,
            "provenance": self.provenance.to_dict(),
            "notes": self.notes,
        }


class FBIdentifier:
    """Component 1.  Structured, control-oriented MIMO gain-plus-dynamics estimate."""

    def __init__(
        self,
        estimator: str = "svd",
        ridge_lambda: float = 1.0,
        identify_dynamics: bool = True,
        quality_layer: Optional[DataQualityLayer] = None,
        seed: int = 0,
    ):
        self.estimator = estimator
        self.ridge_lambda = ridge_lambda
        self.identify_dynamics = identify_dynamics
        self.dq = quality_layer or DataQualityLayer()
        self.seed = seed

    # ------------------------------------------------------------------ #
    def identify(
        self,
        inline: EventTable,
        doe: Optional[EventTable] = None,
        u0: Optional[np.ndarray] = None,
        doe_age_days: float = 0.0,
        doe_factor_names: Optional[List[str]] = None,
    ) -> FBModel:
        prov = ProvenanceLog(seed=self.seed)
        prov.config["component"] = "1-FB-identification"
        n_knob = inline.n_knob
        n_out = inline.n_out
        knob_names = inline.knob_cols_used
        out_names = inline.y_cols
        if u0 is None:
            # baseline = control-off knob mean if available, else global mean (CF-02)
            off = inline.control_off_mask()
            U = inline.used_knobs()
            if off.any():
                u0 = np.nanmean(U[off], axis=0)
                prov.note("u0 from control-off episodes (closer to true open-loop baseline)")
            else:
                u0 = np.nanmean(U, axis=0)
                prov.note("u0 from global knob mean; confirm it is the true open-loop "
                          "baseline (CF-02, open design decision)")

        # ---- data-quality gate (DQ) -----------------------------------
        inline_rep = self.dq.assess_inline(inline)
        prov.merge(self.dq.provenance, prefix="dq:")
        doe_rep: Optional[QualityReport] = None
        if doe is not None:
            doe_rep = self.dq.assess_doe(
                doe.used_knobs() - u0,
                doe_factor_names or doe.knob_cols_used,
                age_days=doe_age_days,
                inline_operating_point=inline.used_knobs() - u0,
            )
            prov.merge(self.dq.provenance, prefix="dq:")
            self.dq.combine(doe_rep, inline_rep)

        # ---- accidental excitation (FB-05, EX-04) ---------------------
        accidental = mine_accidental_excitation(
            inline.recommended_knobs(), inline.used_knobs(),
            control_off=inline.control_off_mask(), knob_names=knob_names)
        prov.note(f"accidental excitation: {int(accidental.override_mask.sum())} override "
                  f"events, {int(accidental.control_off_mask.sum())} control-off events")

        # ---- assemble fused gain-regression data (FB-01) --------------
        X_parts, Y_parts, w_parts, provs = [], [], [], []
        # DOE supplies wide-range gain (if not rejected)
        if doe is not None and doe_rep is not None and doe_rep.decision != Verdict.REJECT:
            Xd = doe.used_knobs() - u0
            Yd = doe.y_matrix()
            good = np.all(np.isfinite(Yd), axis=1)
            X_parts.append(Xd[good])
            Y_parts.append(Yd[good])
            w_parts.append(np.full(good.sum(), doe_rep.weight))
            prov.note(f"DOE contributes {good.sum()} runs at weight {doe_rep.weight:.2f} "
                      f"(wide-range gain)")
        # inline supplies operating-point-local correction + accidental excitation
        Xi = inline.used_knobs() - u0
        Yi = inline.y_matrix()
        meas = np.all(inline.measured_mask(), axis=1) if n_out > 1 else inline.measured_mask()
        # weight: emphasize accidental-excitation events where inline is informative
        inline_w = np.where(accidental.override_mask | accidental.control_off_mask, 1.0,
                            inline_rep.weight)
        good_i = meas & np.all(np.isfinite(Yi), axis=1)
        if good_i.any():
            X_parts.append(Xi[good_i])
            Y_parts.append(Yi[good_i])
            w_parts.append(inline_w[good_i])
            prov.note(f"inline contributes {int(good_i.sum())} measured events "
                      f"(local correction + accidental excitation)")

        X = np.vstack(X_parts)
        Y = np.vstack(Y_parts)
        weights = np.concatenate(w_parts)

        # ---- regularized estimation via the shared engine (FB-02/03) ---
        cfg = EngineConfig(mode="fb", estimator=self.estimator,
                           ridge_lambda=self.ridge_lambda, seed=self.seed)
        engine = EstimationEngine(cfg)
        fit: FittedModel = engine.fit(X, Y, feature_names=knob_names,
                                      output_names=out_names, provenance=prov,
                                      weights=weights)
        M = check_shape_contract(fit.coef.T, n_out, n_knob, where="identified M")

        # ---- closed-loop identifiability limits (FB-04) ----------------
        low_conf_features = set(fit.identifiable.low_confidence_features())
        low_confidence_entries = []
        if low_conf_features:
            for j, kn in enumerate(knob_names):
                if kn in low_conf_features:
                    for i, on in enumerate(out_names):
                        low_confidence_entries.append((on, kn))
            prov.note(f"low-confidence M entries (excitation-deficient knobs "
                      f"{sorted(low_conf_features)}) routed to experiment planner (FB-04)")

        # ---- dynamics (first-order A) ---------------------------------
        A = None
        if self.identify_dynamics:
            A = self._identify_dynamics(inline, M, u0, prov)

        # ---- innovation / residual diagnostics (FB-06) ----------------
        innovation_diag = self._innovation_diagnostics(
            inline, M, u0, fit.sigma2, prov)

        # ---- FB-08 identifiability boundary ---------------------------
        prov.note(
            "FB-08: passive closed-loop data reveals only the PRODUCT of gain error "
            "and estimator tuning; this artifact does NOT attribute residual to one "
            "or the other without active excitation or the estimator spec.")
        prov.assume("gain error separable from estimator tuning", False,
                    "structurally unidentifiable from passive data (FB-08)")

        return FBModel(
            M=M, A=A, param_cov=fit.param_cov, sigma2=fit.sigma2,
            identifiable=fit.identifiable, innovation_diag=innovation_diag,
            accidental=accidental, provenance=prov, knob_names=knob_names,
            out_names=out_names, low_confidence_entries=low_confidence_entries,
        )

    # ------------------------------------------------------------------ #
    def _identify_dynamics(self, inline, M, u0, prov):
        """Crude first-order A from the counterfactual disturbance autocorrelation."""
        from ..foundations.counterfactual import reconstruct
        Y = inline.y_matrix()
        U = inline.used_knobs()
        meas = np.all(np.isfinite(Y), axis=1)
        if meas.sum() < 5:
            prov.note("too few measured events for dynamics; A omitted")
            return None
        cf = reconstruct(np.nan_to_num(Y), M, U, u0)
        d = cf.y_nocontrol[meas]
        if len(d) < 3:
            return None
        D0, D1 = d[:-1], d[1:]
        A_T, *_ = np.linalg.lstsq(D0, D1, rcond=None)
        A = A_T.T
        # keep stable
        eig = np.linalg.eigvals(A)
        if np.max(np.abs(eig)) >= 1.5:
            prov.note(f"identified A unstable (max|eig|={np.max(np.abs(eig)):.2f}); "
                      f"clipping to diagonal AR estimate")
            A = np.diag(np.clip(np.diag(A), -0.99, 0.99))
        return A

    def _innovation_diagnostics(self, inline, M, u0, sigma2, prov):
        """One-step residual variance, whiteness, and noise-floor consistency (FB-06)."""
        from ..foundations.counterfactual import reconstruct
        Y = inline.y_matrix()
        U = inline.used_knobs()
        meas = np.all(np.isfinite(Y), axis=1)
        cf = reconstruct(np.nan_to_num(Y), M, U, u0)
        d = cf.y_nocontrol
        diag = {}
        for j, on in enumerate(inline.y_cols):
            series = d[meas, j]
            if len(series) < 12:
                continue
            innov = np.diff(series)  # one-step innovation proxy on disturbance
            lb = stats.ljung_box(innov, lags=min(10, len(innov) // 3))
            resid_var = float(np.var(innov))
            colored = bool(lb.reject_null) if lb.reject_null is not None else False
            diag[on] = {
                "residual_variance": resid_var,
                "ljung_box": lb.to_dict(),
                "colored_residual": colored,
                "consistent_with_noise_floor": resid_var <= 4.0 * float(sigma2[j]),
            }
            if colored:
                prov.note(f"{on}: colored residual (Ljung-Box p="
                          f"{lb.pvalue:.3f}) => model mis-specification, not mere noise "
                          f"(FB-06)")
        return diag
