"""Component 6 — All-Data Regression Machine (§10, REG-01..09).

Wraps the shared `SensitivityCore` with the Component-6-only regularized
selection layer. Decouples the controller via the counterfactual (with the
A3/E2 phantom-FF guard), picks an estimator by honest forward validation, ranks
features by group-aware stability selection (E12), and routes low-variation
gaps to the experiment planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ...contracts.schema import EventTable
from ...contracts.provenance import ProvenanceLog
from ...core import validation as val
from ...estimation.sensitivity.core import SensitivityCore
from ...estimation.regularized.pls import pls_fit
from ...estimation.regularized.elastic_net import elastic_net_fit
from ...estimation.regularized.stability_selection import stability_selection, StabilityResult
from .decouple import decouple
from .target import innovation_target
from .select import low_variation, feedforward_candidates


@dataclass
class RegressionResult:
    target_kind: str
    estimator: str
    forward_error: float
    fold_errors: List[float]
    importance: StabilityResult
    feedforward_candidates: List[str]
    quarantined_sensors: List[str]
    low_variation_adjustable: List[str]
    observed_only_unidentifiable: List[str]
    concept_drift: Dict
    decoupled: bool
    provenance: ProvenanceLog
    notes: List[str] = field(default_factory=list)

    def to_dict(self):
        d = dict(self.__dict__)
        d["importance"] = self.importance.to_dict()
        d["provenance"] = self.provenance.to_dict()
        return d


def _fit_predict_for(est: str, seed: int):
    def fit_predict(Xtr, ytr, Xte):
        if est == "ridge":
            return SensitivityCore(estimator="ridge", seed=seed).fit(Xtr, ytr).predict(Xte).ravel()
        x_mu, y_mu = Xtr.mean(0), ytr.mean()
        if est == "pls":
            B, _, _ = pls_fit(Xtr, ytr, min(5, Xtr.shape[1]))
            return ((Xte - x_mu) @ B).ravel() + y_mu
        if est == "elastic_net":
            beta = elastic_net_fit(Xtr, ytr, lam=0.1, l1_ratio=0.5)
            sd = Xtr.std(0)
            return ((Xte - x_mu) / np.where(sd > 1e-12, sd, 1.0)) @ beta + y_mu
        raise ValueError(f"unknown estimator {est!r}")
    return fit_predict


class RegressionMachine:
    """Component 6. Broad predictor across all sensors (REG-01, regularized wrapper)."""

    def __init__(self, candidate_estimators: Optional[List[str]] = None,
                 low_variation_quantile: float = 0.05, n_stability_resamples: int = 100,
                 block_size: int = 10, selection_threshold: float = 0.6, seed: int = 0):
        self.candidate_estimators = candidate_estimators or ["ridge", "pls", "elastic_net"]
        self.low_variation_quantile = low_variation_quantile
        self.n_stability_resamples = n_stability_resamples
        self.block_size = block_size
        self.selection_threshold = selection_threshold
        self.seed = seed

    def fit(self, table: EventTable, target_output: str, M: Optional[np.ndarray] = None,
            u0: Optional[np.ndarray] = None, control_present: bool = True,
            target_kind: str = "raw", adjustable_features: Optional[List[str]] = None,
            control_model_error_known: bool = True,
            innovation_series: Optional[np.ndarray] = None,
            rel_uncertainty_M: float = 0.0) -> RegressionResult:
        prov = ProvenanceLog(seed=self.seed)
        prov.config["component"] = "6-regression"
        rng = np.random.default_rng(self.seed)
        notes: List[str] = []
        feat_names = table.regressor_names
        X = table.regressors()
        j_out = table.y_cols.index(target_output)
        Yall = table.y_matrix()

        # ---- target + controller decoupling (REG-02/04, A3/E2) ---------
        quarantined: List[str] = []
        decoupled = False
        if target_kind == "raw" and control_present and control_model_error_known:
            dec = decouple(Yall, M, table.used_knobs(), u0, j_out, ff_matrix=table.ff(),
                           ff_names=table.ff_cols, rel_uncertainty_M=rel_uncertainty_M,
                           provenance=prov)
            decoupled, quarantined, Y = dec.decoupled, dec.quarantined_sensors, dec.target
            if not decoupled:
                target_kind = "innovation"
                notes.append("no M/u0 supplied; modeling the post-control innovation")
                Y = innovation_target(Yall[:, j_out], innovation_series, prov)
        elif target_kind == "raw" and control_present:
            target_kind = "innovation"
            prov.note("control-model error unknown => fall back to innovation-target "
                      "modeling and flag the limitation (REG-02/04)")
            notes.append("decoupling impossible; modeling the post-control innovation")
            Y = innovation_target(Yall[:, j_out], innovation_series, prov)
        elif target_kind == "raw":
            Y = Yall[:, j_out]
        else:
            Y = innovation_target(Yall[:, j_out], innovation_series, prov)

        measured = table.measured_mask(target_output) & np.isfinite(Y)
        Xm, Ym = X[measured], Y[measured]
        if len(Ym) < 8:
            prov.note("too few measured events for honest validation")

        # ---- low-variation handling + planner hand-off (REG-07) -------
        low_adj, obs_only = low_variation(Xm, feat_names,
                                          set(adjustable_features or table.knob_cols_used),
                                          self.low_variation_quantile, prov)

        # ---- estimator selection by forward CV (REG-03/05/08) ---------
        best_est, fold_res = self._select_estimator(Xm, Ym, prov)

        # ---- group-aware stability selection (REG-06/E12) --------------
        importance = stability_selection(Xm, Ym, feat_names,
                                         n_resamples=self.n_stability_resamples,
                                         block_size=self.block_size, rng=rng)
        prov.note(f"stability selection over block-bootstrap resamples (block_size="
                  f"{self.block_size}); per-feature + group-wise frequency (REG-06/E12)")

        # ---- feedforward candidates, excluding quarantined (REG-04) ----
        ff_cands = feedforward_candidates(importance, table.ff_cols, self.selection_threshold,
                                          quarantined)
        prov.note(f"feedforward candidates: {ff_cands}"
                  + (f"; quarantined (phantom FF, A3/E2): {quarantined}" if quarantined else ""))

        # ---- concept drift (VAL-06) -----------------------------------
        drift = val.concept_drift_test(fold_res.horizons, fold_res.fold_errors)
        prov.record_validation("forward_chaining", float(fold_res.mean_error),
                               fold_errors=fold_res.fold_errors.tolist())
        if drift.degrades_with_horizon:
            prov.note("concept drift: forward error grows with horizon => the mapping itself "
                      "is moving; include tool age/state or model per campaign (VAL-06)")

        return RegressionResult(
            target_kind=target_kind, estimator=best_est,
            forward_error=float(fold_res.mean_error),
            fold_errors=fold_res.fold_errors.tolist(), importance=importance,
            feedforward_candidates=ff_cands, quarantined_sensors=quarantined,
            low_variation_adjustable=low_adj, observed_only_unidentifiable=obs_only,
            concept_drift=drift.to_dict(), decoupled=decoupled, provenance=prov, notes=notes)

    def _select_estimator(self, X, y, prov):
        best_est, best_err, best_fold = None, np.inf, None
        for est in self.candidate_estimators:
            fold = val.forward_validate(X, y, _fit_predict_for(est, self.seed), n_splits=4,
                                        purge=1)
            if np.isfinite(fold.mean_error) and fold.mean_error < best_err:
                best_est, best_err, best_fold = est, fold.mean_error, fold
        if best_est is None:
            best_est = "mean"
            best_fold = val.forward_validate(X, y, lambda a, b, c: np.full(len(c), np.nanmean(b)),
                                             n_splits=4)
        prov.note(f"selected estimator '{best_est}' by forward-chaining CV (forward error "
                  f"{best_fold.mean_error:.4f}); high-capacity nonlinear models are NOT used "
                  f"by default at this data scale (REG-08)")
        return best_est, best_fold


__all__ = ["RegressionMachine", "RegressionResult"]
