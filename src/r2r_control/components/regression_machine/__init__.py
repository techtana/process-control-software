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
from .select import low_variation, feedforward_candidates, marginal_pvalues
from .traces import trace_features, resample_traces, raw_retention_check
from ...core.validation.control_off import control_off_gold


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
    gold: Dict = field(default_factory=dict)          # VAL-04 control-off holdout
    traces: Dict = field(default_factory=dict)        # REG-09 representation decisions
    ff_importance: Optional[StabilityResult] = None   # FF-only screen behind the candidates

    def to_dict(self):
        d = dict(self.__dict__)
        d["importance"] = self.importance.to_dict()
        d["ff_importance"] = None if self.ff_importance is None else self.ff_importance.to_dict()
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
                 block_size: int = 10, selection_threshold: float = 0.6,
                 purge: int = 1, embargo: int = 0, gold_holdout: bool = True,
                 trace_substeps: int = 1, seed: int = 0):
        self.candidate_estimators = candidate_estimators or ["ridge", "pls", "elastic_net"]
        self.low_variation_quantile = low_variation_quantile
        self.n_stability_resamples = n_stability_resamples
        self.block_size = block_size
        self.selection_threshold = selection_threshold
        self.purge = purge                  # VAL-03 gap between train and test folds
        self.embargo = embargo              # VAL-03 train points dropped around test folds
        self.gold_holdout = gold_holdout    # VAL-04 control-off episodes held out as gold
        self.trace_substeps = trace_substeps
        self.seed = seed

    def fit(self, table: EventTable, target_output: str, M: Optional[np.ndarray] = None,
            u0: Optional[np.ndarray] = None, control_present: bool = True,
            target_kind: str = "raw", adjustable_features: Optional[List[str]] = None,
            control_model_error_known: bool = True,
            innovation_series: Optional[np.ndarray] = None,
            rel_uncertainty_M: float = 0.0,
            tool_state_cols: Optional[List[str]] = None) -> RegressionResult:
        prov = ProvenanceLog(seed=self.seed)
        prov.config["component"] = "6-regression"
        rng = np.random.default_rng(self.seed)
        notes: List[str] = []
        feat_names = list(table.regressor_names)
        X = table.regressors()
        j_out = table.y_cols.index(target_output)
        Yall = table.y_matrix()

        # ---- traces: aggregated features by default (REG-09) ----------
        trace_info: Dict = {}
        trace_parts = []
        for col in table.trace_cols:
            cells = table.frame[col].to_list()
            F, names = trace_features(cells, col, self.trace_substeps)
            trace_parts.append((col, cells, F, names))
            X = np.column_stack([X, F])
            feat_names += names
        if trace_parts:
            prov.note(f"traces {table.trace_cols} aggregated per substep "
                      f"(substeps={self.trace_substeps}) before fitting (REG-09)")

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

        # ---- raw-trace retention check, now that the target is known (REG-09)
        for col, cells, F, names in trace_parts:
            raw = resample_traces(cells)
            chk = raw_retention_check(raw, F, Y)
            trace_info[col] = chk
            prov.note(f"trace {col}: {chk['detail']} (held-out |r| raw="
                      f"{chk['raw_corr']:.2f}, best aggregate={chk['agg_corr']:.2f})")
            if chk["retain_raw"]:
                X = np.column_stack([X, raw])
                feat_names += [f"{col}_raw{i}" for i in range(raw.shape[1])]

        measured = table.measured_mask(target_output) & np.isfinite(Y)
        measured &= np.all(np.isfinite(X), axis=1)

        # ---- control-off gold holdout, disjoint from fitting (VAL-04, A5) --
        gold_mask = np.zeros(len(Y), dtype=bool)
        gold_rep = None
        if self.gold_holdout and table.control_off_mask().any():
            ts = (table.frame[tool_state_cols].to_numpy(dtype=float)
                  if tool_state_cols else None)
            gold_rep = control_off_gold(table.control_off_mask(), tool_state=ts)
            gold_mask[gold_rep.gold_idx] = True
            prov.note(f"control-off gold holdout: {gold_rep.n_gold_episodes} episode(s), "
                      f"{int((gold_mask & measured).sum())} measured events set aside; "
                      f"{len(gold_rep.excluded_episodes)} excluded as unrepresentative "
                      f"(VAL-04, A5)")
        fit_rows = measured & ~gold_mask
        Xm, Ym = X[fit_rows], Y[fit_rows]
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
        # Under control the knob moves encode the FF inputs the controller already
        # uses, so selecting sensors alongside the knobs picks sensors that merely
        # cancel that part of the knob signal (suppressors). Screen FF on its own,
        # and require Bonferroni-corrected significance (REG-04).
        ff_importance = importance
        ff_idx = [feat_names.index(c) for c in table.ff_cols]
        if control_present and table.ff_cols:
            ff_importance = stability_selection(Xm[:, ff_idx], Ym, list(table.ff_cols),
                                                n_resamples=self.n_stability_resamples,
                                                block_size=self.block_size, rng=rng)
            prov.note("feedforward screening on FF sensors only: knob moves under control "
                      "carry the FF inputs and would act as suppressors (REG-04)")
        ff_p = marginal_pvalues(Xm[:, ff_idx], Ym, list(table.ff_cols))
        ff_cands = feedforward_candidates(ff_importance, table.ff_cols,
                                          self.selection_threshold, quarantined, ff_p)
        prov.note(f"feedforward candidates: {ff_cands}"
                  + (f"; quarantined (phantom FF, A3/E2): {quarantined}" if quarantined else ""))

        # ---- gold evaluation of the selected estimator (VAL-04) --------
        gold: Dict = {}
        gold_rows = measured & gold_mask
        if gold_rep is not None:
            gold = gold_rep.to_dict()
            if gold_rows.sum() >= 3 and best_est != "mean" and len(Ym) > Xm.shape[1]:
                pred = _fit_predict_for(best_est, self.seed)(Xm, Ym, X[gold_rows])
                err = float(np.sqrt(np.nanmean((Y[gold_rows] - pred) ** 2)))
                gold["gold_error"] = err
                prov.note(f"gold (control-off) RMSE of '{best_est}': {err:.4f} vs forward "
                          f"error {fold_res.mean_error:.4f} (VAL-04)")

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
            concept_drift=drift.to_dict(), decoupled=decoupled, provenance=prov, notes=notes,
            gold=gold, traces=trace_info, ff_importance=ff_importance)

    def _select_estimator(self, X, y, prov):
        best_est, best_err, best_fold = None, np.inf, None
        for est in self.candidate_estimators:
            fold = val.forward_validate(X, y, _fit_predict_for(est, self.seed), n_splits=4,
                                        purge=self.purge, embargo=self.embargo)
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
