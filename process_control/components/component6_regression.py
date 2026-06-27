"""Component 6 — All-Data Regression Machine (§10, REG-01..09).

Build a predictor of post-processing measurements from **all** available data
(knobs, tool sensors, pre-processing measurements, process configuration),
whether collected with or without control, decoupling the controller's
contribution when control is present, and routing low-variation gaps to the
experiment planner.

This is the ``mode='regression'`` configuration of the *same* estimation engine
as the FB identifier (§REG-01): it shares the data model, the data-quality
layer, the excitation analysis, the noise-floor estimator, the counterfactual
service, and the validation framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..data.schema import EventTable
from ..provenance import ProvenanceLog
from ..foundations import validation as val
from ..foundations.counterfactual import reconstruct
from ..foundations.engine import (EstimationEngine, EngineConfig,
                                  elastic_net_fit)
from ..foundations.excitation import regressor_spectrum


@dataclass
class FeatureImportance:
    """Stability-selection feature importance (REG-06)."""

    names: List[str]
    selection_frequency: np.ndarray   # fraction of block-bootstrap resamples selected
    mean_coef: np.ndarray

    def ranked(self, top: Optional[int] = None):
        order = np.argsort(self.selection_frequency)[::-1]
        if top:
            order = order[:top]
        return [(self.names[i], float(self.selection_frequency[i]),
                 float(self.mean_coef[i])) for i in order]

    def to_dict(self):
        return {
            "names": self.names,
            "selection_frequency": self.selection_frequency.tolist(),
            "mean_coef": self.mean_coef.tolist(),
            "ranked": self.ranked(),
        }


@dataclass
class RegressionResult:
    target_kind: str                  # 'raw' | 'innovation'
    estimator: str
    forward_error: float
    fold_errors: List[float]
    importance: FeatureImportance
    feedforward_candidates: List[str]
    low_variation_adjustable: List[str]   # routed to planner (REG-07)
    observed_only_unidentifiable: List[str]
    concept_drift: Dict
    decoupled: bool
    provenance: ProvenanceLog
    notes: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "target_kind": self.target_kind,
            "estimator": self.estimator,
            "forward_error": float(self.forward_error),
            "fold_errors": self.fold_errors,
            "importance": self.importance.to_dict(),
            "feedforward_candidates": self.feedforward_candidates,
            "low_variation_adjustable": self.low_variation_adjustable,
            "observed_only_unidentifiable": self.observed_only_unidentifiable,
            "concept_drift": self.concept_drift,
            "decoupled": self.decoupled,
            "provenance": self.provenance.to_dict(),
            "notes": self.notes,
        }


class RegressionMachine:
    """Component 6.  Broad predictor across all sensors (REG-01 engine, mode='regression')."""

    def __init__(
        self,
        candidate_estimators: Optional[List[str]] = None,
        low_variation_quantile: float = 0.05,
        n_stability_resamples: int = 100,
        block_size: int = 10,
        selection_threshold: float = 0.6,
        seed: int = 0,
    ):
        self.candidate_estimators = candidate_estimators or ["ridge", "pls", "elastic_net"]
        self.low_variation_quantile = low_variation_quantile
        self.n_stability_resamples = n_stability_resamples
        self.block_size = block_size
        self.selection_threshold = selection_threshold
        self.seed = seed

    # ------------------------------------------------------------------ #
    def fit(
        self,
        table: EventTable,
        target_output: str,
        M: Optional[np.ndarray] = None,
        u0: Optional[np.ndarray] = None,
        control_present: bool = True,
        target_kind: str = "raw",
        adjustable_features: Optional[List[str]] = None,
        control_model_error_known: bool = True,
        innovation_series: Optional[np.ndarray] = None,
    ) -> RegressionResult:
        prov = ProvenanceLog(seed=self.seed)
        prov.config["component"] = "6-regression"
        rng = np.random.default_rng(self.seed)
        notes: List[str] = []

        feat_names = table.regressor_names
        X = table.regressors()
        j_out = table.y_cols.index(target_output)
        Y = table.y_matrix()[:, j_out]

        # ---- controller decoupling (REG-02) ---------------------------
        decoupled = False
        if control_present and target_kind == "raw":
            if M is not None and u0 is not None and control_model_error_known:
                cf = reconstruct(np.nan_to_num(table.y_matrix()),
                                 M, table.used_knobs(), u0)
                Y = cf.y_nocontrol[:, j_out]
                decoupled = True
                prov.note("decoupled control contribution via counterfactual "
                          "subtraction M@(u_used-u0) (REG-02)")
            else:
                # control model error purely unknown => fall back to innovation
                target_kind = "innovation"
                prov.note("control-model error unknown => fall back to "
                          "innovation-target modeling and flag limitation (REG-02/04)")
                notes.append("decoupling impossible; modeling post-control innovation")
        if target_kind == "innovation":
            Y = self._innovation_target(table, j_out, innovation_series, prov)

        # restrict to measured events
        measured = table.measured_mask(target_output) & np.isfinite(Y)
        Xm = X[measured]
        Ym = Y[measured]
        if len(Ym) < 8:
            prov.note("too few measured events for honest validation")

        # ---- low-variation handling + planner hand-off (REG-07) -------
        adjustable = set(adjustable_features or table.knob_cols_used)
        low_adj, obs_only = self._low_variation(Xm, feat_names, adjustable, prov)

        # ---- estimator selection by forward CV (REG-03/05) ------------
        best_est, fold_res = self._select_estimator(Xm, Ym, feat_names, prov)

        # ---- final fit on the chosen engine config --------------------
        cfg = EngineConfig(mode="regression", estimator=best_est, seed=self.seed,
                           n_components=min(5, Xm.shape[1]))
        engine = EstimationEngine(cfg)
        fit = engine.fit(Xm, Ym, feature_names=feat_names, provenance=prov)

        # ---- defensible feature importance via stability selection ----
        importance = self._stability_selection(Xm, Ym, feat_names, rng, prov)

        # ---- feedforward candidates (REG-04, = DIAG-01 object) --------
        ff_candidates = self._feedforward_candidates(
            importance, table, adjustable, target_kind, decoupled)
        prov.note(f"feedforward candidates (sensors explaining the target/innovation): "
                  f"{ff_candidates}")

        # ---- concept-drift check across horizon (VAL-06) --------------
        drift = val.concept_drift_test(
            np.asarray(fold_res.horizons, dtype=float), fold_res.fold_errors)
        prov.record_validation("forward_chaining", float(fold_res.mean_error),
                               fold_errors=fold_res.fold_errors.tolist())
        if drift.degrades_with_horizon:
            prov.note("concept drift: forward error degrades with horizon => the "
                      "mapping itself is moving; include tool-age/state or model "
                      "per-campaign (VAL-06)")

        return RegressionResult(
            target_kind=target_kind,
            estimator=best_est,
            forward_error=float(fold_res.mean_error),
            fold_errors=fold_res.fold_errors.tolist(),
            importance=importance,
            feedforward_candidates=ff_candidates,
            low_variation_adjustable=low_adj,
            observed_only_unidentifiable=obs_only,
            concept_drift=drift.to_dict(),
            decoupled=decoupled,
            provenance=prov,
            notes=notes,
        )

    # ------------------------------------------------------------------ #
    def _innovation_target(self, table, j_out, innovation_series, prov):
        """Post-control innovation/residual target (REG-04)."""
        if innovation_series is not None:
            prov.note("using provided post-control innovation sequence as target (REG-04)")
            return np.asarray(innovation_series, dtype=float)
        # proxy: one-step difference of the raw output (closer to stationary)
        Y = table.y_matrix()[:, j_out]
        innov = np.full_like(Y, np.nan)
        innov[1:] = Y[1:] - Y[:-1]
        prov.note("innovation target proxied by one-step output difference "
                  "(closer to stationary; softens temporal validation, REG-04)")
        return innov

    def _low_variation(self, X, names, adjustable, prov):
        """Detect low-variation inputs; split adjustable (=> planner) vs observed-only."""
        if len(X) < 3:
            return [], []
        sd = np.nanstd(X, axis=0)
        scale = np.nanmedian(sd[sd > 0]) if np.any(sd > 0) else 1.0
        thresh = self.low_variation_quantile * scale
        low_adj, obs_only = [], []
        for j, nm in enumerate(names):
            if sd[j] <= thresh:
                if nm in adjustable:
                    low_adj.append(nm)
                else:
                    obs_only.append(nm)
        if low_adj:
            prov.note(f"low-variation ADJUSTABLE inputs {low_adj} => route to "
                      f"experiment planner (REG-07)")
        if obs_only:
            prov.note(f"low-variation NON-ADJUSTABLE sensors {obs_only} => "
                      f"observed-only, unidentifiable-by-data, no experiment proposed "
                      f"(REG-07)")
        return low_adj, obs_only

    def _select_estimator(self, X, y, names, prov):
        """Forward-chaining CV to pick ridge / pls / elastic_net (REG-03/05/08)."""
        best_est, best_err, best_fold = None, np.inf, None
        for est in self.candidate_estimators:
            def fit_predict(Xtr, ytr, Xte, est=est):
                cfg = EngineConfig(mode="regression", estimator=est, seed=self.seed,
                                   n_components=min(5, Xtr.shape[1]))
                fm = EstimationEngine(cfg).fit(Xtr, ytr, feature_names=names)
                return fm.predict(Xte).reshape(-1)
            try:
                fold = val.forward_validate(X, y, fit_predict, n_splits=4, purge=1)
            except Exception as e:  # pragma: no cover
                prov.note(f"estimator {est} failed CV: {e}")
                continue
            if np.isfinite(fold.mean_error) and fold.mean_error < best_err:
                best_est, best_err, best_fold = est, fold.mean_error, fold
        if best_est is None:
            best_est = "ridge"
            best_fold = val.forward_validate(
                X, y, lambda a, b, c: np.full(len(c), np.nanmean(b)), n_splits=4)
        prov.note(f"selected estimator '{best_est}' by forward-chaining CV "
                  f"(forward error {best_err:.4f}); high-capacity nonlinear models "
                  f"NOT used by default at this data scale (REG-08)")
        return best_est, best_fold

    def _stability_selection(self, X, y, names, rng, prov):
        """Elastic-net over block-bootstrap resamples; report selection frequency (REG-06)."""
        n, p = X.shape
        counts = np.zeros(p)
        coef_sum = np.zeros(p)
        n_eff = 0
        for _ in range(self.n_stability_resamples):
            idx = val.block_bootstrap(n, min(self.block_size, max(n // 3, 1)), rng)
            if len(np.unique(idx)) < 3:
                continue
            beta = elastic_net_fit(X[idx], y[idx], lam=0.1, l1_ratio=0.7)
            counts += (np.abs(beta) > 1e-6).astype(float)
            coef_sum += beta
            n_eff += 1
        n_eff = max(n_eff, 1)
        freq = counts / n_eff
        prov.note(f"stability selection over {n_eff} block-bootstrap resamples "
                  f"(block_size={self.block_size}); reporting selection frequency, "
                  f"not a single fit (REG-06)")
        return FeatureImportance(names=list(names), selection_frequency=freq,
                                 mean_coef=coef_sum / n_eff)

    def _feedforward_candidates(self, importance, table, adjustable, target_kind, decoupled):
        """Stably-selected NON-knob sensors are feedforward candidates (REG-04/DIAG-01)."""
        cands = []
        for nm, freq, coef in importance.ranked():
            if freq >= self.selection_threshold and nm in table.ff_cols:
                cands.append(nm)
        return cands
