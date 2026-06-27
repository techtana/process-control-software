"""Component 5 — Diagnostics and Visualization (§9, DIAG-01..06).

Diagnose defective control models and tell the engineer *where* a performance
problem lives — including the decisive judgement, borrowed from the
Marchenko-Pastur / Ledoit-Peche noise-floor idea, of whether the controller is
already as good as it can be and the remaining variance is irreducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..foundations import stats
from ..foundations import noise_floor as nf
from ..foundations.counterfactual import realized_gain


@dataclass
class Diagnosis:
    ff_leakage: Dict
    gain_mismatch: Dict
    qr_attribution: Dict
    variance_decomposition: Dict
    achievability: Dict
    figures: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "ff_leakage": self.ff_leakage,
            "gain_mismatch": self.gain_mismatch,
            "qr_attribution": self.qr_attribution,
            "variance_decomposition": self.variance_decomposition,
            "achievability": self.achievability,
            "figures": self.figures,
            "notes": self.notes,
        }


class DiagnosticsEngine:
    """Component 5.  Attribute residual variance to causes with an achievability verdict."""

    def __init__(self, delay: int = 1, alpha: float = 0.05):
        self.delay = delay
        self.alpha = alpha

    # -- DIAG-01 ---------------------------------------------------------
    def ff_leakage(self, state_est: np.ndarray, ff_inputs: np.ndarray,
                   ff_names: Optional[List[str]] = None) -> Dict:
        """Test whether changes in estimated state correlate with FF inputs (DIAG-01).

        A significant correlation means the FF model is failing to account for a
        measurable disturbance, so feedback is reactively absorbing what
        feedforward should pre-compensate.  Such FF inputs are reported as
        feedforward opportunities.
        """
        state_est = np.asarray(state_est, dtype=float)
        ff = np.asarray(ff_inputs, dtype=float)
        if ff.size == 0 or len(state_est) < 5:
            return {"leaking_inputs": [], "detail": "no FF inputs / too few samples"}
        d_state = np.diff(state_est, axis=0)
        d_ff = np.diff(ff, axis=0)
        names = ff_names or [f"ff{j}" for j in range(ff.shape[1])]
        leaking = []
        corrs = {}
        n = len(d_state)
        for j in range(ff.shape[1]):
            best = 0.0
            for k in range(d_state.shape[1]):
                if np.std(d_ff[:, j]) < 1e-9 or np.std(d_state[:, k]) < 1e-9:
                    continue
                r = np.corrcoef(d_ff[:, j], d_state[:, k])[0, 1]
                if abs(r) > abs(best):
                    best = r
            # approximate significance via Fisher z
            if n > 4 and abs(best) < 1:
                z = 0.5 * np.log((1 + best) / (1 - best)) * np.sqrt(n - 3)
                from scipy.stats import norm
                p = 2 * norm.sf(abs(z))
            else:
                p = 1.0
            corrs[names[j]] = {"max_abs_corr": float(abs(best)), "pvalue": float(p)}
            if p < self.alpha and abs(best) > 0.2:
                leaking.append(names[j])
        return {
            "leaking_inputs": leaking,
            "correlations": corrs,
            "feedforward_opportunities": leaking,
            "detail": "estimated-state vs FF correlation; significant => FF leakage",
        }

    # -- DIAG-02 ---------------------------------------------------------
    def gain_mismatch(self, y_observed: np.ndarray, u_used: np.ndarray,
                      M_model: np.ndarray, innovation: Optional[np.ndarray] = None,
                      measured_mask: Optional[np.ndarray] = None) -> Dict:
        """Detect over-correction from underestimated gain (DIAG-02).

        Compare realized gain (Δoutput/Δknob via the counterfactual service)
        against the model gain, and flag oscillation signatures in the
        innovation/output.  Distinguish "moves too large because gain is
        underestimated" from genuine disturbance growth.
        """
        G_real = realized_gain(y_observed, u_used, measured_mask)
        M_model = np.asarray(M_model, dtype=float)
        ratio = np.full_like(M_model, np.nan)
        mask = np.abs(M_model) > 1e-9
        ratio[mask] = G_real[mask] / M_model[mask]
        median_ratio = float(np.nanmedian(ratio))
        underestimated = median_ratio > 1.3   # realized > model => model gain too small

        # oscillation signature: strong NEGATIVE lag-1 autocorrelation of the
        # output LEVEL (alternating above/below target = ringing).  Note we must
        # use the level, not the first difference: a well-tuned controller yields
        # near-white output, whose differences have lag-1 autocorrelation ~ -0.5
        # by construction — so differencing would falsely flag a healthy loop.
        osc_scores = []
        Y = np.asarray(y_observed, dtype=float)
        for j in range(Y.shape[1]):
            col = Y[:, j]
            col = col[np.isfinite(col)]
            col = col - col.mean()
            if len(col) > 6 and np.std(col) > 1e-9:
                ac1 = np.corrcoef(col[1:], col[:-1])[0, 1]
                osc_scores.append(ac1)
        mean_ac1 = float(np.mean(osc_scores)) if osc_scores else np.nan
        oscillating = np.isfinite(mean_ac1) and mean_ac1 < -0.3

        # disturbance growth check (innovation variance trend)
        growing = False
        if innovation is not None:
            iv = np.asarray(innovation, dtype=float)
            half = len(iv) // 2
            if half > 5:
                v1 = np.nanvar(iv[:half]); v2 = np.nanvar(iv[half:])
                growing = v2 > 2.0 * v1
        cause = "none"
        if oscillating and underestimated:
            cause = "gain_underestimated_overcorrection"
        elif oscillating and growing:
            cause = "disturbance_growth"
        elif oscillating:
            cause = "oscillation_unattributed"
        return {
            "realized_vs_model_ratio_median": median_ratio,
            "gain_underestimated": bool(underestimated),
            "oscillation_lag1_autocorr": mean_ac1,
            "oscillating": bool(oscillating),
            "disturbance_growth": bool(growing),
            "diagnosis": cause,
        }

    # -- DIAG-03 ---------------------------------------------------------
    def qr_attribution(self, innovation: np.ndarray, u_used: np.ndarray,
                       y_observed: np.ndarray) -> Dict:
        """Attribute a problem to Controller-QR vs State-QR (DIAG-03).

        Controller-weight problems show in the move/tracking trade-off (knob
        moves large relative to tracking gain); estimator-weight problems show in
        the responsiveness and whiteness of the innovation.  Honors the FB-08
        limit that passive data alone cannot fully separate gain error from
        estimator tuning.
        """
        innov = np.asarray(innovation, dtype=float)
        innov = innov[np.all(np.isfinite(innov), axis=1)] if innov.ndim > 1 else \
            innov[np.isfinite(innov)]
        # innovation whiteness => estimator (State QR) signature
        lb_p = []
        innov2d = innov if innov.ndim > 1 else innov[:, None]
        for j in range(innov2d.shape[1]):
            lb = stats.ljung_box(innov2d[:, j], lags=min(10, len(innov2d) // 3))
            if lb.pvalue is not None:
                lb_p.append(lb.pvalue)
        innov_colored = bool(lb_p and np.nanmin(lb_p) < self.alpha)

        # move aggressiveness => controller (Controller QR) signature
        du = np.diff(np.asarray(u_used, dtype=float), axis=0)
        move_energy = float(np.mean(np.sum(du ** 2, axis=1)))
        Y = np.asarray(y_observed, dtype=float)
        track_energy = float(np.nanmean(np.nansum((Y - np.nanmean(Y, 0)) ** 2, axis=1)))
        move_to_track = move_energy / (track_energy + 1e-9)

        if innov_colored and move_to_track < 0.5:
            primary = "State_QR"          # estimator: innovation not white
        elif move_to_track > 1.0 and not innov_colored:
            primary = "Controller_QR"     # controller: aggressive moves, white innovation
        elif innov_colored:
            primary = "State_QR"
        else:
            primary = "neither_dominant"
        return {
            "innovation_colored": innov_colored,
            "min_ljung_box_p": float(np.nanmin(lb_p)) if lb_p else None,
            "move_to_track_ratio": move_to_track,
            "primary_suspect": primary,
            "fb08_caveat": ("passive data cannot fully separate gain error from "
                            "estimator tuning; resolution requires excitation or the "
                            "estimator spec (FB-08)"),
        }

    # -- DIAG-04 + DIAG-05 ----------------------------------------------
    def variance_decomposition(self, y_observed: np.ndarray, ff_leakage: Dict,
                               gain_mismatch: Dict, qr: Dict,
                               innovation: Optional[np.ndarray] = None) -> Dict:
        """Decompose residual variance into recognized sources (DIAG-04).

        Shares attributed to: incorrect FF model, incorrect process gain,
        sub-optimal estimator tuning, and irreducible noise — combining the
        Harris index (irreducible floor), whiteness testing, and the qualitative
        attributions above.  The identifiability caveat is surfaced, not hidden.
        """
        Y = np.asarray(y_observed, dtype=float)
        total_var = float(np.nanmean(np.nanvar(Y, axis=0)))
        # irreducible share from Harris benchmark
        mv = np.nanmean([nf.minimum_variance_benchmark(Y[:, j], self.delay)
                         for j in range(Y.shape[1])])
        irreducible = float(np.clip(mv / (total_var + 1e-12), 0, 1))
        recoverable = 1.0 - irreducible

        # split recoverable across causes by qualitative evidence weights
        w_ff = 1.0 if ff_leakage.get("leaking_inputs") else 0.0
        w_gain = 1.0 if gain_mismatch.get("gain_underestimated") or \
            gain_mismatch.get("oscillating") else 0.0
        w_est = 1.0 if qr.get("innovation_colored") else 0.0
        w_sum = w_ff + w_gain + w_est
        if w_sum == 0:
            shares = {"incorrect_ff_model": 0.0, "incorrect_process_gain": 0.0,
                      "suboptimal_estimator_tuning": 0.0}
            # unattributed recoverable
            unattributed = recoverable
        else:
            shares = {
                "incorrect_ff_model": recoverable * w_ff / w_sum,
                "incorrect_process_gain": recoverable * w_gain / w_sum,
                "suboptimal_estimator_tuning": recoverable * w_est / w_sum,
            }
            unattributed = 0.0
        shares["irreducible_noise"] = irreducible
        shares["unattributed_recoverable"] = unattributed
        return {
            "total_variance": total_var,
            "shares": shares,
            "identifiability_caveat": ("gain vs estimator-tuning shares are not "
                                       "separable from passive data alone (FB-08)"),
        }

    def achievability_verdict(self, y_observed: np.ndarray) -> Dict:
        """The MP/Ledoit-Peche analog: is the controller at the floor? (DIAG-05)."""
        Y = np.asarray(y_observed, dtype=float)
        harris = [nf.harris_index(Y[:, j], self.delay) for j in range(Y.shape[1])]
        mean_h = float(np.nanmean(harris))
        at_floor = mean_h <= 1.25
        if at_floor:
            verdict = ("the controller is working as well as it could; the remaining "
                       "variance is unexplained / irreducible (at the minimum-variance "
                       "floor, Harris≈1)")
        else:
            verdict = (f"achieved variance sits {mean_h:.2f}x above the minimum-variance "
                       f"floor; recoverable gap exists, route to FF/gain/QR cause")
        return {
            "harris_per_output": [float(h) for h in harris],
            "mean_harris": mean_h,
            "at_achievability_floor": bool(at_floor),
            "verdict": verdict,
        }

    # -- orchestration ---------------------------------------------------
    def diagnose(
        self,
        y_observed: np.ndarray,
        u_used: np.ndarray,
        M_model: np.ndarray,
        state_est: Optional[np.ndarray] = None,
        ff_inputs: Optional[np.ndarray] = None,
        innovation: Optional[np.ndarray] = None,
        ff_names: Optional[List[str]] = None,
        measured_mask: Optional[np.ndarray] = None,
        figure_dir: Optional[str] = None,
    ) -> Diagnosis:
        ff_leak = self.ff_leakage(state_est, ff_inputs, ff_names) if state_est is not None \
            and ff_inputs is not None else {"leaking_inputs": [], "detail": "no state/ff"}
        gm = self.gain_mismatch(y_observed, u_used, M_model, innovation, measured_mask)
        qr = self.qr_attribution(innovation if innovation is not None else
                                 np.diff(y_observed, axis=0), u_used, y_observed)
        vd = self.variance_decomposition(y_observed, ff_leak, gm, qr, innovation)
        ach = self.achievability_verdict(y_observed)
        notes = [ach["verdict"]]
        if ff_leak.get("leaking_inputs"):
            notes.append(f"FF leakage: {ff_leak['leaking_inputs']} are feedforward "
                         f"opportunities")
        if gm["diagnosis"] != "none":
            notes.append(f"gain/stability: {gm['diagnosis']}")
        figures = []
        if figure_dir is not None:
            figures = self.visualize(y_observed, u_used, M_model, state_est, ff_inputs,
                                     innovation, ach, vd, figure_dir, measured_mask)
        return Diagnosis(ff_leakage=ff_leak, gain_mismatch=gm, qr_attribution=qr,
                         variance_decomposition=vd, achievability=ach,
                         figures=figures, notes=notes)

    # -- DIAG-06 ---------------------------------------------------------
    def visualize(self, y_observed, u_used, M_model, state_est, ff_inputs,
                  innovation, achievability, variance_decomp, figure_dir,
                  measured_mask=None) -> List[str]:
        """Render the diagnostic visualizations (DIAG-06)."""
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(figure_dir, exist_ok=True)
        paths = []
        Y = np.asarray(y_observed, dtype=float)

        # (a) innovation time series + whiteness
        if innovation is not None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
            iv = np.asarray(innovation, dtype=float)
            iv0 = iv[:, 0] if iv.ndim > 1 else iv
            ax[0].plot(iv0, lw=0.8)
            ax[0].set_title("Innovation time series"); ax[0].set_xlabel("event")
            iv0c = iv0[np.isfinite(iv0)]
            iv0c = iv0c - iv0c.mean()
            acf = [1.0] + [float(np.corrcoef(iv0c[k:], iv0c[:-k])[0, 1])
                           for k in range(1, min(15, len(iv0c) // 2))]
            ax[1].bar(range(len(acf)), acf)
            ax[1].axhline(0, color="k", lw=0.5)
            ax[1].set_title("Innovation autocorrelation (whiteness)")
            fig.tight_layout()
            p = os.path.join(figure_dir, "innovation.png")
            fig.savefig(p, dpi=90); plt.close(fig); paths.append(p)

        # (b) realized vs model gain
        G_real = realized_gain(Y, u_used, measured_mask)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.scatter(np.asarray(M_model).reshape(-1), G_real.reshape(-1))
        lim = np.nanmax(np.abs([np.asarray(M_model), G_real])) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
        ax.set_xlabel("model gain"); ax.set_ylabel("realized gain")
        ax.set_title("Realized vs model gain (DIAG-02)")
        fig.tight_layout()
        p = os.path.join(figure_dir, "realized_vs_model_gain.png")
        fig.savefig(p, dpi=90); plt.close(fig); paths.append(p)

        # (c) variance-decomposition shares + achievability
        fig, ax = plt.subplots(figsize=(6, 3.2))
        shares = variance_decomp["shares"]
        ax.bar(list(shares.keys()), list(shares.values()))
        ax.set_ylabel("variance share")
        ax.set_title(f"Variance decomposition | mean Harris="
                     f"{achievability['mean_harris']:.2f}")
        plt.xticks(rotation=30, ha="right", fontsize=7)
        fig.tight_layout()
        p = os.path.join(figure_dir, "variance_decomposition.png")
        fig.savefig(p, dpi=90); plt.close(fig); paths.append(p)
        return paths
