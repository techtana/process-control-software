"""DIAG-06 — diagnostic visualizations."""

from __future__ import annotations

import os
from typing import List

import numpy as np

from ...core.counterfactual.reconstruct import realized_gain


def visualize(y_observed, u_used, M_model, innovation, achievability, variance_decomp,
              figure_dir, measured_mask=None) -> List[str]:
    """Render innovation, realized-vs-model gain, and variance-share figures (DIAG-06)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figure_dir, exist_ok=True)
    paths = []
    Y = np.asarray(y_observed, dtype=float)

    if innovation is not None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
        iv = np.asarray(innovation, dtype=float)
        iv0 = iv[:, 0] if iv.ndim > 1 else iv
        ax[0].plot(iv0, lw=0.8)
        ax[0].set_title("Innovation time series")
        ax[0].set_xlabel("event")
        c = iv0[np.isfinite(iv0)]
        c = c - c.mean()
        acf = [1.0] + [float(np.corrcoef(c[k:], c[:-k])[0, 1])
                       for k in range(1, min(15, len(c) // 2))]
        ax[1].bar(range(len(acf)), acf)
        ax[1].axhline(0, color="k", lw=0.5)
        ax[1].set_title("Innovation autocorrelation (whiteness)")
        fig.tight_layout()
        p = os.path.join(figure_dir, "innovation.png")
        fig.savefig(p, dpi=90)
        plt.close(fig)
        paths.append(p)

    G_real = realized_gain(Y, u_used, measured_mask)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(np.asarray(M_model).reshape(-1), G_real.reshape(-1))
    lim = float(np.nanmax(np.abs([np.asarray(M_model), G_real]))) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
    ax.set_xlabel("model gain")
    ax.set_ylabel("realized gain")
    ax.set_title("Realized vs model gain (DIAG-02)")
    fig.tight_layout()
    p = os.path.join(figure_dir, "realized_vs_model_gain.png")
    fig.savefig(p, dpi=90)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    shares = variance_decomp["shares"]
    ax.bar(list(shares), list(shares.values()))
    ax.set_ylabel("variance share")
    ax.set_title(f"Variance decomposition | mean Harris={achievability['mean_harris']:.2f}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    fig.tight_layout()
    p = os.path.join(figure_dir, "variance_decomposition.png")
    fig.savefig(p, dpi=90)
    plt.close(fig)
    paths.append(p)
    return paths
