"""Synthetic data generators with injectable faults (NFR-04).

Each diagnostic / estimation module ships with synthetic smoke tests carrying
deliberately injected faults, including intentional failure cases that confirm
correct behavior at known identifiability limits.  This module is the common
ground truth: a small MIMO process under closed-loop R2R control, with DOE
campaigns, control-off episodes, manual overrides, drift, and FF disturbances,
all seedable and reproducible (NFR-02).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .data.schema import EventTable


@dataclass
class GroundTruth:
    """The true process used to generate synthetic data (known answer for tests)."""

    M: np.ndarray                  # true gain (n_out x n_knob)
    FF_gain: np.ndarray            # true FF gain (n_out x n_ff)
    u0: np.ndarray                 # true open-loop baseline knob
    targets: np.ndarray            # per-output targets
    n_out: int
    n_knob: int
    n_ff: int
    noise_std: np.ndarray
    drift_std: float
    config: dict = field(default_factory=dict)


@dataclass
class SyntheticDataset:
    inline: EventTable
    doe: Optional[EventTable]
    truth: GroundTruth
    disturbance: np.ndarray        # the latent disturbance driving the plant
    knob_names: List[str]
    ff_names: List[str]
    out_names: List[str]


def make_dataset(
    n_events: int = 400,
    n_knob: int = 4,
    n_out: int = 3,
    n_ff: int = 5,
    seed: int = 0,
    *,
    control_gain: float = 0.7,
    drift_std: float = 0.02,
    dist_ar: float = 0.85,
    noise_level: float = 0.3,
    control_off_episodes: int = 2,
    control_off_len: int = 15,
    override_fraction: float = 0.05,
    metrology_sparsity: float = 0.4,
    # ---- injectable faults (NFR-04) ----
    gain_underestimate: float = 1.0,    # >1 => model gain too small vs realized
    ff_leak: float = 0.0,               # FF disturbance the controller can't see
    include_doe: bool = True,
    doe_runs: int = 16,
    doe_age_days: float = 30.0,
    aliased_doe: bool = False,
) -> SyntheticDataset:
    """Generate a closed-loop R2R dataset with optional injected faults.

    The plant is ``y_t = disturbance_t + M @ (u_used_t - u0) + noise``.  A simple
    proportional R2R controller drives ``u`` to push estimated disturbance toward
    target.  Faults: ``gain_underestimate`` makes the *controller's* model gain
    smaller than the true gain (the over-correction/oscillation failure mode,
    DIAG-02); ``ff_leak`` adds a measurable disturbance the FF model ignores
    (FF leakage, DIAG-01); ``aliased_doe`` builds a low-resolution design.
    """
    rng = np.random.default_rng(seed)
    knob_names = [f"u{j}" for j in range(n_knob)]
    ff_names = [f"ff{j}" for j in range(n_ff)]
    out_names = [f"y{j}" for j in range(n_out)]

    M = rng.normal(0.0, 1.0, size=(n_out, n_knob))
    M += np.sign(M) * 0.5                       # keep gains away from zero
    FF_gain = rng.normal(0.0, 0.6, size=(n_out, n_ff))
    u0 = rng.normal(0.0, 0.5, size=n_knob)
    targets = rng.normal(0.0, 0.2, size=n_out)
    noise_std = noise_level * (0.5 + rng.random(n_out))

    # controller believes a (possibly wrong) gain
    M_ctrl = M / gain_underestimate
    M_ctrl_pinv = np.linalg.pinv(M_ctrl)

    # latent AR(1) disturbance + slow drift
    disturbance = np.zeros((n_events, n_out))
    drift = np.zeros(n_out)
    d_prev = rng.normal(0, 1, n_out)
    for t in range(n_events):
        drift = drift + drift_std * rng.standard_normal(n_out)
        d_prev = dist_ar * d_prev + np.sqrt(1 - dist_ar ** 2) * rng.standard_normal(n_out)
        disturbance[t] = drift + d_prev

    # FF inputs: some drive the output (measurable disturbance), some are noise
    FF = rng.normal(0, 1, size=(n_events, n_ff))
    ff_contribution = FF @ FF_gain.T

    # control-off schedule
    control_off = np.zeros(n_events, dtype=bool)
    if control_off_episodes > 0:
        starts = np.linspace(n_events * 0.2, n_events * 0.8,
                             control_off_episodes).astype(int)
        for s in starts:
            control_off[s:s + control_off_len] = True

    u_recommended = np.zeros((n_events, n_knob))
    u_used = np.zeros((n_events, n_knob))
    y = np.zeros((n_events, n_out))

    est_dist = np.zeros(n_out)
    for t in range(n_events):
        # FF compensation only sees the part of ff_contribution the FF model knows;
        # ff_leak fraction is hidden from the controller.
        ff_seen = (1.0 - ff_leak) * ff_contribution[t]
        ff_hidden = ff_leak * ff_contribution[t]
        # controller target: cancel estimated disturbance + seen FF, hit target
        error = est_dist + ff_seen - targets
        move = control_gain * (M_ctrl_pinv @ error)
        rec = u0 - move
        u_recommended[t] = rec
        if control_off[t]:
            used = u0.copy()                     # control off => baseline
        else:
            used = rec.copy()
            # manual override: occasionally a knob is nudged by hand
            if rng.random() < override_fraction:
                k = rng.integers(0, n_knob)
                used[k] += rng.normal(0, 1.0)
        u_used[t] = used
        control_contrib = M @ (used - u0)
        meas_noise = noise_std * rng.standard_normal(n_out)
        y[t] = disturbance[t] + ff_hidden + control_contrib + meas_noise
        # estimator updates its disturbance estimate from the (noisy) outcome,
        # crediting its own control contribution back out
        est_dist = 0.7 * est_dist + 0.3 * (y[t] - M_ctrl @ (used - u0))

    # sparse, asynchronous metrology: drop a fraction of measurements
    measured = rng.random(n_events) > metrology_sparsity
    y_cells = []
    y_time = []
    for t in range(n_events):
        if measured[t]:
            # array-valued cell (per-site), some sites may be NaN
            arr = y[t][:, None] + 0.02 * rng.standard_normal((n_out, 3))
            y_cells.append([row for row in arr])
            y_time.append(t + rng.integers(0, 4))   # asynchronous arrival
        else:
            y_cells.append([np.array([np.nan, np.nan, np.nan]) for _ in range(n_out)])
            y_time.append(np.nan)

    data = {"timestamp": np.arange(n_events)}
    for j, nm in enumerate(knob_names):
        data[f"{nm}_rec"] = u_recommended[:, j]
        data[f"{nm}_used"] = u_used[:, j]
    for j, nm in enumerate(ff_names):
        data[nm] = FF[:, j]
    for j, nm in enumerate(out_names):
        data[nm] = [y_cells[t][j] for t in range(n_events)]
        data[f"{nm}_time"] = [y_time[t] for t in range(n_events)]
    data["control_off"] = control_off
    frame = pd.DataFrame(data)

    inline = EventTable(
        frame=frame,
        knob_cols_recommended=[f"{nm}_rec" for nm in knob_names],
        knob_cols_used=[f"{nm}_used" for nm in knob_names],
        ff_cols=ff_names,
        y_cols=out_names,
        y_time_cols={nm: f"{nm}_time" for nm in out_names},
        time_col="timestamp",
        control_off_col="control_off",
    )

    doe = None
    if include_doe:
        doe = _make_doe(rng, M, FF_gain, u0, knob_names, ff_names, out_names,
                        doe_runs, noise_std, aliased=aliased_doe)

    truth = GroundTruth(
        M=M, FF_gain=FF_gain, u0=u0, targets=targets, n_out=n_out, n_knob=n_knob,
        n_ff=n_ff, noise_std=noise_std, drift_std=drift_std,
        config={"gain_underestimate": gain_underestimate, "ff_leak": ff_leak,
                "control_gain": control_gain, "M_ctrl": M_ctrl},
    )
    return SyntheticDataset(inline=inline, doe=doe, truth=truth,
                            disturbance=disturbance, knob_names=knob_names,
                            ff_names=ff_names, out_names=out_names)


def _make_doe(rng, M, FF_gain, u0, knob_names, ff_names, out_names, runs,
              noise_std, aliased=False):
    n_knob = len(knob_names)
    n_ff = len(ff_names)
    n_out = len(out_names)
    # 2-level coded design over knobs (wide operating window)
    if aliased:
        # build a low-resolution fractional design: last knob = product of first two
        base = rng.choice([-1.0, 1.0], size=(runs, max(n_knob - 1, 1)))
        if n_knob >= 3:
            last = base[:, 0] * base[:, 1]
            design = np.column_stack([base, last])[:, :n_knob]
        else:
            design = base[:, :n_knob]
    else:
        design = rng.choice([-1.0, 1.0], size=(runs, n_knob))
    amp = 2.0
    u_used = u0[None, :] + amp * design
    FF = np.zeros((runs, n_ff))      # DOE holds FF at nominal
    y = (u_used - u0) @ M.T + FF @ FF_gain.T + noise_std * rng.standard_normal((runs, n_out))

    data = {"timestamp": np.arange(runs)}
    for j, nm in enumerate(knob_names):
        data[f"{nm}_rec"] = u_used[:, j]
        data[f"{nm}_used"] = u_used[:, j]
    for j, nm in enumerate(ff_names):
        data[nm] = FF[:, j]
    for j, nm in enumerate(out_names):
        data[nm] = [np.array([v]) for v in y[:, j]]
        data[f"{nm}_time"] = np.arange(runs)
    data["control_off"] = np.ones(runs, dtype=bool)
    frame = pd.DataFrame(data)
    return EventTable(
        frame=frame,
        knob_cols_recommended=[f"{nm}_rec" for nm in knob_names],
        knob_cols_used=[f"{nm}_used" for nm in knob_names],
        ff_cols=ff_names,
        y_cols=out_names,
        y_time_cols={nm: f"{nm}_time" for nm in out_names},
        time_col="timestamp",
        control_off_col="control_off",
    )
