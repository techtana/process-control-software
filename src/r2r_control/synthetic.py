"""Synthetic data generators with injectable faults (NFR-04).

A small MIMO process under closed-loop R2R control, with DOE campaigns,
control-off episodes, manual overrides, drift, and FF disturbances — all
seedable and reproducible (NFR-02). Faults exercise the edge-case guards:
``ff_leak`` (DIAG-01), ``gain_underestimate`` (DIAG-02), ``gain_drift`` (A1/E1),
``coupled_sensor_knob`` (A3/E2 phantom feedforward) and ``aliased_doe`` (E13).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .contracts.schema import EventTable


@dataclass
class GroundTruth:
    M: np.ndarray
    FF_gain: np.ndarray
    u0: np.ndarray
    targets: np.ndarray
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
    disturbance: np.ndarray
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
    gain_underestimate: float = 1.0,
    gain_drift: float = 0.0,
    ff_leak: float = 0.0,
    coupled_sensor_knob: Optional[Tuple[int, int]] = None,
    include_doe: bool = True,
    doe_runs: int = 16,
    doe_age_days: float = 30.0,
    aliased_doe: bool = False,
) -> SyntheticDataset:
    """Generate a closed-loop R2R dataset with optional injected faults.

    ``gain_drift`` scales the true gain linearly from ``1 - gain_drift/2`` to
    ``1 + gain_drift/2`` across the record (A1/E1). ``coupled_sensor_knob=(f, k)``
    makes FF sensor ``f`` a noisy copy of knob ``k``'s movement, so it lives in
    the knob/M-error directions (A3/E2).
    """
    rng = np.random.default_rng(seed)
    knob_names = [f"u{j}" for j in range(n_knob)]
    ff_names = [f"ff{j}" for j in range(n_ff)]
    out_names = [f"y{j}" for j in range(n_out)]

    M = rng.normal(0.0, 1.0, size=(n_out, n_knob))
    M += np.sign(M) * 0.5
    FF_gain = rng.normal(0.0, 0.6, size=(n_out, n_ff))
    u0 = rng.normal(0.0, 0.5, size=n_knob)
    targets = rng.normal(0.0, 0.2, size=n_out)
    noise_std = noise_level * (0.5 + rng.random(n_out))

    M_ctrl = M / gain_underestimate
    M_ctrl_pinv = np.linalg.pinv(M_ctrl)

    disturbance = np.zeros((n_events, n_out))
    drift = np.zeros(n_out)
    d_prev = rng.normal(0, 1, n_out)
    for t in range(n_events):
        drift = drift + drift_std * rng.standard_normal(n_out)
        d_prev = dist_ar * d_prev + np.sqrt(1 - dist_ar ** 2) * rng.standard_normal(n_out)
        disturbance[t] = drift + d_prev

    FF = rng.normal(0, 1, size=(n_events, n_ff))

    control_off = np.zeros(n_events, dtype=bool)
    if control_off_episodes > 0:
        for s in np.linspace(n_events * 0.2, n_events * 0.8, control_off_episodes).astype(int):
            control_off[s:s + control_off_len] = True

    u_recommended = np.zeros((n_events, n_knob))
    u_used = np.zeros((n_events, n_knob))
    y = np.zeros((n_events, n_out))
    est_dist = np.zeros(n_out)
    for t in range(n_events):
        M_t = M * (1.0 + gain_drift * (t / max(n_events - 1, 1) - 0.5))
        ff_contribution = FF[t] @ FF_gain.T
        error = est_dist + (1.0 - ff_leak) * ff_contribution - targets
        rec = u0 - control_gain * (M_ctrl_pinv @ error)
        u_recommended[t] = rec
        if control_off[t]:
            used = u0.copy()
        else:
            used = rec.copy()
            if rng.random() < override_fraction:
                used[rng.integers(0, n_knob)] += rng.normal(0, 1.0)
        u_used[t] = used
        y[t] = (disturbance[t] + ff_leak * ff_contribution + M_t @ (used - u0)
                + noise_std * rng.standard_normal(n_out))
        est_dist = 0.7 * est_dist + 0.3 * (y[t] - M_ctrl @ (used - u0))

    if coupled_sensor_knob is not None:
        fi, ki = coupled_sensor_knob
        FF[:, fi] = (u_used[:, ki] - u0[ki]) + 0.1 * rng.standard_normal(n_events)

    measured = rng.random(n_events) > metrology_sparsity
    y_cells, y_time = [], []
    for t in range(n_events):
        if measured[t]:
            arr = y[t][:, None] + 0.02 * rng.standard_normal((n_out, 3))
            y_cells.append(list(arr))
            y_time.append(t + rng.integers(0, 4))
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
        data[f"{nm}_time"] = y_time
    data["control_off"] = control_off

    inline = EventTable(
        frame=pd.DataFrame(data),
        knob_cols_recommended=[f"{nm}_rec" for nm in knob_names],
        knob_cols_used=[f"{nm}_used" for nm in knob_names],
        ff_cols=ff_names, y_cols=out_names,
        y_time_cols={nm: f"{nm}_time" for nm in out_names},
        time_col="timestamp", control_off_col="control_off",
    )
    doe = (_make_doe(rng, M, FF_gain, u0, knob_names, ff_names, out_names, doe_runs,
                     noise_std, aliased=aliased_doe) if include_doe else None)
    truth = GroundTruth(
        M=M, FF_gain=FF_gain, u0=u0, targets=targets, n_out=n_out, n_knob=n_knob,
        n_ff=n_ff, noise_std=noise_std, drift_std=drift_std,
        config={"gain_underestimate": gain_underestimate, "gain_drift": gain_drift,
                "ff_leak": ff_leak, "control_gain": control_gain, "M_ctrl": M_ctrl,
                "coupled_sensor_knob": coupled_sensor_knob})
    return SyntheticDataset(inline=inline, doe=doe, truth=truth, disturbance=disturbance,
                            knob_names=knob_names, ff_names=ff_names, out_names=out_names)


def _make_doe(rng, M, FF_gain, u0, knob_names, ff_names, out_names, runs, noise_std,
              aliased=False):
    n_knob, n_ff, n_out = len(knob_names), len(ff_names), len(out_names)
    if aliased and n_knob >= 3:
        base = rng.choice([-1.0, 1.0], size=(runs, n_knob - 1))
        design = np.column_stack([base, base[:, 0] * base[:, 1]])
    else:
        design = rng.choice([-1.0, 1.0], size=(runs, n_knob))
    u_used = u0[None, :] + 2.0 * design
    FF = np.zeros((runs, n_ff))
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
    return EventTable(
        frame=pd.DataFrame(data),
        knob_cols_recommended=[f"{nm}_rec" for nm in knob_names],
        knob_cols_used=[f"{nm}_used" for nm in knob_names],
        ff_cols=ff_names, y_cols=out_names,
        y_time_cols={nm: f"{nm}_time" for nm in out_names},
        time_col="timestamp", control_off_col="control_off",
    )
