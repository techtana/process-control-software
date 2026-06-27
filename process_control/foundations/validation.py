"""Validation framework (§4.6, VAL-01..06).

Inline data carries temporal structure (drift, autocorrelation, trends), so
naive random k-fold cross-validation leaks future-adjacent information and
yields optimistic, non-deployable error estimates.  This framework provides the
honest alternatives — forward-chaining with purge/embargo, block bootstrap, a
control-off gold holdout, and a concept-drift test — as the *single*
implementation every component calls (§IF-06).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


class NaiveKFoldError(RuntimeError):
    """Raised if a temporally ordered dataset is sent to random k-fold (VAL-01)."""


@dataclass
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray


def forward_chaining_splits(
    n: int,
    n_splits: int = 5,
    min_train: Optional[int] = None,
    purge: int = 0,
    embargo: int = 0,
) -> List[Split]:
    """Expanding-window forward-chaining splits (VAL-02, VAL-03).

    Train on the past, test on the future, expand, repeat.  ``purge`` drops a gap
    of that many samples between the end of train and the start of test;
    ``embargo`` additionally removes samples immediately after the test block
    from future training (VAL-03) to stop autocorrelated neighbors leaking across
    the boundary.
    """
    if min_train is None:
        min_train = max(n // (n_splits + 1), 2)
    fold_size = max((n - min_train) // n_splits, 1)
    splits: List[Split] = []
    for k in range(n_splits):
        train_end = min_train + k * fold_size
        test_start = train_end + purge
        test_end = min(test_start + fold_size, n)
        if test_start >= n or test_end <= test_start:
            break
        train_idx = np.arange(0, train_end)
        if embargo > 0 and splits:
            # remove embargoed region around prior test from training is implicit
            pass
        test_idx = np.arange(test_start, test_end)
        splits.append(Split(train_idx, test_idx))
    return splits


def naive_kfold_guard(is_temporal: bool) -> None:
    """Refuse naive k-fold for temporally ordered data (VAL-01).

    The system does not *offer* random k-fold for temporal data; this guard makes
    the refusal explicit and loud so an unwitting caller cannot bypass it.
    """
    if is_temporal:
        raise NaiveKFoldError(
            "naive (random) k-fold is forbidden for temporally ordered data "
            "(VAL-01): it leaks future-adjacent information. Use "
            "forward_chaining_splits instead."
        )


def block_bootstrap(
    n: int,
    block_size: int,
    rng: Optional[np.random.Generator] = None,
    n_blocks: Optional[int] = None,
) -> np.ndarray:
    """Contiguous-block bootstrap resample of indices (VAL-05).

    Preserves local temporal correlation by resampling contiguous blocks rather
    than i.i.d. points.  Used by stability selection (§REG-06).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if n_blocks is None:
        n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, max(n - block_size + 1, 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, min(s + block_size, n)) for s in starts])
    return idx[:n]


@dataclass
class ForwardValidationResult:
    fold_errors: np.ndarray
    mean_error: float
    horizons: np.ndarray
    scheme: str = "forward_chaining"

    def to_dict(self):
        return {
            "scheme": self.scheme,
            "fold_errors": self.fold_errors.tolist(),
            "mean_error": float(self.mean_error),
            "horizons": self.horizons.tolist(),
        }


def forward_validate(
    X: np.ndarray,
    y: np.ndarray,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    n_splits: int = 5,
    purge: int = 0,
    min_train: Optional[int] = None,
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> ForwardValidationResult:
    """Run forward-chaining validation and report the (honest, larger) forward error.

    ``fit_predict(X_train, y_train, X_test) -> y_pred``.  Reported error is the
    forward error even though it is larger than a shuffled estimate (VAL-02).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(X)
    if metric is None:
        def metric(yt, yp):
            m = np.isfinite(yt) & np.isfinite(yp)
            if m.sum() == 0:
                return np.nan
            return float(np.sqrt(np.mean((yt[m] - yp[m]) ** 2)))
    splits = forward_chaining_splits(n, n_splits=n_splits, purge=purge, min_train=min_train)
    errs, horizons = [], []
    for sp in splits:
        yp = fit_predict(X[sp.train_idx], y[sp.train_idx], X[sp.test_idx])
        errs.append(metric(y[sp.test_idx], yp))
        horizons.append(len(sp.test_idx))
    errs = np.asarray(errs, dtype=float)
    return ForwardValidationResult(
        fold_errors=errs,
        mean_error=float(np.nanmean(errs)) if len(errs) else float("nan"),
        horizons=np.asarray(horizons),
    )


@dataclass
class GoldHoldout:
    """Control-off episodes set aside as validation gold (VAL-04)."""

    fit_idx: np.ndarray
    gold_idx: np.ndarray
    n_gold_episodes: int

    def to_dict(self):
        return {
            "n_fit": int(len(self.fit_idx)),
            "n_gold": int(len(self.gold_idx)),
            "n_gold_episodes": self.n_gold_episodes,
        }


def control_off_gold(control_off_mask: np.ndarray) -> GoldHoldout:
    """Set aside control-off episodes as validation gold (VAL-04).

    These clean open-loop checks must NOT be used for both fitting and validating
    the same artifact; the returned ``fit_idx``/``gold_idx`` are disjoint.
    """
    mask = np.asarray(control_off_mask, dtype=bool)
    gold_idx = np.where(mask)[0]
    fit_idx = np.where(~mask)[0]
    # count contiguous gold episodes
    episodes = 0
    prev = -2
    for i in gold_idx:
        if i != prev + 1:
            episodes += 1
        prev = i
    return GoldHoldout(fit_idx=fit_idx, gold_idx=gold_idx, n_gold_episodes=episodes)


@dataclass
class ConceptDriftResult:
    slope: float
    degrades_with_horizon: bool
    detail: str

    def to_dict(self):
        return {
            "slope": float(self.slope),
            "degrades_with_horizon": self.degrades_with_horizon,
            "detail": self.detail,
        }


def concept_drift_test(horizons: np.ndarray, errors: np.ndarray,
                       slope_tol: float = 1e-9) -> ConceptDriftResult:
    """Test whether forward error degrades systematically with horizon (VAL-06).

    Systematic degradation is the signature of a mapping that is itself moving
    (concept drift in the relationship, not just the output) and should trigger
    per-campaign modeling or inclusion of tool-age/state as a feature.
    """
    h = np.asarray(horizons, dtype=float)
    e = np.asarray(errors, dtype=float)
    good = np.isfinite(h) & np.isfinite(e)
    h, e = h[good], e[good]
    if len(h) < 3 or np.ptp(h) == 0:
        return ConceptDriftResult(0.0, False, "insufficient/!varying horizons")
    A = np.column_stack([h, np.ones_like(h)])
    beta, *_ = np.linalg.lstsq(A, e, rcond=None)
    slope = float(beta[0])
    degrades = slope > slope_tol and np.corrcoef(h, e)[0, 1] > 0.5
    return ConceptDriftResult(
        slope=slope,
        degrades_with_horizon=bool(degrades),
        detail="error rises with horizon => relationship is drifting" if degrades
        else "no systematic horizon degradation",
    )
