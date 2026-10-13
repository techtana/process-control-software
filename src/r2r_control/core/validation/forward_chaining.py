"""Forward-chaining temporal validation (VAL-01/02/03)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .purge_embargo import apply_embargo


class NaiveKFoldError(RuntimeError):
    """Raised if a temporally ordered dataset is sent to random k-fold (VAL-01)."""


@dataclass
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray


def naive_kfold_guard(is_temporal: bool) -> None:
    """Refuse naive k-fold for temporally ordered data (VAL-01)."""
    if is_temporal:
        raise NaiveKFoldError(
            "naive (random) k-fold is forbidden for temporally ordered data (VAL-01): "
            "it leaks future-adjacent information. Use forward_chaining_splits.")


def forward_chaining_splits(n: int, n_splits: int = 5, min_train: Optional[int] = None,
                            purge: int = 0, embargo: int = 0) -> List[Split]:
    """Expanding-window forward-chaining splits with purge/embargo (VAL-02/03)."""
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
        test_idx = np.arange(test_start, test_end)
        if embargo > 0:
            train_idx = apply_embargo(train_idx, test_idx, embargo)
        splits.append(Split(train_idx, test_idx))
    return splits


@dataclass
class ForwardValidationResult:
    fold_errors: np.ndarray
    mean_error: float
    horizons: np.ndarray
    scheme: str = "forward_chaining"

    def to_dict(self):
        return {"scheme": self.scheme, "fold_errors": self.fold_errors.tolist(),
                "mean_error": float(self.mean_error), "horizons": self.horizons.tolist()}


def _rmse(yt, yp):
    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((yt[m] - yp[m]) ** 2)))


def forward_validate(X: np.ndarray, y: np.ndarray,
                     fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                     n_splits: int = 5, purge: int = 0, min_train: Optional[int] = None,
                     metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
                     ) -> ForwardValidationResult:
    """Run forward-chaining validation; report the honest forward error (VAL-02).

    ``horizons`` records, per fold, how far the test block sits past the start of
    the record (the test-block end index) — used by the concept-drift test.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    metric = metric or _rmse
    splits = forward_chaining_splits(len(X), n_splits=n_splits, purge=purge,
                                     min_train=min_train)
    errs, horizons = [], []
    for sp in splits:
        yp = fit_predict(X[sp.train_idx], y[sp.train_idx], X[sp.test_idx])
        errs.append(metric(y[sp.test_idx], yp))
        horizons.append(int(sp.test_idx.max()))
    errs = np.asarray(errs, dtype=float)
    return ForwardValidationResult(
        fold_errors=errs,
        mean_error=float(np.nanmean(errs)) if len(errs) else float("nan"),
        horizons=np.asarray(horizons))
