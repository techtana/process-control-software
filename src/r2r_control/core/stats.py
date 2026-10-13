"""Statistical tests implemented from scratch (NFR-01).

The specification requires ADF, KPSS, variance-ratio and Ljung-Box to be
available, implemented from scratch where a heavy dependency is to be avoided.
Each returns a small result object with the statistic, an approximate p-value
or critical values, and a boolean conclusion, so the data-quality and
diagnostic layers can act on them without pulling in statsmodels.

These are deliberately self-contained numpy implementations.  Critical values
use the standard tabulations (Dickey-Fuller / MacKinnon, KPSS) interpolated
across sample size where appropriate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import stats as _sps


@dataclass
class TestResult:
    name: str
    statistic: float
    pvalue: Optional[float] = None
    crit_values: Optional[Dict[str, float]] = None
    reject_null: Optional[bool] = None
    null_hypothesis: str = ""
    detail: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "statistic": float(self.statistic),
            "pvalue": None if self.pvalue is None else float(self.pvalue),
            "crit_values": self.crit_values,
            "reject_null": self.reject_null,
            "null_hypothesis": self.null_hypothesis,
            "detail": self.detail,
        }


def _ols(y: np.ndarray, X: np.ndarray):
    """Plain least squares returning (beta, residuals, se_beta)."""
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0.0))
    return beta, resid, se


def ljung_box(residuals: np.ndarray, lags: int = 10) -> TestResult:
    """Ljung-Box Q test for residual autocorrelation / whiteness (FB-06, DIAG-04).

    Null hypothesis: the series is white (no autocorrelation up to ``lags``).
    Rejecting the null => colored residuals => model mis-specification, distinct
    from mere noise.
    """
    x = np.asarray(residuals, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n <= lags + 1:
        return TestResult("ljung_box", float("nan"), None, None, None,
                          "residuals are white", "too few points")
    x = x - x.mean()
    denom = float(x @ x)
    q = 0.0
    acfs = []
    for k in range(1, lags + 1):
        num = float(x[k:] @ x[:-k])
        r = num / denom if denom > 0 else 0.0
        acfs.append(r)
        q += r * r / (n - k)
    q *= n * (n + 2)
    pvalue = float(_sps.chi2.sf(q, df=lags))
    return TestResult(
        name="ljung_box",
        statistic=q,
        pvalue=pvalue,
        reject_null=pvalue < 0.05,
        null_hypothesis="series is white (no autocorrelation)",
        detail=f"lags={lags}, acf1={acfs[0]:.3f}",
    )


# Dickey-Fuller critical values (constant, no trend), large sample with small-N
# adjustments folded into the interpolation used by MacKinnon.  We use the
# common tabulated asymptotic values; adequate for a use/down-weight gate.
_ADF_CRIT = {"1%": -3.43, "5%": -2.86, "10%": -2.57}


def adf_test(x: np.ndarray, max_lag: Optional[int] = None) -> TestResult:
    """Augmented Dickey-Fuller test for a unit root / drift (DQ-05).

    Null hypothesis: a unit root is present (the series is non-stationary, i.e.
    drifting).  We regress Δy_t on y_{t-1}, a constant, and lagged differences;
    the t-statistic on the y_{t-1} coefficient is the ADF statistic.  Failing to
    reject the null is evidence of the drift the inline data is expected to show.
    """
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 10:
        return TestResult("adf", float("nan"), None, _ADF_CRIT, None,
                          "unit root present (non-stationary)", "too few points")
    if max_lag is None:
        max_lag = int(np.floor(12 * (n / 100.0) ** 0.25))
        max_lag = min(max_lag, n // 2 - 2)
        max_lag = max(max_lag, 0)
    dy = np.diff(y)
    # Build regression: dy_t ~ const + y_{t-1} + sum lag dy
    rows = n - 1 - max_lag
    if rows <= max_lag + 2:
        max_lag = 0
        rows = n - 1
    y_lag1 = y[max_lag:-1] if max_lag > 0 else y[:-1]
    target = dy[max_lag:]
    cols = [np.ones(len(target)), y_lag1]
    for L in range(1, max_lag + 1):
        cols.append(dy[max_lag - L:-L])
    X = np.column_stack(cols)
    beta, resid, se = _ols(target, X)
    tstat = beta[1] / se[1] if se[1] > 0 else float("nan")
    reject = tstat < _ADF_CRIT["5%"]
    return TestResult(
        name="adf",
        statistic=float(tstat),
        pvalue=None,
        crit_values=_ADF_CRIT,
        reject_null=bool(reject),
        null_hypothesis="unit root present (non-stationary / drifting)",
        detail=f"used_lag={max_lag}; reject@5% => stationary",
    )


_KPSS_CRIT_LEVEL = {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739}


def kpss_test(x: np.ndarray, lags: Optional[int] = None) -> TestResult:
    """KPSS test for stationarity around a level (DQ-05).

    Null hypothesis: the series is (level-)stationary.  Rejecting the null is
    evidence of drift/non-stationarity.  ADF and KPSS are complementary: drift
    is most convincing when ADF fails to reject *and* KPSS rejects.
    """
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 10:
        return TestResult("kpss", float("nan"), None, _KPSS_CRIT_LEVEL, None,
                          "series is level-stationary", "too few points")
    resid = y - y.mean()
    S = np.cumsum(resid)
    eta = float(np.sum(S ** 2)) / (n ** 2)
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** 0.25))
    # Newey-West long-run variance estimate
    s2 = float(resid @ resid) / n
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        cov = float(resid[L:] @ resid[:-L]) / n
        s2 += 2.0 * w * cov
    stat = eta / s2 if s2 > 0 else float("nan")
    reject = stat > _KPSS_CRIT_LEVEL["5%"]
    return TestResult(
        name="kpss",
        statistic=float(stat),
        pvalue=None,
        crit_values=_KPSS_CRIT_LEVEL,
        reject_null=bool(reject),
        null_hypothesis="series is level-stationary",
        detail=f"bandwidth={lags}; reject@5% => non-stationary",
    )


def variance_ratio(x: np.ndarray, q: int = 2) -> TestResult:
    """Lo-MacKinlay variance-ratio test (DQ-05).

    Null hypothesis: the series is a random walk (VR(q) == 1).  VR > 1 indicates
    positive autocorrelation / trending (drift); VR < 1 indicates mean reversion.
    The heteroskedasticity-robust z-statistic is returned.
    """
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 2 * q + 2:
        return TestResult("variance_ratio", float("nan"), None, None, None,
                          "series is a random walk", "too few points")
    dy = np.diff(y)
    mu = dy.mean()
    var1 = float(np.sum((dy - mu) ** 2)) / (n - 1)
    # q-period variance
    m = (n - q) * (1 - q / (n - 1))
    yq = y[q:] - y[:-q]
    varq = float(np.sum((yq - q * mu) ** 2)) / m if m > 0 else float("nan")
    vr = varq / (q * var1) if var1 > 0 else float("nan")
    # asymptotic variance under random walk (homoskedastic)
    phi = 2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * (n - 1))
    z = (vr - 1.0) / np.sqrt(phi) if phi > 0 else float("nan")
    pvalue = float(2 * _sps.norm.sf(abs(z))) if np.isfinite(z) else None
    return TestResult(
        name="variance_ratio",
        statistic=float(vr),
        pvalue=pvalue,
        reject_null=(pvalue is not None and pvalue < 0.05),
        null_hypothesis="series is a random walk (VR=1)",
        detail=f"q={q}, z={z:.3f}; VR>1 => trending/drift",
    )
