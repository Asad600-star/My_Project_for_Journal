"""Shared fixtures: a synthetic daily panel with the columns of features_daily."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """Price, volume and macro columns for 400 business days (GARCH-like returns)."""
    rng = np.random.default_rng(0)
    n = 400
    r = np.empty(n)
    s2 = 1e-4
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = 2e-6 + 0.08 * r[t] ** 2 + 0.9 * s2
    close = 100.0 * np.exp(np.cumsum(r))
    open_ = close * np.exp(0.002 * rng.standard_normal(n))
    df = pd.DataFrame({
        "date": pd.bdate_range("2018-01-01", periods=n),
        "open": open_,
        "high": np.maximum(open_, close) * (1 + 0.004 * rng.random(n)),
        "low": np.minimum(open_, close) * (1 - 0.004 * rng.random(n)),
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    })
    df["return_1d"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()
    m = 0.8 * df["return_1d"] + 0.005 * rng.standard_normal(n)
    df["mkt_return_1d"] = m
    df["mkt_mom_20"] = m.rolling(20).sum()
    df["mkt_vol_20"] = m.rolling(20).std()
    df["vix_level"] = 18 + 5 * np.abs(rng.standard_normal(n))
    df["vix_return_1d"] = df["vix_level"].pct_change()
    df["irx_level"] = 2 + 0.01 * np.cumsum(rng.standard_normal(n))
    df["tnx_level"] = 3 + 0.01 * np.cumsum(rng.standard_normal(n))
    return df.iloc[1:].reset_index(drop=True)
