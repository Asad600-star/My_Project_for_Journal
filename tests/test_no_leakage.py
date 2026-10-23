"""Causality of the features and of the econometric forecasts, and purging of the targets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from jobs.econometric_baselines import garch_daily, har_features
from jobs.train_baseline import add_technical_features, compute_targets

ENGINEERED = ["hl_range", "oc_return", "mom_5", "mom_10", "mom_20", "rsi_14", "ema_12", "ema_26",
              "macd", "macd_signal", "macd_hist", "bb_width_20", "atr_14", "atrp_14", "vol_z_20",
              "ret_std_20", "ret_mean_20", "corr_mkt_60", "beta_mkt_60", "mkt_trend_20",
              "mkt_risk_20", "vix_log", "vix_z_60", "vix_x_mktret", "yc_slope"]


def test_features_do_not_use_future_rows(synthetic_panel):
    """A feature at day t must not change when every row after t is removed."""
    full = add_technical_features(synthetic_panel)
    for cut in (120, 250, len(synthetic_panel) - 1):
        part = add_technical_features(synthetic_panel.iloc[:cut].copy())
        for col in ENGINEERED:
            a, b = full[col].iloc[:cut].to_numpy(), part[col].to_numpy()
            assert np.allclose(a, b, equal_nan=True), f"{col} uses data after row {cut}"


def test_volatility_target_is_forward_realised_std(synthetic_panel):
    k = 5
    out = compute_targets(synthetic_panel, k)
    lr = synthetic_panel["log_return"].to_numpy()
    for t in (0, 37, len(out) - 1):
        assert np.isclose(out["target_vol_kd"].iloc[t], np.std(lr[t + 1:t + 1 + k]))


def test_targets_are_purged_inside_each_split(synthetic_panel):
    """The last k rows of a block have no complete label and are dropped, so a
    label never reaches into the next block of the walk-forward split."""
    k = 5
    block = synthetic_panel.iloc[:200]
    out = compute_targets(block, k)
    assert len(out) == len(block) - k
    assert out["date"].max() == block["date"].iloc[-k - 1]


def test_garch_forecast_uses_only_past_returns(synthetic_panel):
    r = synthetic_panel["log_return"].to_numpy()
    n_train = 300
    full = garch_daily(r[:n_train], r, n_train)
    for cut in (310, 350, len(r)):
        part = garch_daily(r[:n_train], r[:cut], n_train)
        assert np.allclose(full[:cut], part)


def test_har_features_are_backward_looking(synthetic_panel):
    full = har_features(synthetic_panel)
    part = har_features(synthetic_panel.iloc[:200])
    assert np.allclose(full[:200], part, equal_nan=True)


def test_gjr_forecast_uses_only_past_returns(synthetic_panel):
    from jobs.volatility_models import gjr_daily
    r = synthetic_panel["log_return"].to_numpy()
    n_train = 300
    full = gjr_daily(r[:n_train], r, n_train)
    for cut in (320, len(r)):
        assert np.allclose(full[:cut], gjr_daily(r[:n_train], r[:cut], n_train))


def test_range_features_are_backward_looking(synthetic_panel):
    from jobs.volatility_models import range_features
    full = range_features(synthetic_panel)
    part = range_features(synthetic_panel.iloc[:150])
    assert np.allclose(full[:150], part, equal_nan=True)
    assert np.all(full[np.isfinite(full)] >= 0)


def test_deep_windows_end_at_the_decision_day_and_cover_every_row():
    from jobs.sequence_windows import make_sequences
    X = np.arange(40, dtype=float).reshape(-1, 1)            # feature value = row index
    pos = np.arange(25, 35)                                   # e.g. a test block
    Xs, ys = make_sequences(X, pos, np.arange(10), seq_len=5)
    assert len(Xs) == len(pos)                                # every evaluation row is scored
    assert np.all(Xs[:, -1, 0] == pos)                        # the window ends at the decision day
    assert np.all(Xs[:, 0, 0] == pos - 4)                     # and only looks back
    Xtr, ytr = make_sequences(X, np.arange(10), np.arange(10), seq_len=5)
    assert len(Xtr) == 6 and ytr[0] == 4                      # training rows without a full window are skipped


def test_feature_builder_keeps_the_last_day_and_is_causal(synthetic_panel):
    from jobs.build_features import build_features_for_symbol
    raw = synthetic_panel[["date", "open", "high", "low", "close", "volume"]]
    full = build_features_for_symbol(raw, "X")
    assert full["date"].iloc[-1] == raw["date"].iloc[-1]       # the latest day is available for inference
    part = build_features_for_symbol(raw.iloc[:200], "X")
    cols = [c for c in full.columns if c not in ("target_return_1d", "symbol", "date")]
    n = len(part) - 1                                          # last row of the truncated frame lacks only the target
    assert np.allclose(full[cols].iloc[:n + 1].to_numpy(float), part[cols].to_numpy(float), equal_nan=True)


def test_chronological_blocks_are_purged():
    from jobs.racc import block
    dates = pd.bdate_range("2022-12-01", periods=60)
    data = pd.DataFrame({"date": dates, "label_end": pd.Series(dates).shift(-5)}).dropna()
    tr = block(data, upto="2022-12-30")
    ca = block(data, after="2022-12-30", upto="2023-01-31")
    te = block(data, after="2023-01-31", purge=False)
    assert tr["label_end"].max() <= pd.Timestamp("2022-12-30")     # no training label reaches the next block
    assert ca["label_end"].max() <= pd.Timestamp("2023-01-31")
    assert ca["date"].min() > tr["date"].max() and te["date"].min() > ca["date"].max()
    assert len(block(data, upto="2022-12-30", purge=False)) - len(tr) == 5   # exactly k rows are purged
