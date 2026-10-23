"""Walk-forward economic back-test of the decision pipeline (Table 11 of the paper).

The back-test applies the system exactly as it runs in production, using only
information available at each date:

1. Every RETRAIN_EVERY trading days the direction model (HYBRID_VOTING by default,
   the same model for every instrument) and the ExtraTrees volatility model are
   refitted on all labelled history before the current date, except for the most
   recent CALIB_DAYS rows (plus a purge of k rows), which are held out to fit the
   isotonic PAV calibrator of the direction probability.
2. On each trading date t >= BACKTEST_START_DATE without an open position, the
   calibrated probability p_up and the volatility forecast are passed to the
   RiskManager; a LONG position of the recommended size (10% / 5% / 0% of capital)
   is opened at close[t] and closed at close[t+k].
3. Commission: BACKTEST_FEE_BPS per side. One position per instrument at a time.

The comparison is passive buy-and-hold over exactly the same trading window.

Metrics: total return, CAGR, annualised Sharpe, maximum drawdown, win rate and
number of trades per instrument, and an equal-weight portfolio. The equity curve is
marked to market at every close, so the drawdown includes losses of open positions.

Run:
    python -m jobs.backtest

Parameters (environment):
    TRAIN_START_DATE=2015-01-01   start of the history used for training
    END_DATE=2026-06-01           last date of the sample
    BACKTEST_START_DATE=2022-01-01 first trading date
    BACKTEST_MODEL=HYBRID_VOTING  direction model
    BACKTEST_FEE_BPS=5, BACKTEST_PROB_THRESHOLD=0.55, BACKTEST_RETRAIN_EVERY=21,
    BACKTEST_CALIB_DAYS=252, BACKTEST_INITIAL_CAPITAL=100000, HORIZON_DAYS=5
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from sklearn.isotonic import IsotonicRegression

from jobs.train_baseline import (
    add_technical_features,
    build_feature_matrix,
    compute_targets,
    get_model_by_name,
)
from core.risk.risk_manager import RiskManager

DATABASE_URL = os.environ["DATABASE_URL"]
HORIZON_DAYS = int(os.environ.get("HORIZON_DAYS", "5"))
BACKTEST_START = os.environ.get("BACKTEST_START_DATE", "2022-01-01")
FEE_BPS = float(os.environ.get("BACKTEST_FEE_BPS", "5"))
THRESHOLD = float(os.environ.get("BACKTEST_PROB_THRESHOLD", "0.55"))
INITIAL_CAPITAL = float(os.environ.get("BACKTEST_INITIAL_CAPITAL", "100000"))

WF_MIN_TRAIN = int(os.environ.get("BACKTEST_MIN_TRAIN", "500"))
RETRAIN_EVERY = int(os.environ.get("BACKTEST_RETRAIN_EVERY", "21"))
DATA_START = os.environ.get("TRAIN_START_DATE", "2015-01-01")
END_DATE = os.environ.get("END_DATE", "2026-06-01")         # last date of the paper's sample
MODEL_NAME = os.environ.get("BACKTEST_MODEL", "HYBRID_VOTING")
CALIB_DAYS = int(os.environ.get("BACKTEST_CALIB_DAYS", "252"))

ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

risk_mgr = RiskManager()


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity / np.where(peak == 0, np.nan, peak)) - 1.0
    return float(np.nanmin(dd))


def _annualize_sharpe(daily_returns: np.ndarray) -> float:
    daily = np.asarray(daily_returns, dtype=float)
    daily = daily[np.isfinite(daily)]
    if len(daily) < 2:
        return 0.0
    mu, sigma = float(np.mean(daily)), float(np.std(daily, ddof=1))
    if sigma <= 1e-12:
        return 0.0
    return float((mu / sigma) * np.sqrt(252.0))


def _cagr(initial: float, final: float, n_days: int) -> float:
    if n_days <= 0 or initial <= 0:
        return 0.0
    years = n_days / 365.25
    if years <= 0:
        return 0.0
    return float((final / initial) ** (1.0 / years) - 1.0)


def _backtest_symbol(engine, symbol: str, dir_model_name: str) -> tuple[pd.DataFrame, dict, list]:
    """Walk-forward по одному символу."""
    print(f"\n=== Backtest: {symbol} | model={dir_model_name} ===")

    df = pd.read_sql_query(
        text("""
            SELECT symbol, date,
                   open, high, low, close, volume,
                   return_1d, log_return,
                   sma_5, volatility_5, sma_10, volatility_10, sma_20, volatility_20,
                   return_lag_1, return_lag_2, return_lag_3, return_lag_4, return_lag_5,
                   mkt_return_1d, mkt_log_return, mkt_mom_5, mkt_mom_10, mkt_mom_20, mkt_vol_20,
                   vix_level, vix_return_1d, vix_change_1d,
                   irx_level, irx_change_1d, tnx_level, tnx_change_1d
            FROM features_daily
            WHERE symbol = :symbol AND date >= :start AND date <= :end
            ORDER BY date
        """),
        con=engine, params={"symbol": symbol, "start": DATA_START, "end": END_DATE},
        parse_dates=["date"],
    )
    if df.empty:
        print(f"[WARN] {symbol}: no data from {DATA_START}")
        return pd.DataFrame(), {}, []

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume", "return_1d", "log_return"]).reset_index(drop=True)
    df = add_technical_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    df2, feature_cols = build_feature_matrix(df)
    if df2.empty or not feature_cols:
        print(f"[WARN] {symbol}: пусто после фич")
        return pd.DataFrame(), {}, []

    n = len(df2)
    dates = pd.to_datetime(df2["date"])
    first = int(np.searchsorted(dates.values, np.datetime64(pd.to_datetime(BACKTEST_START))))
    first = max(first, WF_MIN_TRAIN + CALIB_DAYS)
    if first >= n - HORIZON_DAYS:
        print(f"[WARN] {symbol}: too few rows ({n}) for a back-test from {BACKTEST_START}")
        return pd.DataFrame(), {}, []

    fee = FEE_BPS / 10000.0
    equity = INITIAL_CAPITAL
    position_size = 0.0
    entry_price = None
    entry_date = None
    exit_at_idx = None

    trades = []
    eq_curve = []

    model = vol_model = iso = None
    last_train_idx = -10**9

    for i in range(first, n):
        today = df2.iloc[i]
        today_close = float(today["close"])
        today_date = pd.to_datetime(today["date"]).date()

        # === Exit ===
        if exit_at_idx is not None and i >= exit_at_idx:
            exit_price = today_close
            ret = (exit_price / entry_price) - 1.0
            gross_pnl = position_size * equity * ret
            commissions = position_size * equity * fee * 2.0
            net_pnl = gross_pnl - commissions
            equity_after = equity + net_pnl
            trades.append({
                "symbol": symbol,
                "entry_date": str(entry_date),
                "exit_date": str(today_date),
                "entry_price": round(float(entry_price), 4),
                "exit_price": round(float(exit_price), 4),
                "position_pct": round(position_size, 4),
                "ret_pct": round(ret * 100, 4),
                "pnl": round(net_pnl, 2),
                "equity_after": round(equity_after, 2),
                "holding_days": HORIZON_DAYS,
            })
            equity = equity_after
            position_size = 0.0
            entry_price = None
            entry_date = None
            exit_at_idx = None

        # === Signal ===
        if position_size == 0.0:
            # Refit every RETRAIN_EVERY days. compute_targets drops the last k rows of
            # the history, so every label used here is resolved before day i.
            if i - last_train_idx >= RETRAIN_EVERY or model is None:
                lab = compute_targets(df2.iloc[:i].copy(), HORIZON_DAYS)
                fit = lab.iloc[:len(lab) - CALIB_DAYS - HORIZON_DAYS]   # purge k rows before calibration
                cal = lab.iloc[len(lab) - CALIB_DAYS:]
                try:
                    model, _ = get_model_by_name("direction", dir_model_name)
                    model.fit(fit[feature_cols].values, fit["target_direction"].values)
                    p_cal_block = model.predict_proba(cal[feature_cols].values)[:, 1]
                    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
                        p_cal_block, cal["target_direction"].values)
                    vol_model, _ = get_model_by_name("volatility", "EXTRATREES")
                    vol_model.fit(fit[feature_cols].values, fit["target_vol_kd"].values)
                    last_train_idx = i
                except Exception as ex:
                    print(f"[WARN] {symbol}@{today_date}: fit failed: {ex}")
                    eq_curve.append({"date": str(today_date), "equity": equity})
                    continue

            X_now = df2.iloc[[i]][feature_cols].values
            p_up = float(iso.predict(model.predict_proba(X_now)[:, 1])[0])
            vol_hat = max(0.0, float(vol_model.predict(X_now)[0]))

            if p_up >= THRESHOLD:
                pos_pct = risk_mgr.position_size_pct(p_up, vol_hat)
                if pos_pct > 0:
                    position_size = pos_pct
                    entry_price = today_close
                    entry_date = today_date
                    exit_at_idx = i + HORIZON_DAYS

        # equity marked to market at today's close (entry commission already paid)
        if position_size > 0.0 and entry_price is not None:
            mtm = equity + position_size * equity * (today_close / entry_price - 1.0 - fee)
        else:
            mtm = equity
        eq_curve.append({"date": str(today_date), "equity": mtm})

    # Закрываем висящую позицию
    if position_size > 0 and entry_price is not None:
        last_close = float(df2.iloc[-1]["close"])
        last_date = pd.to_datetime(df2.iloc[-1]["date"]).date()
        ret = (last_close / entry_price) - 1.0
        gross_pnl = position_size * equity * ret
        commissions = position_size * equity * fee * 2.0
        net_pnl = gross_pnl - commissions
        equity += net_pnl
        trades.append({
            "symbol": symbol,
            "entry_date": str(entry_date),
            "exit_date": str(last_date),
            "entry_price": round(float(entry_price), 4),
            "exit_price": round(float(last_close), 4),
            "position_pct": round(position_size, 4),
            "ret_pct": round(ret * 100, 4),
            "pnl": round(net_pnl, 2),
            "equity_after": round(equity, 2),
            "holding_days": "open->close",
        })
        if eq_curve:                                   # the last point carries the realised result
            eq_curve[-1]["equity"] = equity

    eq_df = pd.DataFrame(eq_curve)
    if eq_df.empty:
        return pd.DataFrame(), {}, trades
    eq_df["symbol"] = symbol

    # === Метрики ===
    eq_arr = eq_df["equity"].astype(float).values
    eq_df["daily_ret"] = pd.Series(eq_arr).pct_change().fillna(0.0).values
    sharpe = _annualize_sharpe(eq_df["daily_ret"].values)
    mdd = _max_drawdown(eq_arr)
    total_ret = float(eq_arr[-1] / INITIAL_CAPITAL - 1.0)
    n_days = (pd.to_datetime(eq_df["date"].iloc[-1]) - pd.to_datetime(eq_df["date"].iloc[0])).days
    cagr = _cagr(INITIAL_CAPITAL, eq_arr[-1], n_days)

    trades_df = pd.DataFrame(trades)
    win_rate = float((trades_df["ret_pct"] > 0).mean() * 100) if not trades_df.empty else 0.0
    n_trades = int(len(trades_df))
    avg_ret_per_trade = float(trades_df["ret_pct"].mean()) if n_trades else 0.0

    # Buy & Hold
    bh_close = df2["close"].values[first:]
    bh_first = float(bh_close[0])
    bh_last = float(bh_close[-1])
    bh_total = bh_last / bh_first - 1.0 if bh_first > 0 else float("nan")
    bh_cagr = _cagr(bh_first, bh_last, n_days) if bh_first > 0 else float("nan")

    summary = {
        "symbol": symbol,
        "model": dir_model_name,
        "threshold": THRESHOLD,
        "fee_bps": FEE_BPS,
        "horizon_days": HORIZON_DAYS,
        "period_days": n_days,
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate, 2),
        "avg_ret_per_trade_pct": round(avg_ret_per_trade, 4),
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(mdd * 100, 2),
        "final_equity": round(float(eq_arr[-1]), 2),
        "buy_hold_total_pct": round(bh_total * 100, 2) if np.isfinite(bh_total) else None,
        "buy_hold_cagr_pct": round(bh_cagr * 100, 2) if np.isfinite(bh_cagr) else None,
    }

    print(
        f"[BT] {symbol}: trades={n_trades} winrate={win_rate:.1f}% "
        f"total={total_ret*100:.2f}% CAGR={cagr*100:.2f}% Sharpe={sharpe:.2f} MDD={mdd*100:.2f}%"
    )
    return eq_df, summary, trades


def main():
    print("🚀 Walk-forward backtest")
    print(f"   period from: {BACKTEST_START}")
    print(f"   threshold:   {THRESHOLD}")
    print(f"   fee:         {FEE_BPS} bps per side")
    print(f"   horizon:     {HORIZON_DAYS} days")
    print(f"   capital:     ${INITIAL_CAPITAL:,.0f}")
    print(f"   retrain ev.: {RETRAIN_EVERY} days")
    print(f"   model:       {MODEL_NAME} + isotonic calibration on the last {CALIB_DAYS} labelled days")

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    only_symbol = (os.environ.get("ONLY_SYMBOL") or "").strip()
    if only_symbol:
        symbols = [only_symbol]
    else:
        symbols = pd.read_sql_query(
            "SELECT DISTINCT symbol FROM features_daily "
            "WHERE symbol NOT IN ('^VIX','^IRX','^TNX') ORDER BY symbol",
            con=engine,
        )["symbol"].tolist()

    all_eq = []
    summaries = []
    all_trades = []
    for sym in symbols:
        # the same model for every instrument: choosing a model per instrument from
        # the walk-forward test results would select on the evaluation period
        eq_df, summary, trades = _backtest_symbol(engine, sym, MODEL_NAME)
        if not eq_df.empty:
            all_eq.append(eq_df)
            summaries.append(summary)
        if trades:
            all_trades.extend(trades)

    if not summaries:
        print("❌ Нет результатов бэктеста.")
        return

    summary_df = pd.DataFrame(summaries)
    eq_df = pd.concat(all_eq, ignore_index=True)
    trades_df = pd.DataFrame(all_trades)

    summary_path = ARTIFACTS / "backtest_summary.csv"
    eq_path = ARTIFACTS / "backtest_equity.csv"
    trades_path = ARTIFACTS / "backtest_trades.csv"
    summary_df.to_csv(summary_path, index=False)
    eq_df.to_csv(eq_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    print("\n" + "=" * 100)
    print("📊 СВОДКА БЭКТЕСТА")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)

    # Портфельная кривая (равные веса)
    port = (
        eq_df.pivot_table(index="date", columns="symbol", values="equity", aggfunc="last")
        .ffill()
        .dropna(how="all")
    )
    port_norm = port / port.iloc[0] * INITIAL_CAPITAL
    port_norm["PORTFOLIO"] = port_norm.mean(axis=1)
    port_norm.reset_index().to_csv(ARTIFACTS / "backtest_portfolio.csv", index=False)

    port_eq = port_norm["PORTFOLIO"].astype(float).values
    port_daily = pd.Series(port_eq).pct_change().fillna(0.0).values
    port_sharpe = _annualize_sharpe(port_daily)
    port_mdd = _max_drawdown(port_eq)
    port_total = float(port_eq[-1] / port_eq[0] - 1.0)
    port_days = (pd.to_datetime(port_norm.index[-1]) - pd.to_datetime(port_norm.index[0])).days
    port_cagr = _cagr(port_eq[0], port_eq[-1], port_days)

    print("\n📦 ПОРТФЕЛЬ (равные веса по инструментам):")
    print(f"  Total return:  {port_total*100:.2f}%")
    print(f"  CAGR:          {port_cagr*100:.2f}%")
    print(f"  Sharpe:        {port_sharpe:.2f}")
    print(f"  Max Drawdown:  {port_mdd*100:.2f}%")

    print(f"\n✅ Saved: {summary_path}")
    print(f"✅ Saved: {eq_path}")
    print(f"✅ Saved: {trades_path}")


if __name__ == "__main__":
    main()
