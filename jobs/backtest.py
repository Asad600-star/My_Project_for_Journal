"""Полноценный walk-forward бэктест.

Стратегия:
1. На каждой исторической дате t (после WF_MIN_TRAIN) обучаем direction-модель
   на [start, t]. Если AUC модели в реестре < 0.5 — используем LOGREG fallback.
2. Берём предсказание p_up для t. Если p_up >= threshold → открываем LONG позицию
   на k=HORIZON_DAYS дней (вход по close[t], выход по close[t+k]).
3. Размер позиции: из RiskManager.position_size_pct (10% / 5% / 0%).
4. Комиссия: BACKTEST_FEE_BPS (по умолчанию 5 bps на каждую сторону).
5. Одновременно одна позиция на инструмент (новые сигналы игнорируются пока активна старая).

Метрики:
- Total return, CAGR, Sharpe (annualized), Max Drawdown, Win-rate, # trades.
- Сравнение с buy & hold для каждого инструмента.

Запуск:
    python -m jobs.backtest

Параметры (через .env):
    BACKTEST_START_DATE=2022-01-01
    BACKTEST_FEE_BPS=5
    BACKTEST_PROB_THRESHOLD=0.55
    BACKTEST_INITIAL_CAPITAL=100000
    BACKTEST_RETRAIN_EVERY=21
    HORIZON_DAYS=5
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

from jobs.train_baseline import (
    add_technical_features,
    build_feature_matrix,
    compute_targets,
    get_model_by_name,
    select_best_models,
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
            WHERE symbol = :symbol AND date >= :start
            ORDER BY date
        """),
        con=engine, params={"symbol": symbol, "start": BACKTEST_START},
        parse_dates=["date"],
    )
    if df.empty:
        print(f"[WARN] {symbol}: нет данных с {BACKTEST_START}")
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
    if n < WF_MIN_TRAIN + HORIZON_DAYS + 5:
        print(f"[WARN] {symbol}: мало строк ({n})")
        return pd.DataFrame(), {}, []

    fee = FEE_BPS / 10000.0
    equity = INITIAL_CAPITAL
    position_size = 0.0
    entry_price = None
    entry_date = None
    exit_at_idx = None

    trades = []
    eq_curve = []

    model = None
    last_train_idx = -10**9

    for i in range(WF_MIN_TRAIN, n):
        today = df2.iloc[i]
        today_close = float(today["close"])
        today_date = pd.to_datetime(today["date"]).date()

        # === Выход из позиции ===
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

        # === Сигнал на сегодня ===
        if position_size == 0.0:
            # Переобучение
            if i - last_train_idx >= RETRAIN_EVERY or model is None:
                tr = df2.iloc[:i].copy()
                tr_lab = compute_targets(tr, HORIZON_DAYS)
                if len(tr_lab) < 200:
                    eq_curve.append({"date": str(today_date), "equity": equity})
                    continue

                X_tr = tr_lab[feature_cols].values
                y_tr = tr_lab["target_direction"].values

                try:
                    model_obj, _ = get_model_by_name("direction", dir_model_name)
                    model_obj.fit(X_tr, y_tr)
                    model = model_obj
                    last_train_idx = i
                except Exception as e:
                    print(f"[WARN] {symbol}@{today_date}: fit failed: {e}")
                    eq_curve.append({"date": str(today_date), "equity": equity})
                    continue

            # Прогноз на текущей строке
            X_now = df2.iloc[[i]][feature_cols].values
            try:
                if hasattr(model, "predict_proba"):
                    p_up = float(model.predict_proba(X_now)[:, 1][0])
                else:
                    s = float(model.decision_function(X_now)[0])
                    p_up = 1.0 / (1.0 + np.exp(-s))
            except Exception as e:
                print(f"[WARN] {symbol}@{today_date}: predict failed: {e}")
                p_up = 0.5

            # Открываем сделку
            if p_up >= THRESHOLD:
                last_vol = float(df2.iloc[i].get("volatility_5", 0.02) or 0.02)
                pos_pct = risk_mgr.position_size_pct(p_up, last_vol)
                if pos_pct > 0:
                    position_size = pos_pct
                    entry_price = today_close
                    entry_date = today_date
                    exit_at_idx = i + HORIZON_DAYS

        eq_curve.append({"date": str(today_date), "equity": equity})

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
    bh_close = df2["close"].values
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

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    selected = select_best_models(HORIZON_DAYS)
    direction_models = selected.get("direction", {})

    only_symbol = (os.environ.get("ONLY_SYMBOL") or "").strip()
    if only_symbol:
        symbols = [only_symbol]
    else:
        symbols = pd.read_sql_query(
            "SELECT DISTINCT symbol FROM features_daily "
            "WHERE symbol NOT IN ('^VIX','^IRX','^TNX') ORDER BY symbol",
            con=engine,
        )["symbol"].tolist()

    # AUC из реестра — для fallback на LOGREG если AUC<0.5
    reg_path = ARTIFACTS / f"model_registry_k{HORIZON_DAYS}.csv"
    auc_map = {}
    if reg_path.exists():
        reg = pd.read_csv(reg_path)
        rd = reg[reg["task"] == "direction"]
        for _, r in rd.iterrows():
            try:
                auc_map[r["symbol"]] = float(r["metric_value"])
            except Exception:
                pass

    all_eq = []
    summaries = []
    all_trades = []
    for sym in symbols:
        chosen = direction_models.get(sym, "LOGREG")
        if auc_map.get(sym, 1.0) < 0.50:
            print(f"[WARN] {sym}: registry AUC={auc_map[sym]:.3f} < 0.5 → fallback на LOGREG")
            chosen = "LOGREG"

        eq_df, summary, trades = _backtest_symbol(engine, sym, chosen)
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
