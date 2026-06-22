"""
Ablation Study — какие компоненты системы реально вносят вклад?
================================================================

Для Q1 reviewer вопрос: "Все ли эти компоненты нужны?"
Ablation study убирает по очереди каждую группу признаков и показывает,
как падает качество прогноза.

Тестируется ExtraTrees (главный регрессор) с разными подмножествами признаков:

    1. PRICE_ONLY:        5 признаков    (только OHLCV)
    2. PRICE+TECHNICAL:   29 признаков    (+24 технических)
    3. PRICE+TECH+LAG:    34 признака     (+5 лагов)
    4. ALL_NO_MACRO:      42 признака     (без 14 макро)
    5. ALL_NO_REGIME:     48 признаков    (без 8 regime)
    6. ALL_FULL:          56 признаков    ← полная система

Ожидаемый результат для статьи:
- Чем больше групп признаков, тем лучше прогноз
- Каждая группа даёт значимый вклад

Запуск:
    python -m jobs.ablation_study
"""
from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")
load_dotenv()


# Конфигурация (синхронизировано с LSTM/ARIMA)
SEED = int(os.environ.get("SEED", "42"))
END_DATE = os.environ.get("END_DATE", "2026-06-01")
HORIZON_DAYS = int(os.environ.get("HORIZON_DAYS", "5"))

WF_MIN_TRAIN_ROWS = int(os.environ.get("WF_MIN_TRAIN_ROWS", "1200"))
WF_VAL_DAYS = int(os.environ.get("WF_VAL_DAYS", "126"))
WF_TEST_DAYS = int(os.environ.get("WF_TEST_DAYS", "126"))
WF_STEP_DAYS = int(os.environ.get("WF_STEP_DAYS", "63"))

# Те же 8 символов что в LSTM
SYMBOLS_DEFAULT = ["AAPL", "TSLA", "^GSPC", "^IXIC", "^DJI", "^RUT", "GLD", "MSFT"]
SYMBOLS_ENV = os.environ.get("SYMBOLS", "")
SYMBOLS = (
    [s.strip() for s in SYMBOLS_ENV.split(",") if s.strip()]
    if SYMBOLS_ENV
    else SYMBOLS_DEFAULT
)

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = ARTIFACTS_DIR / "ablation_study_results.csv"


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────────────
#  Признаки разбиваем на 5 групп
# ─────────────────────────────────────────────────────────────────────
FEATURE_GROUPS = {
    "price": [
        "open", "high", "low", "close", "volume",
    ],
    "technical": [
        "log_return", "simple_return",
        "ema_5", "ema_10", "ema_26", "ema_50", "ema_5_minus_26",
        "sma_5", "sma_20",
        "volatility_5", "volatility_10", "volatility_20",
        "rsi_14",
        "macd_line", "macd_signal", "macd_hist",
        "bb_mean_20", "bb_std_20", "bb_width_20",
        "hl_range", "oc_return",
        "mom_5", "mom_10", "mom_20",
        "atr_14", "atrp_14", "vol_z_20", "ret_std_20",
    ],
    "lag": [
        "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_4", "return_lag_5",
    ],
    "macro": [
        "vix_close", "irx_level", "tnx_level",
        "vix_return", "vix_ma_20", "vix_z_60",
        "mkt_return_1d", "yc_slope", "vix_x_mktret",
        "tnx_change", "irx_change",
        "tnx_z_60", "irx_z_60",
    ],
    "regime": [
        "mkt_mom_5", "mkt_mom_10", "mkt_mom_20",
        "mkt_vol_20",
        "mkt_trend_5", "mkt_trend_20", "mkt_risk_20",
        "corr_mkt_60", "beta_mkt_60",
    ],
}

# Ablation сценарии (какие группы оставить)
ABLATION_SCENARIOS = {
    "PRICE_ONLY": ["price"],
    "PRICE+TECH": ["price", "technical"],
    "PRICE+TECH+LAG": ["price", "technical", "lag"],
    "ALL_NO_MACRO": ["price", "technical", "lag", "regime"],
    "ALL_NO_REGIME": ["price", "technical", "lag", "macro"],
    "ALL_FULL": ["price", "technical", "lag", "macro", "regime"],
}


# ─────────────────────────────────────────────────────────────────────
#  Загрузка + построение признаков (копия LSTM скрипта)
# ─────────────────────────────────────────────────────────────────────
def get_engine():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL env var не задан")
    return create_engine(db_url, pool_pre_ping=True)


def load_ohlcv(engine, symbol: str) -> pd.DataFrame:
    q = text("""
        SELECT date, open, high, low, close, adj_close, volume
        FROM market_ohlcv
        WHERE symbol = :sym AND date <= :end_date
        ORDER BY date ASC
    """)
    df = pd.read_sql(q, engine, params={"sym": symbol, "end_date": END_DATE})
    df["date"] = pd.to_datetime(df["date"])
    return df


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    ag = gain.ewm(alpha=1 / period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = ag / (al + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["simple_return"] = df["close"].pct_change()
    df["ema_5"] = _ema(df["close"], 5)
    df["ema_10"] = _ema(df["close"], 10)
    df["ema_26"] = _ema(df["close"], 26)
    df["ema_50"] = _ema(df["close"], 50)
    df["ema_5_minus_26"] = df["ema_5"] - df["ema_26"]
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["volatility_5"] = df["simple_return"].rolling(5).std(ddof=0)
    df["volatility_10"] = df["simple_return"].rolling(10).std(ddof=0)
    df["volatility_20"] = df["simple_return"].rolling(20).std(ddof=0)
    df["rsi_14"] = _rsi(df["close"], 14)
    df["macd_line"] = _ema(df["close"], 12) - _ema(df["close"], 26)
    df["macd_signal"] = _ema(df["macd_line"], 9)
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    df["bb_mean_20"] = df["close"].rolling(20).mean()
    df["bb_std_20"] = df["close"].rolling(20).std(ddof=0)
    df["bb_width_20"] = 2 * df["bb_std_20"] / (df["bb_mean_20"] + 1e-12)
    df["hl_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-12)
    df["oc_return"] = (df["close"] - df["open"]) / (df["open"] + 1e-12)
    df["mom_5"] = df["close"].pct_change(5)
    df["mom_10"] = df["close"].pct_change(10)
    df["mom_20"] = df["close"].pct_change(20)
    df["atr_14"] = _atr(df, 14)
    df["atrp_14"] = df["atr_14"] / (df["close"] + 1e-12)
    df["vol_z_20"] = (df["volume"] - df["volume"].rolling(20).mean()) / (
        df["volume"].rolling(20).std(ddof=0) + 1e-12
    )
    df["ret_std_20"] = df["simple_return"].rolling(20).std(ddof=0)
    for k in range(1, 6):
        df[f"return_lag_{k}"] = df["simple_return"].shift(k)
    return df


def add_macro_and_regime_features(df: pd.DataFrame, engine) -> pd.DataFrame:
    df = df.copy()
    macro_q = text("""
        SELECT date, symbol, close
        FROM market_ohlcv
        WHERE symbol IN ('^VIX', '^IRX', '^TNX') AND date <= :end_date
        ORDER BY date ASC
    """)
    macro = pd.read_sql(macro_q, engine, params={"end_date": END_DATE})
    macro["date"] = pd.to_datetime(macro["date"])
    macro_pivot = macro.pivot(index="date", columns="symbol", values="close")
    macro_pivot.columns = ["vix_close", "irx_level", "tnx_level"]
    macro_pivot = macro_pivot.reset_index()
    df = df.merge(macro_pivot, on="date", how="left")

    df["vix_return"] = df["vix_close"].pct_change()
    df["vix_ma_20"] = df["vix_close"].rolling(20).mean()
    df["vix_z_60"] = (df["vix_close"] - df["vix_close"].rolling(60).mean()) / (
        df["vix_close"].rolling(60).std(ddof=0) + 1e-12
    )
    df["mkt_return_1d"] = df["simple_return"]
    df["yc_slope"] = df["tnx_level"] - df["irx_level"]
    df["vix_x_mktret"] = df["vix_close"] * df["mkt_return_1d"]
    df["tnx_change"] = df["tnx_level"].pct_change()
    df["irx_change"] = df["irx_level"].pct_change()
    df["tnx_z_60"] = (df["tnx_level"] - df["tnx_level"].rolling(60).mean()) / (
        df["tnx_level"].rolling(60).std(ddof=0) + 1e-12
    )
    df["irx_z_60"] = (df["irx_level"] - df["irx_level"].rolling(60).mean()) / (
        df["irx_level"].rolling(60).std(ddof=0) + 1e-12
    )

    df["mkt_mom_5"] = df["close"].pct_change(5)
    df["mkt_mom_10"] = df["close"].pct_change(10)
    df["mkt_mom_20"] = df["close"].pct_change(20)
    df["mkt_vol_20"] = df["simple_return"].rolling(20).std(ddof=0)
    df["mkt_trend_5"] = df["mkt_mom_5"]
    df["mkt_trend_20"] = df["mkt_mom_20"]
    df["mkt_risk_20"] = df["mkt_vol_20"]
    df["corr_mkt_60"] = df["simple_return"].rolling(60).corr(df["mkt_return_1d"])
    df["beta_mkt_60"] = (
        df["simple_return"].rolling(60).cov(df["mkt_return_1d"])
        / (df["mkt_return_1d"].rolling(60).var(ddof=0) + 1e-12)
    )
    return df


def compute_target_vol(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy()
    fut = pd.concat(
        [df["simple_return"].shift(-i) for i in range(1, horizon + 1)],
        axis=1,
    )
    df["target_vol_kd"] = fut.std(axis=1, ddof=0)
    return df


# ─────────────────────────────────────────────────────────────────────
#  Walk-forward для одного символа × одного сценария
# ─────────────────────────────────────────────────────────────────────
@dataclass
class AblationRow:
    symbol: str
    scenario: str
    n_features: int
    fold: int
    split: str
    n_rows: int
    mae: float
    rmse: float
    r2: float


def get_features_for_scenario(scenario: str) -> list[str]:
    """Возвращает список признаков для данного ablation сценария."""
    groups = ABLATION_SCENARIOS[scenario]
    cols = []
    for g in groups:
        cols.extend(FEATURE_GROUPS[g])
    return cols


def run_ablation_for_symbol(engine, symbol: str) -> list[AblationRow]:
    """Walk-forward ablation для одного символа со всеми сценариями."""
    print(f"\n[ABLATION] === {symbol} ===")
    df = load_ohlcv(engine, symbol)
    if len(df) < WF_MIN_TRAIN_ROWS + WF_VAL_DAYS + WF_TEST_DAYS:
        print(f"[ABLATION] {symbol}: недостаточно данных, пропускаем")
        return []

    df = add_technical_features(df)
    df = add_macro_and_regime_features(df, engine)
    df = df.dropna().reset_index(drop=True)

    n = len(df)
    start_idx = WF_MIN_TRAIN_ROWS
    last_start = n - (WF_VAL_DAYS + WF_TEST_DAYS)

    rows: list[AblationRow] = []
    fold = 0
    i = start_idx

    while i <= last_start:
        tr = df.iloc[:i].copy()
        va = df.iloc[i : i + WF_VAL_DAYS].copy()
        te = df.iloc[i + WF_VAL_DAYS : i + WF_VAL_DAYS + WF_TEST_DAYS].copy()

        tr = compute_target_vol(tr, HORIZON_DAYS).dropna(subset=["target_vol_kd"])
        va = compute_target_vol(va, HORIZON_DAYS).dropna(subset=["target_vol_kd"])
        te = compute_target_vol(te, HORIZON_DAYS).dropna(subset=["target_vol_kd"])

        if len(tr) < 500 or len(va) < 80 or len(te) < 80:
            i += WF_STEP_DAYS
            continue

        y_tr = tr["target_vol_kd"].values
        y_te = te["target_vol_kd"].values

        # Для каждого ablation сценария обучаем модель
        for scenario in ABLATION_SCENARIOS:
            feature_cols = get_features_for_scenario(scenario)
            # Только те колонки, которые реально есть
            feature_cols = [c for c in feature_cols if c in df.columns]
            X_tr = tr[feature_cols].values.astype(np.float32)
            X_te = te[feature_cols].values.astype(np.float32)

            model = ExtraTreesRegressor(
                n_estimators=500,
                random_state=SEED,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            rows.append(AblationRow(
                symbol=symbol,
                scenario=scenario,
                n_features=len(feature_cols),
                fold=fold,
                split="test",
                n_rows=len(y_te),
                mae=float(mean_absolute_error(y_te, y_pred)),
                rmse=float(np.sqrt(mean_squared_error(y_te, y_pred))),
                r2=float(r2_score(y_te, y_pred)),
            ))

        print(f"[ABLATION] {symbol} fold={fold}: done all scenarios")
        fold += 1
        i += WF_STEP_DAYS

    return rows


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print(" ABLATION STUDY — какие группы признаков нужны?")
    print("=" * 70)
    print(f" Модель:    ExtraTrees (главный регрессор для volatility)")
    print(f" Horizon:   k = {HORIZON_DAYS}")
    print(f" Seed:      {SEED}")
    print(f" END_DATE:  {END_DATE}")
    print(f" Symbols:   {SYMBOLS}")
    print(f" Scenarios: {list(ABLATION_SCENARIOS.keys())}")
    print("=" * 70)

    set_seeds(SEED)
    engine = get_engine()

    all_rows: list[AblationRow] = []
    for sym in SYMBOLS:
        try:
            sym_rows = run_ablation_for_symbol(engine, sym)
            all_rows.extend(sym_rows)
        except Exception as e:
            print(f"[ERROR] {sym}: {e}")
            import traceback
            traceback.print_exc()

    if not all_rows:
        print("[WARN] Нет результатов")
        return

    df = pd.DataFrame([asdict(r) for r in all_rows])
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[ARTIFACT] Сохранено: {OUT_CSV} ({len(df)} строк)")

    # Сводка для статьи
    print("\n" + "=" * 70)
    print(" СВОДКА: средний RMSE по сценариям (test split)")
    print("=" * 70)

    summary = (
        df.groupby(["symbol", "scenario", "n_features"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            n_folds=("fold", "nunique"),
        )
        .sort_values(["symbol", "n_features"])
    )
    print(summary.to_string(index=False))

    # Pivot для статьи: символ × сценарий
    print("\n" + "=" * 70)
    print(" PIVOT для статьи (symbol × scenario, mean RMSE)")
    print("=" * 70)
    pivot = df.pivot_table(
        index="symbol",
        columns="scenario",
        values="rmse",
        aggfunc="mean",
    )
    # Упорядочим колонки по числу признаков
    col_order = ["PRICE_ONLY", "PRICE+TECH", "PRICE+TECH+LAG", "ALL_NO_MACRO", "ALL_NO_REGIME", "ALL_FULL"]
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]
    print(pivot.round(6).to_string())

    pivot_path = ARTIFACTS_DIR / "ablation_pivot.csv"
    pivot.to_csv(pivot_path)
    print(f"\n[ARTIFACT] Pivot table сохранён: {pivot_path}")


if __name__ == "__main__":
    main()
