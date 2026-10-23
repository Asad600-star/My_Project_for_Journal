import os
from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime, date as dt_date

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
    StackingRegressor,
)

# === НОВЫЕ СИЛЬНЫЕ МОДЕЛИ ===
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from core.explain.shap_explainer import compute_and_save_shap

load_dotenv()

import sys
# === ИСПРАВЛЕНИЕ ПУТЕЙ ===
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Читаем дату начала обучения из .env (можно легко менять)
TRAIN_START_DATE = os.environ.get("TRAIN_START_DATE", "2022-01-01")
print(f"[CONFIG] Обучаем модель только с даты: {TRAIN_START_DATE}")
# ==========================================================
# Baselines for financial time series (no leakage)
# Supports tasks via env:
#   TASK=return|direction|volatility
# Horizon via env:
#   HORIZON_DAYS=1..N
# Split mode:
#   MODE=single|walk
#     - single: one fixed split using TRAIN_END/VAL_END
#     - walk  : walk-forward evaluation (recommended)
# ==========================================================

TASK = (os.environ.get("TASK", "return") or "return").strip().lower()
HORIZON_DAYS = int(os.environ.get("HORIZON_DAYS", "5"))
if HORIZON_DAYS < 1:
    raise ValueError("HORIZON_DAYS must be >= 1")

MODE = (os.environ.get("MODE", "walk") or "walk").strip().lower()
if MODE not in ("single", "walk"):
    raise ValueError("MODE must be 'single' or 'walk'")

# --------- Production/Inference/Backtest config ---------
ACTION = (os.environ.get("ACTION", "train") or "train").strip().lower()
if ACTION not in ("train", "infer", "backtest"):
    raise ValueError("ACTION must be 'train', 'infer', or 'backtest'")

# Inference/backtest inputs
METRICS_DIR = Path(os.environ.get("METRICS_DIR", "artifacts"))
PRED_OUT = Path(os.environ.get("PRED_OUT", "artifacts/predictions_latest.csv"))
PRED_OUT.parent.mkdir(parents=True, exist_ok=True)

LOG_TARGET = int(os.environ.get("LOG_TARGET", "0"))  # 1 -> train on log1p(y) when y > -1

# Fixed split boundaries (used in MODE=single)
TRAIN_END = (os.environ.get("TRAIN_END", "2022-12-31") or "2022-12-31").strip()
VAL_END = (os.environ.get("VAL_END", "2024-12-31") or "2024-12-31").strip()

# Output
OUT_PATH = Path(
    os.environ.get(
        "BASELINE_OUT",
        f"artifacts/metrics_{MODE}_{TASK}_k{HORIZON_DAYS}.csv",
    )
)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

RETURN_TARGET_MODE = (os.environ.get("RETURN_TARGET_MODE", "log") or "log").strip().lower()
if RETURN_TARGET_MODE not in ("log", "simple"):
    raise ValueError("RETURN_TARGET_MODE must be 'log' or 'simple'")

RANDOM_STATE = int(os.environ.get("SEED", "42"))


# Walk-forward knobs (used in MODE=walk)
WF_MIN_TRAIN_ROWS = int(os.environ.get("WF_MIN_TRAIN_ROWS", "1200"))
WF_VAL_DAYS = int(os.environ.get("WF_VAL_DAYS", "126"))   # ~6 months trading days
WF_TEST_DAYS = int(os.environ.get("WF_TEST_DAYS", "126")) # ~6 months trading days
WF_STEP_DAYS = int(os.environ.get("WF_STEP_DAYS", "63"))  # ~3 months step

# Robustness knobs
Y_CLIP_PCT = float(os.environ.get("Y_CLIP_PCT", "0.0"))  # e.g. 0.01 clips to [1%,99%]
if not (0.0 <= Y_CLIP_PCT < 0.5):
    raise ValueError("Y_CLIP_PCT must be in [0, 0.5)")


def get_env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if v is None or str(v).strip() == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return str(v).strip()


# Helper: convert pandas/py datetime-like to python date
def to_pydate(x) -> dt_date:
    """Convert pandas/py datetime-like value to python `date` safely."""
    if x is None:
        raise ValueError("to_pydate: got None")

    if isinstance(x, pd.Timestamp):
        return x.date()

    if isinstance(x, datetime):
        return x.date()

    if isinstance(x, dt_date):
        return x

    return pd.to_datetime(x).date()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """No-leakage features computed only from past & present."""
    df = df.copy()

    # Basic price/volume derived
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0.0, np.nan)
    df["oc_return"] = (df["close"] - df["open"]) / df["open"].replace(0.0, np.nan)

    # Momentum
    df["mom_5"] = df["close"].pct_change(5)
    df["mom_10"] = df["close"].pct_change(10)
    df["mom_20"] = df["close"].pct_change(20)

    # RSI
    df["rsi_14"] = _rsi(df["close"], 14)

    # EMAs + MACD
    df["ema_12"] = _ema(df["close"], 12)
    df["ema_26"] = _ema(df["close"], 26)
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = _ema(df["macd"], 9)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger width (20)
    mid = df["close"].rolling(20, min_periods=20).mean()
    std = df["close"].rolling(20, min_periods=20).std(ddof=0)
    df["bb_width_20"] = (2.0 * std) / mid.replace(0.0, np.nan)

    # ATR (volatility proxy)
    df["atr_14"] = _atr(df, 14)
    df["atrp_14"] = df["atr_14"] / df["close"].replace(0.0, np.nan)

    # Volume z-score (20)
    v = df["volume"].astype(float)
    v_mean = v.rolling(20, min_periods=20).mean()
    v_std = v.rolling(20, min_periods=20).std(ddof=0)
    df["vol_z_20"] = (v - v_mean) / v_std.replace(0.0, np.nan)

    # Return distribution (20)
    r = df["return_1d"]
    df["ret_std_20"] = r.rolling(20, min_periods=20).std(ddof=0)
    df["ret_mean_20"] = r.rolling(20, min_periods=20).mean()

    # -------- Macro interaction / regime features (use only current & past) --------
    # These require macro/context columns to be present in the incoming df.
    if "mkt_return_1d" in df.columns:
        # Rolling correlation and beta to the market (60 trading days)
        ar = df["return_1d"].astype(float)
        mr = df["mkt_return_1d"].astype(float)
        corr60 = ar.rolling(60, min_periods=60).corr(mr)
        df["corr_mkt_60"] = corr60

        mvar = mr.rolling(60, min_periods=60).var(ddof=0)
        mcov = ar.rolling(60, min_periods=60).cov(mr, ddof=0)
        df["beta_mkt_60"] = mcov / mvar.replace(0.0, np.nan)

        # Simple market regime proxy: trend vs. volatility
        df["mkt_trend_20"] = df.get("mkt_mom_20", np.nan)
        df["mkt_risk_20"] = df.get("mkt_vol_20", np.nan)

    if "vix_level" in df.columns:
        vx = df["vix_level"].astype(float)
        df["vix_log"] = np.log(vx.replace(0.0, np.nan))
        df["vix_z_60"] = (vx - vx.rolling(60, min_periods=60).mean()) / vx.rolling(60, min_periods=60).std(ddof=0).replace(0.0, np.nan)
        # Risk-on / risk-off proxy
        if "mkt_return_1d" in df.columns:
            df["vix_x_mktret"] = df["vix_return_1d"].astype(float) * df["mkt_return_1d"].astype(float)

    if "irx_level" in df.columns and "tnx_level" in df.columns:
        irx = df["irx_level"].astype(float)
        tnx = df["tnx_level"].astype(float)
        # Yield curve slope proxy (10Y - 3M)
        df["yc_slope"] = (tnx - irx)

    return df


def compute_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Targets without leakage."""
    df = df.copy()
    df["log_return"] = df["log_return"].replace([np.inf, -np.inf], np.nan)

    df["target_return_kd"] = df["close"].shift(-horizon) / df["close"] - 1.0
    df["target_logret_kd"] = np.log(df["close"].shift(-horizon) / df["close"]).replace(
        [np.inf, -np.inf], np.nan
    )
    df["target_direction"] = (df["target_return_kd"] >= 0).astype(int)

    # Realized vol over NEXT k days: std(log_return[t+1..t+k])
    fut = [df["log_return"].shift(-i) for i in range(1, horizon + 1)]
    tmp = pd.concat(fut, axis=1)
    df["target_vol_kd"] = tmp.std(axis=1, ddof=0)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=[
            "target_return_kd",
            "target_logret_kd",
            "target_direction",
            "target_vol_kd",
        ]
    ).reset_index(drop=True)
    return df
def _read_csv_safe(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()

def _read_registry_safe(horizon_days: int) -> pd.DataFrame:
    p = METRICS_DIR / f"model_registry_k{horizon_days}.csv"
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def select_best_models(horizon_days: int) -> dict:
    """Select best models per symbol for direction and volatility.

    Priority:
      1. model_registry_k{horizon}.csv if it exists and is valid
      2. fallback to metrics_walk_direction/volatility files

    Returns:
      {
        "direction": {symbol: model_name, ...},
        "volatility": {symbol: model_name, ...},
      }
    """
    out = {"direction": {}, "volatility": {}}

    # Baseline-модели (LSTM, ARIMA, GARCH) служат только для научного сравнения
    # в статье; production-инференс (бот/сайт) их не обучает на лету, поэтому
    # исключаем их из выбора и берём лучшую обучаемую модель.
    EXTERNAL_BASELINES = {"LSTM", "ARIMA", "GARCH", "GARCH_X"}

    # --- 1) Prefer persisted registry if present ---
    reg = _read_registry_safe(horizon_days)
    if not reg.empty and {"task", "symbol", "model"}.issubset(reg.columns):
        reg = reg.copy()
        if "horizon_days" in reg.columns:
            reg = reg[reg["horizon_days"].astype(int) == int(horizon_days)]
        # исключаем baseline-модели из production-выбора
        reg = reg[~reg["model"].astype(str).isin(EXTERNAL_BASELINES)]

        for task_name in ("direction", "volatility"):
            part = reg[reg["task"].astype(str) == task_name].copy()
            if not part.empty:
                # keep latest row per symbol if duplicates exist
                if "saved_at_utc" in part.columns:
                    part = part.sort_values(["symbol", "saved_at_utc"])
                part = part.groupby("symbol", as_index=False).tail(1)
                out[task_name] = dict(
                    zip(part["symbol"].astype(str), part["model"].astype(str))
                )

    # --- 2) Fallback to metrics files (дополняет недостающие символы) ---
    # Считаем лучшую обучаемую модель из walk-forward метрик и ДОПОЛНЯЕМ ею
    # символы, для которых registry не дал production-модель (например там был
    # только LSTM, который мы исключили выше).
    d_path = METRICS_DIR / f"metrics_walk_direction_k{horizon_days}.csv"
    v_path = METRICS_DIR / f"metrics_walk_volatility_k{horizon_days}.csv"

    d = _read_csv_safe(d_path)
    v = _read_csv_safe(v_path)

    # Direction: maximize AUC on test splits
    if not d.empty and {"symbol", "model", "split", "auc"}.issubset(d.columns):
        dd = d[d["split"] == "test"].copy()
        dd = dd[~dd["model"].astype(str).str.startswith("BASELINE")]
        dd = dd[~dd["model"].astype(str).isin(EXTERNAL_BASELINES)]
        if not dd.empty:
            s = (
                dd.groupby(["symbol", "model"], as_index=False)
                .agg(auc_mean=("auc", "mean"))
                .sort_values(["symbol", "auc_mean"], ascending=[True, False])
            )
            best = s.groupby("symbol", as_index=False).head(1)
            fallback_dir = dict(zip(best["symbol"].astype(str), best["model"].astype(str)))
            # дополняем только недостающие символы (registry имеет приоритет)
            for sym, model in fallback_dir.items():
                out["direction"].setdefault(sym, model)

    # Volatility: minimize RMSE on test splits
    if not v.empty and {"symbol", "model", "split", "rmse"}.issubset(v.columns):
        vv = v[v["split"] == "test"].copy()
        vv = vv[~vv["model"].astype(str).str.startswith("BASELINE")]
        vv = vv[~vv["model"].astype(str).isin(EXTERNAL_BASELINES)]
        if not vv.empty:
            s = (
                vv.groupby(["symbol", "model"], as_index=False)
                .agg(rmse_mean=("rmse", "mean"))
                .sort_values(["symbol", "rmse_mean"], ascending=[True, True])
            )
            best = s.groupby("symbol", as_index=False).head(1)
            fallback_vol = dict(zip(best["symbol"].astype(str), best["model"].astype(str)))
            for sym, model in fallback_vol.items():
                out["volatility"].setdefault(sym, model)

    return out

def save_model_registry(task: str, horizon_days: int, best_df: pd.DataFrame) -> Path:
    """Persist best-per-symbol model selection for production use."""
    registry_path = METRICS_DIR / f"model_registry_k{horizon_days}.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    if best_df is None or best_df.empty:
        return registry_path

    reg = best_df.copy()
    reg.insert(0, "task", task)
    reg["horizon_days"] = int(horizon_days)
    reg["saved_at_utc"] = pd.Timestamp.now("UTC").isoformat()

    if task == "direction":
        metric_name = "auc_mean"
        reg["metric_name"] = metric_name
        reg["metric_value"] = reg[metric_name]
        reg["selection_note"] = "max auc_mean on walk-forward test"
    else:
        metric_name = "rmse_mean"
        reg["metric_name"] = metric_name
        reg["metric_value"] = reg[metric_name]
        reg["selection_note"] = "min rmse_mean on walk-forward test"

    keep_cols = [
        "task",
        "horizon_days",
        "symbol",
        "model",
        "metric_name",
        "metric_value",
        "selection_note",
        "saved_at_utc",
    ]
    reg = reg[keep_cols]

    if registry_path.exists():
        try:
            prev = pd.read_csv(registry_path)
        except Exception:
            prev = pd.DataFrame(columns=keep_cols)
        if not prev.empty:
            prev = prev[prev["task"].astype(str) != str(task)]
            reg = pd.concat([prev, reg], ignore_index=True)

    reg = reg.sort_values(["task", "symbol"]).reset_index(drop=True)
    reg.to_csv(registry_path, index=False)
    print(f"[ARTIFACT] Saved model registry to: {registry_path}")
    return registry_path


def get_model_by_name(task: str, name: str):
    """Instantiate a model from make_models() by its name."""
    for n, model, extra in make_models(task):
        if n == name:
            return model, extra
    raise ValueError(f"Unknown model '{name}' for task='{task}'")


MODELS_DIR = Path(os.environ.get("MODELS_DIR", "artifacts/models"))
LIVE_CALIB_DAYS = int(os.environ.get("LIVE_CALIB_DAYS", "252"))   # calibration block of the live model
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _model_path(sym: str, task: str, name: str, horizon: int) -> Path:
    safe_sym = sym.replace("^", "_")
    return MODELS_DIR / f"{safe_sym}_{task}_{name}_k{horizon}.joblib"


def _load_or_train_model(sym: str, task: str, name: str, X_train: np.ndarray, y_train: np.ndarray, force_retrain: bool = False):
    """Кэшированная тренировка: если модель уже сохранена и свежая — берём с диска."""
    import joblib
    path = _model_path(sym, task, name, HORIZON_DAYS)
    if path.exists() and not force_retrain:
        try:
            payload = joblib.load(path)
            n_trained = int(payload.get("n_train_rows", 0))
            # Если объём данных вырос больше чем на 5 строк — переобучаем
            if abs(n_trained - len(X_train)) <= 5:
                return payload["model"]
        except Exception as e:
            print(f"[WARN] Не удалось загрузить {path}: {e} — переобучаем")

    model, _ = get_model_by_name(task, name)
    model.fit(X_train, y_train)
    try:
        joblib.dump({"model": model, "n_train_rows": len(X_train),
                     "saved_at": pd.Timestamp.now("UTC").isoformat()}, path)
        print(f"[CACHE] Модель сохранена: {path}")
    except Exception as e:
        print(f"[WARN] Не удалось сохранить модель {path}: {e}")
    return model


def infer_latest_for_symbol(df: pd.DataFrame, sym: str, best_dir: str, best_vol: str,
                            engine=None, force_retrain: bool = False) -> dict | None:
    """Инференс на самую свежую дату с кэшированием обученных моделей."""
    df2, feature_cols = build_feature_matrix(df)
    if df2.empty or not feature_cols:
        print(f"[WARN] {sym}: cannot infer (no usable features)")
        return None

    if engine is None:
        engine = create_engine(get_env("DATABASE_URL"), pool_pre_ping=True)

    latest_price = pd.read_sql_query(
        text("SELECT date FROM market_ohlcv WHERE symbol = :symbol ORDER BY date DESC LIMIT 1"),
        engine, params={"symbol": sym}
    )
    if latest_price.empty:
        print(f"[WARN] {sym}: нет цен в market_ohlcv")
        return None
    latest_date = pd.to_datetime(latest_price["date"].iloc[0])
    print(f"[DEBUG] Latest price date used for {sym}: {latest_date.date()}")

    latest = df2[df2["date"] <= latest_date].sort_values("date").iloc[[-1]].copy()

    labeled = compute_targets(df2.copy(), HORIZON_DAYS)
    if labeled.empty:
        print(f"[WARN] {sym}: cannot infer (no labeled rows)")
        return None

    X_train = labeled[feature_cols].values
    X_latest = latest[feature_cols].values
    feature_names = feature_cols

    # === Direction: model fitted on all labelled rows except the most recent
    # LIVE_CALIB_DAYS (purged by k rows), which are used for the isotonic PAV
    # calibration of its probability (Lemma 1) ===
    y_dir = labeled["target_direction"].values
    n_lab = len(labeled)
    n_fit = n_lab - LIVE_CALIB_DAYS - HORIZON_DAYS
    if n_fit < 200:
        print(f"[WARN] {sym}: too few labelled rows for a calibrated forecast ({n_lab})")
        return None
    dir_model = _load_or_train_model(sym, "direction", best_dir, X_train[:n_fit], y_dir[:n_fit], force_retrain)
    p_cal_block = dir_model.predict_proba(X_train[n_lab - LIVE_CALIB_DAYS:])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        p_cal_block, y_dir[n_lab - LIVE_CALIB_DAYS:])
    p_up_raw = float(dir_model.predict_proba(X_latest)[:, 1][0])
    p_up = float(iso.predict([p_up_raw])[0])

    # === No-skill fallback: if the model has no walk-forward skill on this
    # instrument (mean test AUC < 0.5), report the base rate instead ===
    auc_reg = _walkforward_auc(sym, best_dir)
    if auc_reg is not None and auc_reg < 0.50:
        baseline_p = float(np.mean(y_dir == 1))
        print(f"[FALLBACK] {sym}: walk-forward AUC={auc_reg:.3f} < 0.50, reporting the base rate {baseline_p:.3f}")
        p_up = baseline_p

    try:
        # the training rows are the background of the linear member (LinearExplainer);
        # with the explained row as its own background its attribution would be zero
        compute_and_save_shap(dir_model, X_latest, feature_names, f"{sym}_direction",
                              X_background=X_train[:n_fit])
    except Exception as e:
        print(f"[WARN] SHAP direction для {sym} не посчитан: {e}")

    # === Volatility ===
    # The reported forecast is H-ECON-ET (Section 4.4); the machine-learning
    # regressor is still fitted because its SHAP values explain the forecast.
    y_vol = labeled["target_vol_kd"].values
    vol_model = _load_or_train_model(sym, "volatility", best_vol, X_train, y_vol, force_retrain)
    vol_pred_ml = max(0.0, float(vol_model.predict(X_latest)[0]))
    try:
        from jobs.volatility_models import hybrid_forecast
        upto = df2[df2["date"] <= latest_date].reset_index(drop=True)
        vol_pred = max(0.0, hybrid_forecast(upto, feature_cols, base="ECON", k=HORIZON_DAYS))
        vol_name = "HYBRID_ECON_ET"
    except Exception as e:
        print(f"[WARN] {sym}: H-ECON-ET failed ({e}); using {best_vol}")
        vol_pred, vol_name = vol_pred_ml, str(best_vol)

    try:
        compute_and_save_shap(vol_model, X_latest, feature_names, f"{sym}_volatility",
                              X_background=X_train)
    except Exception as e:
        print(f"[WARN] SHAP volatility для {sym} не посчитан: {e}")

    return {
        "symbol": sym,
        "asof_date": str(pd.to_datetime(latest["date"].iloc[0]).date()),   # date of the features used
        "horizon_days": int(HORIZON_DAYS),
        "direction_model": str(best_dir),
        "p_up": p_up,
        "p_up_raw": p_up_raw,
        "auc_registered": auc_reg if auc_reg is not None else float("nan"),
        "volatility_model": vol_name,
        "vol_pred": vol_pred,
        "vol_pred_ml": vol_pred_ml,
    }


def _walkforward_auc(symbol: str, model: str) -> float | None:
    """Mean walk-forward test AUC of a direction model on one instrument."""
    p = METRICS_DIR / f"metrics_walk_direction_k{HORIZON_DAYS}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[(d["split"] == "test") & (d["symbol"] == symbol) & (d["model"] == model)]
    return float(d["auc"].mean()) if len(d) else None


def run_infer(engine) -> None:
    print(f"[INFO] ACTION=infer HORIZON_DAYS={HORIZON_DAYS}")
    selected = select_best_models(HORIZON_DAYS)

    # Defaults if registry/metrics not found for a symbol
    default_dir = os.environ.get("DEFAULT_DIR_MODEL", "LOGREG")
    default_vol = os.environ.get("DEFAULT_VOL_MODEL", "EXTRATREES")

    print(f"[INFO] Selected direction models: {selected.get('direction', {})}")
    print(f"[INFO] Selected volatility models: {selected.get('volatility', {})}")

    only_symbol = os.environ.get("ONLY_SYMBOL")
    only_symbol = only_symbol.strip() if only_symbol else None

    if only_symbol:
        symbols = [only_symbol]
    else:
        symbols = pd.read_sql_query(
            """
            SELECT DISTINCT symbol
            FROM features_daily
            WHERE symbol NOT IN ('^VIX','^IRX','^TNX')
            ORDER BY symbol
            """,
            con=engine,
        )["symbol"].tolist()

    rows = []
    for sym in symbols:
        df = pd.read_sql_query(
            text(
                """
                SELECT symbol, date,
                       open, high, low, close, volume,
                       return_1d, log_return,
                       sma_5, volatility_5,
                       sma_10, volatility_10,
                       sma_20, volatility_20,
                       return_lag_1, return_lag_2, return_lag_3, return_lag_4, return_lag_5,
                       mkt_return_1d, mkt_log_return, mkt_mom_5, mkt_mom_10, mkt_mom_20, mkt_vol_20,
                       vix_level, vix_return_1d, vix_change_1d,
                       irx_level, irx_change_1d,
                       tnx_level, tnx_change_1d
                FROM features_daily
                WHERE symbol = :symbol
                ORDER BY date
                """
            ),
            con=engine,
            params={"symbol": sym},
            parse_dates=["date"],
        )
        if df.empty:
            continue

        required_core = ["date", "open", "high", "low", "close", "volume", "return_1d", "log_return"]
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=required_core).reset_index(drop=True)
        if df.empty:
            continue

        df = add_technical_features(df)
        df = df.replace([np.inf, -np.inf], np.nan)

        # One direction model for every instrument (the system model of the paper);
        # LIVE_DIR_MODEL=registry restores the per-instrument choice of the registry.
        live_dir = os.environ.get("LIVE_DIR_MODEL", "HYBRID_VOTING")
        best_dir = selected.get("direction", {}).get(sym, default_dir) if live_dir == "registry" else live_dir
        best_vol = selected.get("volatility", {}).get(sym, default_vol)
        try:
            get_model_by_name("direction", best_dir)
        except ValueError:
            print(f"[WARN] {sym}: model {best_dir} is not available for inference, using {default_dir}")
            best_dir = default_dir
        try:
            get_model_by_name("volatility", best_vol)
        except ValueError:
            best_vol = default_vol

        r = infer_latest_for_symbol(df, sym, best_dir, best_vol, engine=engine)
        if r is not None:
            rows.append(r)

    if not rows:
        raise RuntimeError("Inference produced no rows. Check data / symbols.")

    out_df = pd.DataFrame(rows).sort_values(["symbol"]).reset_index(drop=True)
    out_df["generated_at_utc"] = pd.Timestamp.now("UTC").isoformat()

    if only_symbol and PRED_OUT.exists():
        try:
            prev = pd.read_csv(PRED_OUT)
            if not prev.empty and "symbol" in prev.columns:
                prev = prev[prev["symbol"].astype(str).str.upper() != only_symbol.upper()]
                out_df = (
                    pd.concat([prev, out_df], ignore_index=True)
                    .sort_values(["symbol"])
                    .reset_index(drop=True)
                )
        except Exception:
            pass

    out_df.to_csv(PRED_OUT, index=False)
    print("\n[INFER] Latest predictions:")
    print(out_df.to_string(index=False))
    print(f"\n[ARTIFACT] Saved predictions to: {PRED_OUT}")


def run_backtest(engine) -> None:
    """ACTION=backtest runs the economic back-test of the paper (jobs/backtest.py, Table 11):
    walk-forward refits every 21 days, PAV calibration on the most recent 252 labelled days,
    the position sizes of the risk module and 5 bps per side."""
    from jobs.backtest import main as backtest_main
    backtest_main()


def print_reg(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }
    print(f"[{name}] MAE={m['mae']:.6f} RMSE={m['rmse']:.6f} R2={m['r2']:.4f}")
    return m


def print_clf(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    auc = safe_auc(y_true, y_prob)
    posrate = float(np.mean(y_true == 1))
    print(
        f"[{name}] Acc={acc:.4f} BalAcc={bacc:.4f} F1={f1:.4f} "
        f"AUC={(auc if np.isfinite(auc) else np.nan):.4f} PosRate={posrate:.4f}"
    )
    return {"acc": acc, "balacc": bacc, "f1": f1, "auc": float(auc), "posrate": posrate}


def _best_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= t).astype(int)
        f1 = float(f1_score(y_true, y_pred))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


@dataclass
class RowReg:
    symbol: str
    mode: str
    fold: int
    split: str
    task: str
    horizon_days: int
    model: str
    n_rows: int
    mae: float
    rmse: float
    r2: float
    extra: str


@dataclass
class RowClf:
    symbol: str
    mode: str
    fold: int
    split: str
    task: str
    horizon_days: int
    model: str
    n_rows: int
    acc: float
    balacc: float
    f1: float
    auc: float
    posrate: float
    threshold: float
    extra: str


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1d",
        "log_return",
        "sma_5",
        "volatility_5",
        "sma_10",
        "volatility_10",
        "sma_20",
        "volatility_20",
        "return_lag_1",
        "return_lag_2",
        "return_lag_3",
        "return_lag_4",
        "return_lag_5",
        # engineered
        "hl_range",
        "oc_return",
        "mom_5",
        "mom_10",
        "mom_20",
        "rsi_14",
        "ema_12",
        "ema_26",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_width_20",
        "atr_14",
        "atrp_14",
        "vol_z_20",
        "ret_std_20",
        "ret_mean_20",
        # macro/context (from features_daily)
        "mkt_return_1d",
        "mkt_log_return",
        "mkt_mom_5",
        "mkt_mom_10",
        "mkt_mom_20",
        "mkt_vol_20",
        "vix_level",
        "vix_return_1d",
        "vix_change_1d",
        "irx_level",
        "irx_change_1d",
        "tnx_level",
        "tnx_change_1d",
        # macro interaction / regime
        "corr_mkt_60",
        "beta_mkt_60",
        "mkt_trend_20",
        "mkt_risk_20",
        "vix_log",
        "vix_z_60",
        "vix_x_mktret",
        "yc_slope",
    ]

    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    # Make feature set robust:
    # - keep only columns that exist
    # - drop columns that are entirely NaN (common when macro series weren't ingested)
    missing = [c for c in feature_cols if c not in df.columns]
    usable = [c for c in feature_cols if c in df.columns and df[c].notna().any()]
    dropped_all_nan = [c for c in feature_cols if c in df.columns and not df[c].notna().any()]

    if missing:
        print(f"[WARN] build_feature_matrix: missing columns will be ignored: {missing}")
    if dropped_all_nan:
        print(f"[WARN] build_feature_matrix: all-NaN columns will be ignored: {dropped_all_nan}")

    if not usable:
        # Nothing usable -> empty
        return df.iloc[0:0].copy(), []

    df = df.dropna(subset=usable).reset_index(drop=True)
    return df, usable


def clip_targets(y: np.ndarray, pct: float) -> tuple[np.ndarray, str]:
    """Winsorize targets to reduce extreme-outlier impact (train-only)."""
    if pct <= 0.0:
        return y, ""
    y = np.asarray(y, dtype=float)
    lo = float(np.nanquantile(y, pct))
    hi = float(np.nanquantile(y, 1.0 - pct))
    y2 = np.clip(y, lo, hi)
    return y2, f"y_clip={pct:.3g}[{lo:.4g},{hi:.4g}]"


def make_models(task: str):
    """Улучшенный гибридный ensemble.

    Direction:
      LOGREG (с масштабированием), HGB (sklearn), XGB, LGBM, RF,
      HYBRID_VOTING (soft-vote с весами по силе модели),
      HYBRID_STACK (Stacking с мета-моделью LogReg, StratifiedKFold(5) внутри train).

    Volatility:
      EXTRATREES, XGB, LGBM, HGB,
      HYBRID_STACK_REG (Stacking с мета-моделью Ridge).
    """
    if task == "direction":
        logreg = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced", random_state=RANDOM_STATE)),
        ])
        hgb = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.04, max_iter=600,
            l2_regularization=1.0, min_samples_leaf=30,
            class_weight="balanced", random_state=RANDOM_STATE,
        )
        xgb = XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.85, colsample_bytree=0.85,
            reg_lambda=1.5, reg_alpha=0.1,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        )
        lgbm = LGBMClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=7, num_leaves=63,
            subsample=0.85, colsample_bytree=0.85,
            reg_lambda=1.5, reg_alpha=0.1,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        rf = RandomForestClassifier(
            n_estimators=400, max_depth=10, min_samples_leaf=20,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )

        # Soft-voting (быстрый гибрид)
        voting = VotingClassifier(
            estimators=[("logreg", logreg), ("hgb", hgb), ("xgb", xgb), ("lgbm", lgbm), ("rf", rf)],
            voting="soft", weights=[1, 2, 2, 3, 1],
        )

        # Stacking — мета-модель учится комбинировать предсказания базовых.
        # ВАЖНО: sklearn StackingClassifier требует CV-partition (каждая строка ровно
        # в одном test-фолде). TimeSeriesSplit этому не удовлетворяет, поэтому
        # используем StratifiedKFold(5). Time-leak ограничен train-сегментом
        # walk-forward'а — финальный test всегда идёт ПОСЛЕ train по времени.
        stacking = StackingClassifier(
            estimators=[("logreg", logreg), ("hgb", hgb), ("xgb", xgb), ("lgbm", lgbm), ("rf", rf)],
            final_estimator=LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE),
            stack_method="predict_proba",
            passthrough=False,
            cv=5,
            n_jobs=1,
        )

        return [
            ("LOGREG", logreg, ""),
            ("HGB", hgb, ""),
            ("XGB", xgb, ""),
            ("LGBM", lgbm, ""),
            ("RF", rf, ""),
            ("HYBRID_VOTING", voting, "soft-voting ensemble"),
            ("HYBRID_STACK", stacking, "stacking with StratifiedKFold(5) meta-LogReg"),
        ]

    elif task == "volatility":
        extratrees = ExtraTreesRegressor(
            n_estimators=500, max_depth=12, min_samples_leaf=10,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        xgb = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.85, colsample_bytree=0.85,
            reg_lambda=1.5, reg_alpha=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        )
        lgbm = LGBMRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=7, num_leaves=63,
            subsample=0.85, colsample_bytree=0.85,
            reg_lambda=1.5, reg_alpha=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        hgb = HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.04, max_iter=600,
            l2_regularization=1.0, min_samples_leaf=30, random_state=RANDOM_STATE,
        )

        # KFold(5) — sklearn StackingRegressor требует partition-CV.
        stack_reg = StackingRegressor(
            estimators=[("et", extratrees), ("xgb", xgb), ("lgbm", lgbm), ("hgb", hgb)],
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            passthrough=False,
            n_jobs=1,
        )

        return [
            ("EXTRATREES", extratrees, ""),
            ("XGB", xgb, ""),
            ("LGBM", lgbm, ""),
            ("HGB", hgb, ""),
            ("HYBRID_STACK_REG", stack_reg, "stacking regressor with Ridge meta"),
        ]

    return []


def eval_one_split_reg(sym: str, fold: int, split_name: str, task: str, y_true, y_pred, n_rows: int, model_name: str, extra: str, out: list[RowReg]):
    m = print_reg(f"{model_name}({split_name})", y_true, y_pred)
    out.append(RowReg(sym, MODE, fold, split_name, task, HORIZON_DAYS, model_name, n_rows, **m, extra=extra))


def eval_one_split_clf(sym: str, fold: int, split_name: str, task: str, y_true, y_pred, y_prob, n_rows: int, model_name: str, threshold: float, extra: str, out: list[RowClf]):
    m = print_clf(f"{model_name}({split_name})", y_true, y_pred, y_prob)
    out.append(RowClf(sym, MODE, fold, split_name, task, HORIZON_DAYS, model_name, n_rows, **m, threshold=threshold, extra=extra))


def run_single(df: pd.DataFrame, sym: str) -> tuple[list[RowReg], list[RowClf]]:
    """Single fixed split evaluation WITHOUT target leakage.

    Key rule: targets must be computed *within each split* (train/val/test)
    so that y for the last rows of train never uses prices from val/test.
    """

    results_reg: list[RowReg] = []
    results_clf: list[RowClf] = []

    # First build feature matrix and drop feature NaNs globally (chronology preserved)
    df2, feature_cols = build_feature_matrix(df)
    if df2.empty:
        print(f"[WARN] {sym}: empty after feature drop")
        return results_reg, results_clf

    # Causal persistence predictors (use ONLY past information)
    df2 = df2.copy()
    df2["persist_return_kd"] = df2["close"] / df2["close"].shift(HORIZON_DAYS) - 1.0
    df2["persist_logret_kd"] = np.log(df2["close"] / df2["close"].shift(HORIZON_DAYS)).replace(
        [np.inf, -np.inf], np.nan
    )
    df2["persist_vol_kd"] = df2["log_return"].rolling(HORIZON_DAYS, min_periods=HORIZON_DAYS).std(ddof=0)

    # Time split on the cleaned feature matrix
    train = df2[df2["date"] <= TRAIN_END].copy()
    val = df2[(df2["date"] > TRAIN_END) & (df2["date"] <= VAL_END)].copy()
    test = df2[df2["date"] > VAL_END].copy()

    if len(train) < 500 or len(val) < 150 or len(test) < 150:
        print(
            f"[WARN] {sym}: not enough rows after feature drop + split: "
            f"train={len(train)} val={len(val)} test={len(test)}"
        )
        return results_reg, results_clf

    # Compute targets PER SPLIT to avoid leakage across split boundaries
    train = compute_targets(train, HORIZON_DAYS)
    val = compute_targets(val, HORIZON_DAYS)
    test = compute_targets(test, HORIZON_DAYS)

    if len(train) < 500 or len(val) < 150 or len(test) < 150:
        print(
            f"[WARN] {sym}: not enough rows after target drop: "
            f"train={len(train)} val={len(val)} test={len(test)}"
        )
        return results_reg, results_clf

    X_train = train[feature_cols].values
    X_val = val[feature_cols].values
    X_test = test[feature_cols].values

    print(f"\n=== {sym} ===")
    print(f"[ROWS] train={len(train)} val={len(val)} test={len(test)}")

    if TASK in ("return", "volatility"):
        ycol = (
            "target_vol_kd"
            if TASK == "volatility"
            else (
                "target_logret_kd" if RETURN_TARGET_MODE == "log" else "target_return_kd"
            )
        )
        y_train = train[ycol].values
        y_val = val[ycol].values
        y_test = test[ycol].values

        # BASELINE_PERSIST: causal persistence using past k-day return / past k-day vol proxy
        if TASK == "volatility":
            pcol = "persist_vol_kd"
        else:
            pcol = "persist_logret_kd" if (RETURN_TARGET_MODE == "log") else "persist_return_kd"

        y_persist_val = val[pcol].values
        y_persist_test = test[pcol].values

        # Fill any missing persistence values (early rows) with train mean
        mean_train = float(np.mean(y_train))
        y_persist_val = np.where(np.isfinite(y_persist_val), y_persist_val, mean_train)
        y_persist_test = np.where(np.isfinite(y_persist_test), y_persist_test, mean_train)

        eval_one_split_reg(
            sym,
            0,
            "val",
            TASK,
            y_val,
            y_persist_val,
            len(val),
            "BASELINE_PERSIST",
            f"pcol={pcol}",
            results_reg,
        )
        eval_one_split_reg(
            sym,
            0,
            "test",
            TASK,
            y_test,
            y_persist_test,
            len(test),
            "BASELINE_PERSIST",
            f"pcol={pcol}",
            results_reg,
        )

        # Baselines
        eval_one_split_reg(
            sym,
            0,
            "val",
            TASK,
            y_val,
            np.zeros_like(y_val),
            len(val),
            "BASELINE_ZERO",
            "",
            results_reg,
        )
        eval_one_split_reg(
            sym,
            0,
            "test",
            TASK,
            y_test,
            np.zeros_like(y_test),
            len(test),
            "BASELINE_ZERO",
            "",
            results_reg,
        )
        eval_one_split_reg(
            sym,
            0,
            "val",
            TASK,
            y_val,
            np.full_like(y_val, mean_train, dtype=float),
            len(val),
            "BASELINE_MEAN_TRAIN",
            f"mean={mean_train:.6g}",
            results_reg,
        )
        eval_one_split_reg(
            sym,
            0,
            "test",
            TASK,
            y_test,
            np.full_like(y_test, mean_train, dtype=float),
            len(test),
            "BASELINE_MEAN_TRAIN",
            f"mean={mean_train:.6g}",
            results_reg,
        )

        # Robust target clipping (train-only)
        y_train_fit, clip_extra = clip_targets(y_train, Y_CLIP_PCT)
        # Optional log-target training (scores always computed on original scale)
        use_log = bool(LOG_TARGET) and (float(np.min(y_train_fit)) > -0.999999)
        if bool(LOG_TARGET) and not use_log:
            print("[WARN] LOG_TARGET=1 but y_train has values <= -1; skipping log1p transform")

        y_fit = np.log1p(y_train_fit) if use_log else y_train_fit

        for name, model, extra in make_models(TASK):
            model.fit(X_train, y_fit)

            pv = model.predict(X_val)
            pt = model.predict(X_test)

            if use_log:
                pv = np.expm1(pv)
                pt = np.expm1(pt)

            eval_one_split_reg(sym, 0, "val", TASK, y_val, pv, len(val), name, " ".join([extra, clip_extra]).strip(), results_reg)
            eval_one_split_reg(sym, 0, "test", TASK, y_test, pt, len(test), name, " ".join([extra, clip_extra]).strip(), results_reg)

    else:  # direction
        y_train = train["target_direction"].values
        y_val = val["target_direction"].values
        y_test = test["target_direction"].values

        # Baselines
        eval_one_split_clf(
            sym,
            0,
            "val",
            TASK,
            y_val,
            np.zeros_like(y_val),
            np.zeros_like(y_val, dtype=float),
            len(val),
            "BASELINE_ALL_ZERO",
            0.5,
            "",
            results_clf,
        )
        eval_one_split_clf(
            sym,
            0,
            "test",
            TASK,
            y_test,
            np.zeros_like(y_test),
            np.zeros_like(y_test, dtype=float),
            len(test),
            "BASELINE_ALL_ZERO",
            0.5,
            "",
            results_clf,
        )

        p = float(np.mean(y_train == 1))
        prob_val = np.full_like(y_val, p, dtype=float)
        prob_test = np.full_like(y_test, p, dtype=float)
        pred_val = (prob_val >= 0.5).astype(int)
        pred_test = (prob_test >= 0.5).astype(int)
        eval_one_split_clf(
            sym,
            0,
            "val",
            TASK,
            y_val,
            pred_val,
            prob_val,
            len(val),
            "BASELINE_CONST_FROM_TRAIN",
            0.5,
            f"p_train={p:.3f}",
            results_clf,
        )
        eval_one_split_clf(
            sym,
            0,
            "test",
            TASK,
            y_test,
            pred_test,
            prob_test,
            len(test),
            "BASELINE_CONST_FROM_TRAIN",
            0.5,
            f"p_train={p:.3f}",
            results_clf,
        )

        for name, model, extra in make_models(TASK):
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                pv = model.predict_proba(X_val)[:, 1]
                pt = model.predict_proba(X_test)[:, 1]
            else:
                sv = model.decision_function(X_val)
                st = model.decision_function(X_test)
                pv = 1.0 / (1.0 + np.exp(-sv))
                pt = 1.0 / (1.0 + np.exp(-st))

            t_best, f1_best = _best_threshold_f1(y_val, pv)
            eval_one_split_clf(
                sym,
                0,
                "val",
                TASK,
                y_val,
                (pv >= t_best).astype(int),
                pv,
                len(val),
                name,
                t_best,
                f"t_f1={f1_best:.4f} {extra}",
                results_clf,
            )
            eval_one_split_clf(
                sym,
                0,
                "test",
                TASK,
                y_test,
                (pt >= t_best).astype(int),
                pt,
                len(test),
                name,
                t_best,
                f"t_f1={f1_best:.4f} {extra}",
                results_clf,
            )

    return results_reg, results_clf


def run_walk(df: pd.DataFrame, sym: str) -> tuple[list[RowReg], list[RowClf]]:
    results_reg: list[RowReg] = []
    results_clf: list[RowClf] = []

    df2, feature_cols = build_feature_matrix(df)
    if df2.empty:
        print(f"[WARN] {sym}: empty after feature drop")
        return results_reg, results_clf

    # Causal persistence predictors (use ONLY past information)
    df2 = df2.copy()
    df2["persist_return_kd"] = df2["close"] / df2["close"].shift(HORIZON_DAYS) - 1.0
    df2["persist_logret_kd"] = np.log(df2["close"] / df2["close"].shift(HORIZON_DAYS)).replace(
        [np.inf, -np.inf], np.nan
    )
    df2["persist_vol_kd"] = df2["log_return"].rolling(HORIZON_DAYS, min_periods=HORIZON_DAYS).std(ddof=0)

    dates = df2["date"].reset_index(drop=True)
    n = len(df2)

    # Start so we have enough train + val + test
    start_idx = WF_MIN_TRAIN_ROWS
    min_total = WF_MIN_TRAIN_ROWS + WF_VAL_DAYS + WF_TEST_DAYS
    if n < min_total:
        print(f"[WARN] {sym}: not enough rows for walk-forward: n={n}, need>={min_total}")
        return results_reg, results_clf

    # last index where we can still fit val+test
    last_start = n - (WF_VAL_DAYS + WF_TEST_DAYS)

    fold = 0
    i = start_idx
    while i <= last_start:
        # expanding train up to i (exclusive)
        tr = df2.iloc[:i].copy()
        va = df2.iloc[i : i + WF_VAL_DAYS].copy()
        te = df2.iloc[i + WF_VAL_DAYS : i + WF_VAL_DAYS + WF_TEST_DAYS].copy()

        # compute targets *after* split boundaries already fixed by indexing
        tr = compute_targets(tr, HORIZON_DAYS)
        va = compute_targets(va, HORIZON_DAYS)
        te = compute_targets(te, HORIZON_DAYS)

        # After target creation, re-align: we must drop tail rows that lost targets.
        # Ensure we still have reasonable sizes.
        if len(tr) < 500 or len(va) < 80 or len(te) < 80:
            i += WF_STEP_DAYS
            continue

        X_train = tr[feature_cols].values
        X_val = va[feature_cols].values
        X_test = te[feature_cols].values

        train_end = to_pydate(dates.iloc[i - 1])
        print(f"\n=== {sym} | fold={fold} | train_end={train_end} ===")
        print(f"[ROWS] train={len(tr)} val={len(va)} test={len(te)}")

        if TASK in ("return", "volatility"):
            ycol = "target_vol_kd" if TASK == "volatility" else ("target_logret_kd" if RETURN_TARGET_MODE == "log" else "target_return_kd")
            y_train = tr[ycol].values
            y_val = va[ycol].values
            y_test = te[ycol].values

            mean_train = float(np.mean(y_train))

            # BASELINE_PERSIST: causal persistence using past k-day return / past k-day vol proxy
            if TASK == "volatility":
                pcol = "persist_vol_kd"
            else:
                pcol = "persist_logret_kd" if (RETURN_TARGET_MODE == "log") else "persist_return_kd"

            y_persist_val = va[pcol].values
            y_persist_test = te[pcol].values
            y_persist_val = np.where(np.isfinite(y_persist_val), y_persist_val, mean_train)
            y_persist_test = np.where(np.isfinite(y_persist_test), y_persist_test, mean_train)

            eval_one_split_reg(sym, fold, "val", TASK, y_val, y_persist_val, len(va), "BASELINE_PERSIST", f"pcol={pcol}", results_reg)
            eval_one_split_reg(sym, fold, "test", TASK, y_test, y_persist_test, len(te), "BASELINE_PERSIST", f"pcol={pcol}", results_reg)

            eval_one_split_reg(sym, fold, "val", TASK, y_val, np.zeros_like(y_val), len(va), "BASELINE_ZERO", "", results_reg)
            eval_one_split_reg(sym, fold, "test", TASK, y_test, np.zeros_like(y_test), len(te), "BASELINE_ZERO", "", results_reg)
            eval_one_split_reg(sym, fold, "val", TASK, y_val, np.full_like(y_val, mean_train, dtype=float), len(va), "BASELINE_MEAN_TRAIN", f"mean={mean_train:.6g}", results_reg)
            eval_one_split_reg(sym, fold, "test", TASK, y_test, np.full_like(y_test, mean_train, dtype=float), len(te), "BASELINE_MEAN_TRAIN", f"mean={mean_train:.6g}", results_reg)

            # Robust target clipping (train-only)
            y_train_fit, clip_extra = clip_targets(y_train, Y_CLIP_PCT)
            # Optional log-target training (scores always computed on original scale)
            use_log = bool(LOG_TARGET) and (float(np.min(y_train_fit)) > -0.999999)
            if bool(LOG_TARGET) and not use_log:
                print("[WARN] LOG_TARGET=1 but y_train has values <= -1; skipping log1p transform")

            y_fit = np.log1p(y_train_fit) if use_log else y_train_fit

            for name, model, extra in make_models(TASK):
                model.fit(X_train, y_fit)

                pv = model.predict(X_val)
                pt = model.predict(X_test)

                if use_log:
                    pv = np.expm1(pv)
                    pt = np.expm1(pt)

                eval_one_split_reg(sym, fold, "val", TASK, y_val, pv, len(va), name, " ".join([extra, clip_extra]).strip(), results_reg)
                eval_one_split_reg(sym, fold, "test", TASK, y_test, pt, len(te), name, " ".join([extra, clip_extra]).strip(), results_reg)

        else:
            y_train = tr["target_direction"].values
            y_val = va["target_direction"].values
            y_test = te["target_direction"].values

            eval_one_split_clf(sym, fold, "val", TASK, y_val, np.zeros_like(y_val), np.zeros_like(y_val, dtype=float), len(va), "BASELINE_ALL_ZERO", 0.5, "", results_clf)
            eval_one_split_clf(sym, fold, "test", TASK, y_test, np.zeros_like(y_test), np.zeros_like(y_test, dtype=float), len(te), "BASELINE_ALL_ZERO", 0.5, "", results_clf)

            p = float(np.mean(y_train == 1))
            pv0 = np.full_like(y_val, p, dtype=float)
            pt0 = np.full_like(y_test, p, dtype=float)
            eval_one_split_clf(sym, fold, "val", TASK, y_val, (pv0 >= 0.5).astype(int), pv0, len(va), "BASELINE_CONST_FROM_TRAIN", 0.5, f"p_train={p:.3f}", results_clf)
            eval_one_split_clf(sym, fold, "test", TASK, y_test, (pt0 >= 0.5).astype(int), pt0, len(te), "BASELINE_CONST_FROM_TRAIN", 0.5, f"p_train={p:.3f}", results_clf)

            for name, model, extra in make_models(TASK):
                model.fit(X_train, y_train)
                if hasattr(model, "predict_proba"):
                    pv = model.predict_proba(X_val)[:, 1]
                    pt = model.predict_proba(X_test)[:, 1]
                else:
                    sv = model.decision_function(X_val)
                    st = model.decision_function(X_test)
                    pv = 1.0 / (1.0 + np.exp(-sv))
                    pt = 1.0 / (1.0 + np.exp(-st))

                t_best, f1_best = _best_threshold_f1(y_val, pv)
                eval_one_split_clf(sym, fold, "val", TASK, y_val, (pv >= t_best).astype(int), pv, len(va), name, t_best, f"t_f1={f1_best:.4f} {extra}", results_clf)
                eval_one_split_clf(sym, fold, "test", TASK, y_test, (pt >= t_best).astype(int), pt, len(te), name, t_best, f"t_f1={f1_best:.4f} {extra}", results_clf)

        fold += 1
        i += WF_STEP_DAYS

    return results_reg, results_clf


def main() -> None:
    db_url = get_env("DATABASE_URL")

    only_symbol = os.environ.get("ONLY_SYMBOL")
    only_symbol = only_symbol.strip() if only_symbol else None

    engine = create_engine(db_url, pool_pre_ping=True)

    # Production actions
    if ACTION == "infer":
        run_infer(engine)
        return
    if ACTION == "backtest":
        run_backtest(engine)
        return

    if only_symbol:
        symbols = [only_symbol]
    else:
        symbols = pd.read_sql_query(
            """
            SELECT DISTINCT symbol
            FROM features_daily
            WHERE symbol NOT IN ('^VIX','^IRX','^TNX')
            ORDER BY symbol
            """,
            con=engine,
        )["symbol"].tolist()

    if not symbols:
        raise RuntimeError("No symbols in features_daily. Run build_features.py first.")

    print(f"[INFO] MODE={MODE} TASK={TASK} HORIZON_DAYS={HORIZON_DAYS} RETURN_TARGET_MODE={RETURN_TARGET_MODE}")
    print(f"[INFO] Symbols={symbols}")
    if MODE == "single":
        print(f"[INFO] Split: train_end={TRAIN_END}, val_end={VAL_END}")
    else:
        print(
            f"[INFO] WalkForward: min_train_rows={WF_MIN_TRAIN_ROWS} val_days={WF_VAL_DAYS} test_days={WF_TEST_DAYS} step_days={WF_STEP_DAYS}"
        )

    all_reg: list[RowReg] = []
    all_clf: list[RowClf] = []

    for sym in symbols:
        df = pd.read_sql_query(
            text(
                """
                SELECT symbol, date,
                    open, high, low, close, volume,
                    return_1d, log_return,
                    sma_5, volatility_5,
                    sma_10, volatility_10,
                    sma_20, volatility_20,
                    return_lag_1, return_lag_2, return_lag_3, return_lag_4, return_lag_5,
                    mkt_return_1d, mkt_log_return, mkt_mom_5, mkt_mom_10, mkt_mom_20, mkt_vol_20,
                    vix_level, vix_return_1d, vix_change_1d,
                    irx_level, irx_change_1d,
                    tnx_level, tnx_change_1d
                FROM features_daily
                WHERE symbol = :symbol
                AND date >= :start_date
                AND date <= :end_date
                ORDER BY date
                """
            ),
            con=engine,
            params={
                "symbol": sym,
                "start_date": TRAIN_START_DATE,
                "end_date": os.environ.get("END_DATE", "2026-06-01"),
            },
            parse_dates=["date"],
        )

        _end_date_env = os.environ.get("END_DATE", "2026-06-01")
        print(f"[INFO] {sym}: данные с {TRAIN_START_DATE} по {_end_date_env} → {len(df)} строк")
        if df.empty:
            print(f"[WARN] {sym}: empty, skip")
            continue

        # Do NOT dropna() across all columns here: macro/context columns can be NULL
        # (different start dates, rolling windows). We only require core OHLCV and
        # base return columns to be present; the final feature selection/drop is
        # handled inside build_feature_matrix().
        df = df.replace([np.inf, -np.inf], np.nan)

        required_core = ["date", "open", "high", "low", "close", "volume", "return_1d", "log_return"]
        missing_core = [c for c in required_core if c not in df.columns]
        if missing_core:
            print(f"[WARN] {sym}: missing core columns in features_daily query: {missing_core}")
            continue

        df = df.dropna(subset=required_core).reset_index(drop=True)
        if df.empty:
            print(f"[WARN] {sym}: empty after dropping NA core columns")
            continue

        df = add_technical_features(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        # Keep rows even if some engineered/macro features are NA; build_feature_matrix will decide.

        print(f"[INFO] {sym}: rows after core cleaning + technical features = {len(df)}")

        if MODE == "single":
            r, c = run_single(df, sym)
        else:
            r, c = run_walk(df, sym)

        all_reg.extend(r)
        all_clf.extend(c)

    if TASK in ("return", "volatility"):
        if not all_reg:
            raise RuntimeError("No results produced. Check data size/splits.")
        out_df = pd.DataFrame([asdict(r) for r in all_reg])
        # Сохраняем результаты внешних baselines (deep, GARCH, HAR, HYBRID_GARCH_ET)
        # из существующего файла, чтобы они не перезаписались.
        EXTERNAL_MODELS = {"LSTM", "GRU", "TRANSFORMER", "NBEATSX", "GARCH", "HAR", "HYBRID_GARCH_ET",
                           "GJR_GARCH", "HAR_RG", "ECON", "HYBRID_ECON_ET"}
        if OUT_PATH.exists():
            try:
                prev = pd.read_csv(OUT_PATH)
                external = prev[prev["model"].isin(EXTERNAL_MODELS)]
                if not external.empty:
                    out_df = pd.concat([out_df, external], ignore_index=True)
                    print(f"[INFO] Preserved {len(external)} external baseline rows: {sorted(external['model'].unique())}")
            except Exception as e:
                print(f"[WARN] Cannot preserve external baselines: {e}")
        out_df.to_csv(OUT_PATH, index=False)

        # quick summary across folds (test only)
        if MODE == "walk":
            test_df = out_df[out_df["split"] == "test"].copy()

            s = (
                test_df
                .groupby(["symbol", "model"], as_index=False)
                .agg(
                    rmse_mean=("rmse", "mean"),
                    rmse_std=("rmse", "std"),
                    r2_mean=("r2", "mean"),
                    mae_mean=("mae", "mean"),
                )
                .sort_values(["symbol", "rmse_mean"])
            )

            base = (
                s[s["model"] == "BASELINE_MEAN_TRAIN"]
                [["symbol", "rmse_mean"]]
                .rename(columns={"rmse_mean": "rmse_base"})
            )
            s2 = s.merge(base, on="symbol", how="left")
            s2["rmse_impr_pct_vs_base"] = 100.0 * (s2["rmse_base"] - s2["rmse_mean"]) / s2["rmse_base"]
            s2 = s2.drop(columns=["rmse_base"]).sort_values(["symbol", "rmse_mean"]).reset_index(drop=True)

            print("\n[SUMMARY] Walk-forward TEST (mean/std across folds) + improvement vs BASELINE_MEAN_TRAIN:")
            print(s2.to_string(index=False))

            non_base = s2[~s2["model"].str.startswith("BASELINE")].copy()
            if not non_base.empty:
                best = non_base.sort_values(["symbol", "rmse_mean"]).groupby("symbol", as_index=False).head(1)
                print("\n[BEST] Per symbol (excluding baselines):")
                print(best[["symbol", "model", "rmse_mean", "rmse_impr_pct_vs_base", "r2_mean"]].to_string(index=False))
                best_df = best.reset_index(drop=True).copy()
                save_model_registry(TASK, HORIZON_DAYS, best_df)

    else:
        if not all_clf:
            raise RuntimeError("No results produced. Check data size/splits.")
        out_df = pd.DataFrame([asdict(r) for r in all_clf])
        # Сохраняем результаты внешних baselines (deep, GARCH, HAR, HYBRID_GARCH_ET)
        # из существующего файла, чтобы они не перезаписались.
        EXTERNAL_MODELS = {"LSTM", "GRU", "TRANSFORMER", "NBEATSX", "GARCH", "HAR", "HYBRID_GARCH_ET",
                           "GJR_GARCH", "HAR_RG", "ECON", "HYBRID_ECON_ET"}
        if OUT_PATH.exists():
            try:
                prev = pd.read_csv(OUT_PATH)
                external = prev[prev["model"].isin(EXTERNAL_MODELS)]
                if not external.empty:
                    out_df = pd.concat([out_df, external], ignore_index=True)
                    print(f"[INFO] Preserved {len(external)} external baseline rows: {sorted(external['model'].unique())}")
            except Exception as e:
                print(f"[WARN] Cannot preserve external baselines: {e}")
        out_df.to_csv(OUT_PATH, index=False)

        if MODE == "walk":
            test_df = out_df[out_df["split"] == "test"].copy()

            s = (
                test_df
                .groupby(["symbol", "model"], as_index=False)
                .agg(
                    auc_mean=("auc", "mean"),
                    auc_std=("auc", "std"),
                    balacc_mean=("balacc", "mean"),
                    f1_mean=("f1", "mean"),
                )
                .sort_values(["symbol", "auc_mean"], ascending=[True, False])
            )

            base = (
                s[s["model"] == "BASELINE_ALL_ZERO"]
                [["symbol", "auc_mean"]]
                .rename(columns={"auc_mean": "auc_base"})
            )
            s2 = s.merge(base, on="symbol", how="left")
            s2["auc_delta_vs_base"] = s2["auc_mean"] - s2["auc_base"]
            s2 = s2.drop(columns=["auc_base"]).sort_values(["symbol", "auc_mean"], ascending=[True, False]).reset_index(drop=True)

            print("\n[SUMMARY] Walk-forward TEST (mean/std across folds) + delta vs BASELINE_ALL_ZERO:")
            print(s2.to_string(index=False))

            non_base = s2[~s2["model"].str.startswith("BASELINE")].copy()
            if not non_base.empty:
                best = non_base.sort_values(["symbol", "auc_mean"], ascending=[True, False]).groupby("symbol", as_index=False).head(1)
                print("\n[BEST] Per symbol (excluding baselines):")
                print(best[["symbol", "model", "auc_mean", "auc_delta_vs_base", "balacc_mean"]].to_string(index=False))
                best_df = best.reset_index(drop=True).copy()
                save_model_registry("direction", HORIZON_DAYS, best_df)

    # После walk-forward инвалидируем закэшированные joblib-модели,
    # чтобы при следующем infer они переобучились на актуальных данных.
    try:
        for p in MODELS_DIR.glob("*.joblib"):
            p.unlink()
        print(f"[CACHE] Очищен кэш моделей: {MODELS_DIR}")
    except Exception as e:
        print(f"[WARN] Не удалось очистить кэш моделей: {e}")

    print("\n[DONE] Finished.")
    print(f"[ARTIFACT] Saved metrics to: {OUT_PATH}")


if __name__ == "__main__":
    main()