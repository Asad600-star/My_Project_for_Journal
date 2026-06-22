"""
LSTM baseline for direction forecasting.
================================================================

Сравнительный baseline для статьи: LSTM на тех же 56 причинных признаках,
что и HYBRID_VOTING / HYBRID_STACK, с идентичным walk-forward CV (K=14).

Цель: показать в Q1 статье, что предложенный HYBRID подход
превосходит классический deep-learning baseline LSTM.

Архитектура:
    Input  -> (batch, seq_len=20, n_features=56)
    LSTM   -> 2 слоя, hidden=32, dropout=0.2
    Dense  -> 32 -> 16 -> 1
    Output -> Sigmoid (вероятность направления вверх)

Запуск:
    python -m jobs.train_lstm_baseline

Результаты добавляются в artifacts/metrics_walk_direction_k5.csv
с model="LSTM" для прямого сравнения с другими моделями.

Воспроизводимость:
- Random seed зафиксирован (SEED=42 по умолчанию)
- END_DATE = 2026-06-01 (фиксация для статьи; убрать env END_DATE для актуальных данных)
- Все гиперпараметры можно изменить через env переменные
"""
from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass, asdict
from datetime import date as dt_date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")
load_dotenv()


# ─────────────────────────────────────────────────────────────────────
#  Конфигурация (воспроизводимость + та же сетка фолдов что у HYBRID)
# ─────────────────────────────────────────────────────────────────────
SEED = int(os.environ.get("SEED", "42"))
END_DATE = os.environ.get("END_DATE", "2026-06-01")  # фиксация для статьи
HORIZON_DAYS = int(os.environ.get("HORIZON_DAYS", "5"))

# Параметры walk-forward (идентичны train_baseline.py)
WF_MIN_TRAIN_ROWS = int(os.environ.get("WF_MIN_TRAIN_ROWS", "1200"))
WF_VAL_DAYS = int(os.environ.get("WF_VAL_DAYS", "126"))
WF_TEST_DAYS = int(os.environ.get("WF_TEST_DAYS", "126"))
WF_STEP_DAYS = int(os.environ.get("WF_STEP_DAYS", "63"))

# Гиперпараметры LSTM
SEQ_LEN = int(os.environ.get("SEQ_LEN", "20"))
LSTM_HIDDEN = int(os.environ.get("LSTM_HIDDEN", "32"))
LSTM_LAYERS = int(os.environ.get("LSTM_LAYERS", "2"))
LSTM_DROPOUT = float(os.environ.get("LSTM_DROPOUT", "0.2"))
LSTM_EPOCHS = int(os.environ.get("LSTM_EPOCHS", "50"))
LSTM_PATIENCE = int(os.environ.get("LSTM_PATIENCE", "10"))
LSTM_LR = float(os.environ.get("LSTM_LR", "1e-3"))
LSTM_BATCH = int(os.environ.get("LSTM_BATCH", "32"))

# Символы (расширенный список для Q1: 4 индекса + 4 акции/asset = 8 всего)
# Можно переопределить через env: SYMBOLS="AAPL,TSLA,^GSPC,^IXIC,^DJI,^RUT,GLD,MSFT"
SYMBOLS_DEFAULT = ["AAPL", "TSLA", "^GSPC", "^IXIC", "^DJI", "^RUT", "GLD", "MSFT"]
SYMBOLS_ENV = os.environ.get("SYMBOLS", "")
SYMBOLS = (
    [s.strip() for s in SYMBOLS_ENV.split(",") if s.strip()]
    if SYMBOLS_ENV
    else SYMBOLS_DEFAULT
)

# Выходные пути
ARTIFACTS_DIR = Path(os.environ.get("METRICS_DIR", "artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = ARTIFACTS_DIR / f"metrics_walk_direction_k{HORIZON_DAYS}.csv"


# Фиксация random seeds для полной воспроизводимости
def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS (Apple Silicon) deterministic mode
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# Выбор устройства (MPS на Mac M-series, CUDA на NVIDIA, иначе CPU)
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


# ─────────────────────────────────────────────────────────────────────
#  Загрузка данных из PostgreSQL (та же таблица что у train_baseline)
# ─────────────────────────────────────────────────────────────────────
def get_engine():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL env var не задан")
    return create_engine(db_url, pool_pre_ping=True)


def load_ohlcv(engine, symbol: str, end_date: str) -> pd.DataFrame:
    """Загружает OHLCV из market_ohlcv с фиксацией END_DATE для воспроизводимости."""
    q = text("""
        SELECT date, open, high, low, close, adj_close, volume
        FROM market_ohlcv
        WHERE symbol = :sym AND date <= :end_date
        ORDER BY date ASC
    """)
    df = pd.read_sql(q, engine, params={"sym": symbol, "end_date": end_date})
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─────────────────────────────────────────────────────────────────────
#  Технические признаки (тот же 56-мерный вектор что у HYBRID)
# ─────────────────────────────────────────────────────────────────────
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
    """Строит 24 технических признака (копия train_baseline.py)."""
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

    # Лаги (5)
    for k in range(1, 6):
        df[f"return_lag_{k}"] = df["simple_return"].shift(k)

    return df


def add_macro_and_regime_features(df: pd.DataFrame, engine) -> pd.DataFrame:
    """Добавляет 14 макро признаков (^VIX/^IRX/^TNX) + 8 regime признаков."""
    df = df.copy()

    # Загружаем макро тикеры
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

    # Макро features
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

    # Regime features
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


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Возвращает 56 признаков в стандартном порядке (как у HYBRID)."""
    exclude = {"date", "symbol", "log_return", "target_direction"}
    cols = [c for c in df.columns if c not in exclude and df[c].dtype in (float, "float64", "float32", int, "int64", "int32")]
    return cols


def compute_target_direction(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Бинарная цель: 1 если C_{t+k} >= C_t, иначе 0."""
    df = df.copy()
    df["target_return_kd"] = df["close"].shift(-horizon) / df["close"] - 1
    df["target_direction"] = (df["target_return_kd"] >= 0).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────
#  LSTM модель (PyTorch)
# ─────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    """2-слойный LSTM для бинарной классификации направления."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = LSTM_HIDDEN,
        num_layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, (h, c) = self.lstm(x)
        last = out[:, -1, :]  # последний timestep
        z = self.dropout(last)
        z = self.relu(self.fc1(z))
        logit = self.fc2(z).squeeze(-1)
        return logit


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Преобразует (n_rows, 56) -> (n_seqs, seq_len, 56), каждая seq = последние seq_len дней."""
    n = len(X)
    if n <= seq_len:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,))
    X_seq = np.zeros((n - seq_len, seq_len, X.shape[1]), dtype=np.float32)
    y_seq = np.zeros((n - seq_len,), dtype=np.float32)
    for i in range(seq_len, n):
        X_seq[i - seq_len] = X[i - seq_len : i]
        y_seq[i - seq_len] = y[i]
    return X_seq, y_seq


def train_lstm_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
) -> tuple[LSTMClassifier, StandardScaler]:
    """Обучение LSTM на одном фолде с early stopping по val AUC."""
    set_seeds(SEED)

    # Стандартизация (fit только на train)
    scaler = StandardScaler()
    n_tr = X_train.shape[0]
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    scaler.fit(X_train_flat)
    X_train_s = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_s = scaler.transform(X_val_flat).reshape(X_val.shape)

    # Перевод в тензоры
    X_tr_t = torch.tensor(X_train_s, dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_va_t = torch.tensor(X_val_s, dtype=torch.float32).to(DEVICE)
    y_va_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    model = LSTMClassifier(input_size=n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -np.inf
    best_state = None
    patience_counter = 0

    n_batches = max(1, X_tr_t.shape[0] // LSTM_BATCH)

    for epoch in range(LSTM_EPOCHS):
        model.train()
        perm = torch.randperm(X_tr_t.shape[0])
        epoch_loss = 0.0
        for bi in range(n_batches):
            idx = perm[bi * LSTM_BATCH : (bi + 1) * LSTM_BATCH]
            if len(idx) == 0:
                continue
            x_batch = X_tr_t[idx]
            y_batch = y_tr_t[idx]
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Val AUC для early stopping
        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t).cpu().numpy()
            val_probs = 1.0 / (1.0 + np.exp(-val_logits))
            try:
                val_auc = roc_auc_score(y_val, val_probs)
            except Exception:
                val_auc = 0.5

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= LSTM_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler


def predict_lstm(model: LSTMClassifier, scaler: StandardScaler, X: np.ndarray, n_features: int) -> np.ndarray:
    """Предсказание вероятности на новых данных."""
    if len(X) == 0:
        return np.array([])
    X_flat = X.reshape(-1, n_features)
    X_s = scaler.transform(X_flat).reshape(X.shape)
    X_t = torch.tensor(X_s, dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(X_t).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits))


# ─────────────────────────────────────────────────────────────────────
#  Walk-forward CV для одного символа
# ─────────────────────────────────────────────────────────────────────
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


def evaluate_split(
    sym: str, fold: int, split_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, threshold: float
) -> RowClf:
    """Расчёт метрик для одного split."""
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5
    return RowClf(
        symbol=sym,
        mode="walk",
        fold=fold,
        split=split_name,
        task="direction",
        horizon_days=HORIZON_DAYS,
        model="LSTM",
        n_rows=len(y_true),
        acc=float(accuracy_score(y_true, y_pred)),
        balacc=float(balanced_accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred)),
        auc=auc,
        posrate=float(np.mean(y_true == 1)),
        threshold=threshold,
        extra=f"seq_len={SEQ_LEN},hidden={LSTM_HIDDEN},layers={LSTM_LAYERS},seed={SEED}",
    )


def run_lstm_walk_forward(engine, symbol: str) -> list[RowClf]:
    """Полный walk-forward LSTM для одного символа."""
    print(f"\n[LSTM] === Символ: {symbol} ===")
    df = load_ohlcv(engine, symbol, END_DATE)
    if len(df) < WF_MIN_TRAIN_ROWS + WF_VAL_DAYS + WF_TEST_DAYS:
        print(f"[LSTM] {symbol}: недостаточно данных ({len(df)} строк), пропускаем")
        return []

    df = add_technical_features(df)
    df = add_macro_and_regime_features(df, engine)
    df = df.dropna().reset_index(drop=True)

    feature_cols = get_feature_columns(df)
    n_features = len(feature_cols)
    print(f"[LSTM] {symbol}: {len(df)} строк, {n_features} признаков")

    n = len(df)
    start_idx = WF_MIN_TRAIN_ROWS
    last_start = n - (WF_VAL_DAYS + WF_TEST_DAYS)

    results: list[RowClf] = []
    fold = 0
    i = start_idx

    while i <= last_start:
        tr = df.iloc[:i].copy()
        va = df.iloc[i : i + WF_VAL_DAYS].copy()
        te = df.iloc[i + WF_VAL_DAYS : i + WF_VAL_DAYS + WF_TEST_DAYS].copy()

        # Целевая переменная вычисляется ВНУТРИ split (как в train_baseline.py)
        tr = compute_target_direction(tr, HORIZON_DAYS).dropna(subset=["target_direction"])
        va = compute_target_direction(va, HORIZON_DAYS).dropna(subset=["target_direction"])
        te = compute_target_direction(te, HORIZON_DAYS).dropna(subset=["target_direction"])

        if len(tr) < 500 or len(va) < 80 or len(te) < 80:
            i += WF_STEP_DAYS
            continue

        X_tr_raw = tr[feature_cols].values.astype(np.float32)
        X_va_raw = va[feature_cols].values.astype(np.float32)
        X_te_raw = te[feature_cols].values.astype(np.float32)
        y_tr_raw = tr["target_direction"].values.astype(np.float32)
        y_va_raw = va["target_direction"].values.astype(np.float32)
        y_te_raw = te["target_direction"].values.astype(np.float32)

        # Преобразование в последовательности (seq_len = SEQ_LEN)
        X_tr_seq, y_tr_seq = make_sequences(X_tr_raw, y_tr_raw, SEQ_LEN)
        X_va_seq, y_va_seq = make_sequences(X_va_raw, y_va_raw, SEQ_LEN)
        X_te_seq, y_te_seq = make_sequences(X_te_raw, y_te_raw, SEQ_LEN)

        if len(X_tr_seq) < 100 or len(X_va_seq) < 50 or len(X_te_seq) < 50:
            i += WF_STEP_DAYS
            continue

        print(f"[LSTM] {symbol} fold={fold}: train={len(X_tr_seq)} val={len(X_va_seq)} test={len(X_te_seq)}")

        # Обучение
        model, scaler = train_lstm_one_fold(X_tr_seq, y_tr_seq, X_va_seq, y_va_seq, n_features)

        # Предсказание
        val_probs = predict_lstm(model, scaler, X_va_seq, n_features)
        test_probs = predict_lstm(model, scaler, X_te_seq, n_features)

        # Бинаризация при пороге 0.5
        val_preds = (val_probs >= 0.5).astype(int)
        test_preds = (test_probs >= 0.5).astype(int)

        results.append(evaluate_split(symbol, fold, "val", y_va_seq.astype(int), val_preds, val_probs, 0.5))
        results.append(evaluate_split(symbol, fold, "test", y_te_seq.astype(int), test_preds, test_probs, 0.5))

        last = results[-1]
        print(
            f"[LSTM] {symbol} fold={fold} TEST: AUC={last.auc:.4f} F1={last.f1:.4f} BalAcc={last.balacc:.4f}"
        )

        fold += 1
        i += WF_STEP_DAYS

    return results


# ─────────────────────────────────────────────────────────────────────
#  Сохранение результатов
# ─────────────────────────────────────────────────────────────────────
def append_to_metrics_csv(rows: list[RowClf]) -> None:
    """Добавляет строки LSTM в metrics_walk_direction_k5.csv (как у других моделей)."""
    if not rows:
        print("[LSTM] Нет результатов для сохранения")
        return

    new_df = pd.DataFrame([asdict(r) for r in rows])

    if OUT_CSV.exists():
        old = pd.read_csv(OUT_CSV)
        # Удаляем только LSTM строки для ТЕХ символов, что пересчитываем
        # (чтобы досчёт новых тикеров не стирал старые результаты).
        recomputed_symbols = set(new_df["symbol"].unique())
        mask_drop = (old["model"] == "LSTM") & (old["symbol"].isin(recomputed_symbols))
        old = old[~mask_drop]
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(OUT_CSV, index=False)
    print(f"\n[LSTM] Сохранено {len(rows)} строк в {OUT_CSV}")
    print(f"[LSTM] Общее количество строк в файле: {len(combined)}")


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print(" LSTM baseline для direction forecasting")
    print("=" * 70)
    print(f" Устройство:    {DEVICE}")
    print(f" PyTorch:       {torch.__version__}")
    print(f" Random seed:   {SEED}")
    print(f" END_DATE:      {END_DATE} (фиксация для статьи)")
    print(f" Horizon:       k = {HORIZON_DAYS} дней")
    print(f" Sequence len:  {SEQ_LEN} дней")
    print(f" LSTM hidden:   {LSTM_HIDDEN}, layers: {LSTM_LAYERS}")
    print(f" Epochs (max):  {LSTM_EPOCHS}, patience: {LSTM_PATIENCE}")
    print(f" Символы:       {SYMBOLS}")
    print("=" * 70)

    set_seeds(SEED)
    engine = get_engine()

    all_results: list[RowClf] = []
    for sym in SYMBOLS:
        try:
            sym_results = run_lstm_walk_forward(engine, sym)
            all_results.extend(sym_results)
        except Exception as e:
            print(f"[ERROR] {sym}: {e}")
            import traceback
            traceback.print_exc()

    append_to_metrics_csv(all_results)

    # Сводка
    if all_results:
        df = pd.DataFrame([asdict(r) for r in all_results])
        test_df = df[df["split"] == "test"]
        if not test_df.empty:
            print("\n[LSTM] Сводка по test split:")
            summary = test_df.groupby("symbol").agg(
                auc_mean=("auc", "mean"),
                auc_std=("auc", "std"),
                f1_mean=("f1", "mean"),
                n_folds=("fold", "count"),
            )
            print(summary.to_string())


if __name__ == "__main__":
    main()
