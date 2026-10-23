"""
Modern deep-learning baselines on the canonical 56-feature vector.
==================================================================

Trains three sequence models — LSTM, GRU and a Transformer encoder — under the
SAME expanding-window walk-forward protocol and on the SAME 56 features as the
production tree ensembles, for BOTH tasks:

  - direction  (binary classification, metric AUC/F1/bal-acc)
  - volatility (regression, metric RMSE/MAE/R2)

Results are appended to artifacts/metrics_walk_direction_k5.csv and
artifacts/metrics_walk_volatility_k5.csv (replacing any previous rows for these
model names), so they flow directly into Tables 5-6 and the critical-difference diagrams.

Run:
    python -m jobs.train_deep_baselines                 # both tasks, all models
    TASK=volatility MODELS=TRANSFORMER python -m jobs.train_deep_baselines
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             balanced_accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score)

from jobs.train_baseline import add_technical_features, build_feature_matrix, compute_targets
from jobs.sequence_windows import make_sequences

load_dotenv()

SEED = int(os.environ.get("SEED", "42"))
END_DATE = os.environ.get("END_DATE", "2026-06-01")
HORIZON_DAYS = int(os.environ.get("HORIZON_DAYS", "5"))
WF_MIN_TRAIN_ROWS = int(os.environ.get("WF_MIN_TRAIN_ROWS", "1200"))
WF_VAL_DAYS = int(os.environ.get("WF_VAL_DAYS", "126"))
WF_TEST_DAYS = int(os.environ.get("WF_TEST_DAYS", "126"))
WF_STEP_DAYS = int(os.environ.get("WF_STEP_DAYS", "63"))

SEQ_LEN = int(os.environ.get("SEQ_LEN", "20"))
HIDDEN = int(os.environ.get("DEEP_HIDDEN", "32"))
EPOCHS = int(os.environ.get("DEEP_EPOCHS", "50"))
PATIENCE = int(os.environ.get("DEEP_PATIENCE", "10"))
LR = float(os.environ.get("DEEP_LR", "1e-3"))
BATCH = int(os.environ.get("DEEP_BATCH", "32"))

SYMBOLS_DEFAULT = ["AAPL", "TSLA", "MSFT", "GLD", "^GSPC", "^IXIC", "^DJI", "^RUT"]
_env = os.environ.get("SYMBOLS", "")
SYMBOLS = [s.strip() for s in _env.split(",") if s.strip()] or SYMBOLS_DEFAULT
TASKS = [t.strip() for t in os.environ.get("TASK", "direction,volatility").split(",") if t.strip()]
MODELS = [m.strip().upper() for m in os.environ.get("MODELS", "LSTM,GRU,TRANSFORMER").split(",") if m.strip()]

A = Path(os.environ.get("METRICS_DIR", "artifacts"))


def set_seeds(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    forced = os.environ.get("DEEP_DEVICE")              # e.g. "cpu" to reproduce without a GPU
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


def get_engine():
    return create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)


_SQL = text("""
    SELECT symbol, date, open, high, low, close, volume,
           return_1d, log_return,
           sma_5, volatility_5, sma_10, volatility_10, sma_20, volatility_20,
           return_lag_1, return_lag_2, return_lag_3, return_lag_4, return_lag_5,
           mkt_return_1d, mkt_log_return, mkt_mom_5, mkt_mom_10, mkt_mom_20, mkt_vol_20,
           vix_level, vix_return_1d, vix_change_1d,
           irx_level, irx_change_1d, tnx_level, tnx_change_1d
    FROM features_daily WHERE symbol = :s ORDER BY date ASC
""")


def load_features(engine, sym):
    df = pd.read_sql(_SQL, engine, params={"s": sym}, parse_dates=["date"])
    return df[df["date"] <= pd.to_datetime(END_DATE)].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
#  Models (seq -> one)
# ─────────────────────────────────────────────────────────────────────
class RecurrentNet(nn.Module):
    def __init__(self, kind: str, n_features: int, hidden=HIDDEN, layers=2, dropout=0.2):
        super().__init__()
        rnn = nn.LSTM if kind == "LSTM" else nn.GRU
        self.rnn = rnn(n_features, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden // 2),
                                  nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TransformerNet(nn.Module):
    def __init__(self, n_features: int, d_model=64, nhead=4, layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2 * d_model,
                                         dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))

    def forward(self, x):
        z = self.proj(x) + self.pos[:, : x.shape[1], :]
        z = self.enc(z)
        return self.head(z[:, -1, :]).squeeze(-1)


class NBeatsBlock(nn.Module):
    """Generic N-BEATS block: FC stack -> backcast + forecast (with dropout)."""
    def __init__(self, in_dim, hidden, dropout=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.backcast = nn.Linear(hidden, in_dim)
        self.forecast = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.fc(x)
        return self.backcast(h), self.forecast(h)


class NBeatsNet(nn.Module):
    """N-BEATSx-style generic stack with exogenous inputs (the flattened
    SEQ_LEN x n_features window), with doubly-residual block stacking. A compact,
    regularised configuration is used to avoid over-fitting the small per-fold
    training windows of daily financial data."""
    def __init__(self, n_features, nblocks=3, hidden=64):
        super().__init__()
        self.in_dim = SEQ_LEN * n_features
        self.blocks = nn.ModuleList([NBeatsBlock(self.in_dim, hidden) for _ in range(nblocks)])

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        res = x
        fsum = 0.0
        for b in self.blocks:
            bc, fc = b(res)
            res = res - bc
            fsum = fsum + fc
        return fsum.squeeze(-1)


def make_model(name, n_features):
    if name in ("LSTM", "GRU"):
        return RecurrentNet(name, n_features)
    if name == "TRANSFORMER":
        return TransformerNet(n_features)
    if name == "NBEATSX":
        return NBeatsNet(n_features)
    raise ValueError(name)


def train_one(name, task, Xtr, ytr, Xva, yva, n_features):
    set_seeds(SEED)
    model = make_model(name, n_features).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss() if task == "direction" else nn.MSELoss()
    Xtr_t = torch.tensor(Xtr).to(DEVICE); ytr_t = torch.tensor(ytr).to(DEVICE)
    Xva_t = torch.tensor(Xva).to(DEVICE)
    best, best_state, wait = (np.inf if task == "volatility" else -np.inf), None, 0
    nb = max(1, Xtr_t.shape[0] // BATCH)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(Xtr_t.shape[0])
        for bi in range(nb):
            idx = perm[bi * BATCH:(bi + 1) * BATCH]
            if len(idx) == 0:
                continue
            opt.zero_grad()
            loss = crit(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xva_t).cpu().numpy()
        if task == "direction":
            prob = 1 / (1 + np.exp(-vp))
            try:
                score = roc_auc_score(yva, prob)
            except Exception:
                score = 0.5
            improved = score > best
        else:
            score = -np.sqrt(mean_squared_error(yva, vp))   # higher = better
            improved = score > best
        if improved:
            best, best_state, wait = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def predict(model, Xte):
    if len(Xte) == 0:
        return np.array([])
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(Xte).to(DEVICE)).cpu().numpy()


def walk_forward(engine, sym, name, task):
    if name == "NBEATSX" and task == "direction":
        return []   # N-BEATSx is a volatility (regression) baseline only
    df = load_features(engine, sym)
    if df.empty:
        return []
    df = add_technical_features(df)
    df, feat = build_feature_matrix(df)
    if df.empty:
        return []
    n = len(df)
    rows = []
    fold = 0
    i = WF_MIN_TRAIN_ROWS
    last = n - (WF_VAL_DAYS + WF_TEST_DAYS)
    while i <= last:
        tr = compute_targets(df.iloc[:i].copy(), HORIZON_DAYS)
        va = compute_targets(df.iloc[i:i + WF_VAL_DAYS].copy(), HORIZON_DAYS)
        te = compute_targets(df.iloc[i + WF_VAL_DAYS:i + WF_VAL_DAYS + WF_TEST_DAYS].copy(), HORIZON_DAYS)
        if len(tr) < 500 or len(va) < 80 or len(te) < 80:
            i += WF_STEP_DAYS; continue
        ycol = "target_direction" if task == "direction" else "target_vol_kd"
        scaler = StandardScaler().fit(tr[feat].values)          # training rows only
        X_all = scaler.transform(df[feat].values).astype(np.float32)
        pos = {"tr": np.arange(len(tr)), "va": np.arange(i, i + len(va)),
               "te": np.arange(i + WF_VAL_DAYS, i + WF_VAL_DAYS + len(te))}
        Xtr, ytr = make_sequences(X_all, pos["tr"], tr[ycol].values, SEQ_LEN)
        Xva, yva = make_sequences(X_all, pos["va"], va[ycol].values, SEQ_LEN)
        Xte, yte = make_sequences(X_all, pos["te"], te[ycol].values, SEQ_LEN)
        if len(Xtr) < 100 or len(Xte) == 0:
            i += WF_STEP_DAYS; fold += 1; continue
        # standardise regression target on train
        if task == "volatility":
            mu, sd = ytr.mean(), ytr.std() + 1e-12
            ytr_s, yva_s = (ytr - mu) / sd, (yva - mu) / sd
        else:
            ytr_s, yva_s = ytr, yva
        model = train_one(name, task, Xtr, ytr_s, Xva, yva_s, len(feat))
        pred = predict(model, Xte)
        if task == "direction":
            prob = 1 / (1 + np.exp(-pred))
            yhat = (prob >= 0.5).astype(int)
            try:
                auc = float(roc_auc_score(yte, prob)) if len(np.unique(yte)) > 1 else 0.5
            except Exception:
                auc = 0.5
            rows.append(dict(symbol=sym, mode="walk", fold=fold, split="test", task="direction",
                             horizon_days=HORIZON_DAYS, model=name, n_rows=len(yte),
                             acc=float(accuracy_score(yte, yhat)), balacc=float(balanced_accuracy_score(yte, yhat)),
                             f1=float(f1_score(yte, yhat, zero_division=0)), auc=auc,
                             posrate=float(np.mean(yte == 1)), threshold=0.5,
                             extra=f"seq={SEQ_LEN},hidden={HIDDEN},seed={SEED}"))
        else:
            pred_un = pred * sd + mu
            rows.append(dict(symbol=sym, mode="walk", fold=fold, split="test", task="volatility",
                             horizon_days=HORIZON_DAYS, model=name, n_rows=len(yte),
                             mae=float(mean_absolute_error(yte, pred_un)),
                             rmse=float(np.sqrt(mean_squared_error(yte, pred_un))),
                             r2=float(r2_score(yte, pred_un)),
                             extra=f"seq={SEQ_LEN},hidden={HIDDEN},seed={SEED}"))
        print(f"[{name}/{task}] {sym} fold={fold} done")
        fold += 1
        i += WF_STEP_DAYS
    return rows


def append_csv(path, rows, task):
    cols_dir = ["symbol", "mode", "fold", "split", "task", "horizon_days", "model", "n_rows",
                "acc", "balacc", "f1", "auc", "posrate", "threshold", "extra"]
    cols_vol = ["symbol", "mode", "fold", "split", "task", "horizon_days", "model", "n_rows",
                "mae", "rmse", "r2", "extra"]
    cols = cols_dir if task == "direction" else cols_vol
    new = pd.DataFrame(rows)[cols]
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["model"].isin(new["model"].unique())]   # replace these models
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(path, index=False)
    print(f"[ARTIFACT] {path}: +{len(new)} rows ({sorted(new['model'].unique())})")


def main():
    print(f"device={DEVICE} models={MODELS} tasks={TASKS} symbols={len(SYMBOLS)}")
    engine = get_engine()
    for task in TASKS:
        path = A / f"metrics_walk_{task}_k{HORIZON_DAYS}.csv"
        all_rows = []
        for name in MODELS:
            for sym in SYMBOLS:
                all_rows += walk_forward(engine, sym, name, task)
        if all_rows:
            append_csv(path, all_rows, task)


if __name__ == "__main__":
    main()
