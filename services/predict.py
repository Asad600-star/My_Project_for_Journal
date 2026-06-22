"""High-level prediction service.

Использует:
- закэшированные joblib-модели (если есть) — быстрый путь без переобучения,
- subprocess только когда явно нужен refresh (новая загрузка цен / пересчёт фич).
"""

import os
import subprocess
import sys
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from core.risk.risk_manager import RiskManager

ARTIFACTS_DIR = ROOT / "artifacts"
PREDICTIONS_FILE = ARTIFACTS_DIR / "predictions_latest.csv"

risk_manager = RiskManager()

NAME_MAP = {
    "AAPL": "Apple Inc.",
    "TSLA": "Tesla Inc.",
    "MSFT": "Microsoft Corp.",
    "GLD": "SPDR Gold Trust",
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq Composite",
    "^DJI": "Dow Jones Industrial Average",
    "^RUT": "Russell 2000",
}

# Поддерживаемые горизонты прогноза (дней вперёд)
SUPPORTED_HORIZONS = [5, 10, 20]


def _run(cmd: list[str], extra_env: dict | None = None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _refresh_prices_and_features():
    _run([sys.executable, "-m", "jobs.ingest_prices"])
    _run([sys.executable, "-m", "jobs.build_features"])


def _predictions_file(horizon: int) -> Path:
    """Отдельный файл прогнозов на каждый горизонт."""
    return ARTIFACTS_DIR / f"predictions_latest_k{horizon}.csv"


def _run_inference(symbol: str | None = None, horizon: int = 5):
    extra_env = {
        "ACTION": "infer",
        "HORIZON_DAYS": str(horizon),
        "PRED_OUT": str(_predictions_file(horizon)),
    }
    if symbol:
        extra_env["ONLY_SYMBOL"] = symbol
    _run([sys.executable, "-m", "jobs.train_baseline"], extra_env)


def get_prediction(symbol: str, horizon: int = 5, refresh: bool = False) -> dict:
    """Возвращает прогноз по символу на заданный горизонт (5, 10 или 20 дней).

    horizon       → горизонт прогноза в торговых днях (по умолчанию 5).
    refresh=True  → обновить цены, пересчитать фичи, заново посчитать инференс.
    refresh=False → попытаться вернуть последнее сохранённое для этого горизонта;
                    если нет строки для символа — запустить инференс на лету.
    """
    symbol = symbol.strip().upper()
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(f"Горизонт {horizon} не поддерживается. Доступны: {SUPPORTED_HORIZONS}")

    pred_file = _predictions_file(horizon)

    if refresh:
        _refresh_prices_and_features()
        _run_inference(symbol, horizon)
    elif not pred_file.exists():
        _run_inference(symbol, horizon)

    df = pd.read_csv(pred_file)
    matched = df[df["symbol"] == symbol]
    if matched.empty:
        # Не было прогноза для этого символа на этом горизонте — считаем
        _run_inference(symbol, horizon)
        df = pd.read_csv(pred_file)
        matched = df[df["symbol"] == symbol]
        if matched.empty:
            raise RuntimeError(f"Нет прогноза для {symbol} (горизонт {horizon}) даже после инференса")

    row = matched.iloc[-1].to_dict()

    pred = {
        "symbol": symbol,
        "name_ru": NAME_MAP.get(symbol, symbol),
        "horizon_days": horizon,
        "asof_date": row["asof_date"],
        "p_up": round(float(row["p_up"]), 4),
        "vol_pred": round(float(row["vol_pred"]), 4),
    }
    pred = risk_manager.add_to_prediction(pred)

    # === SHAP ===
    shap_file = ARTIFACTS_DIR / f"shap_{symbol}_direction.json"
    if shap_file.exists():
        try:
            with open(shap_file, encoding="utf-8") as f:
                data = json.load(f)
            shap_values = data.get("shap_values", [])
            if isinstance(shap_values, list) and len(shap_values) > 0 and isinstance(shap_values[0], list):
                shap_values = shap_values[0]
            pred["shap_values"] = shap_values
            pred["shap_feature_names"] = data.get("feature_names", [])
            pred["shap_base_value"] = data.get("base_value", 0.0)
            top = sorted(zip(pred["shap_feature_names"], shap_values), key=lambda x: abs(x[1]), reverse=True)[:8]
            pred["shap_top_factors_ru"] = [f"{name} ({val:+.4f})" for name, val in top]
        except Exception as e:
            pred["shap_top_factors_ru"] = [f"Ошибка SHAP: {e}"]
    else:
        pred["shap_top_factors_ru"] = ["SHAP пока не посчитан"]

    return pred
