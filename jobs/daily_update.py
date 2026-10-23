"""Daily update: prices -> features -> calibrated forecasts for every horizon.

The forecasts are written to artifacts/predictions_latest_k{5,10,20}.csv, the
files read by the Streamlit dashboard and the Telegram bot (services/predict.py);
artifacts/predictions_latest.csv keeps a copy of the 5-day file for older tools.
The macroeconomic series ^VIX, ^IRX and ^TNX are always ingested, because the
features of a new day cannot be built without them.

Run:
    python -m jobs.daily_update
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MACRO = ("^VIX", "^IRX", "^TNX")
INSTRUMENTS = ("AAPL", "TSLA", "MSFT", "GLD", "^GSPC", "^IXIC", "^DJI", "^RUT")
HORIZONS = (5, 10, 20)

print("[RUN] Starting daily update...")


def _run(args: list[str], extra_env: dict | None = None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    print(f"▶️  {' '.join(args)}  ({extra_env or ''})")
    result = subprocess.run(args, cwd=ROOT, env=env)
    if result.returncode != 0:
        print(f"[ERROR] Failed step: {' '.join(args)}")
        sys.exit(1)


# 1) Ingest new prices for the instruments and the macroeconomic series
symbols = [s.strip() for s in os.environ.get("SYMBOLS", "").replace(" ", ",").split(",") if s.strip()]
symbols = symbols or list(INSTRUMENTS)          # the eight instruments of the paper by default
symbols += [m for m in MACRO if m not in symbols]
_run([sys.executable, "-m", "jobs.ingest_prices"], {"SYMBOLS": ",".join(symbols)})

# 2) Rebuild features
_run([sys.executable, "-m", "jobs.build_features"])

# 3) Inference (direction and volatility are produced together) for every horizon
for h in HORIZONS:
    _run([sys.executable, "-m", "jobs.train_baseline"],
         {"ACTION": "infer", "HORIZON_DAYS": h, "PRED_OUT": ROOT / "artifacts" / f"predictions_latest_k{h}.csv"})
shutil.copyfile(ROOT / "artifacts" / "predictions_latest_k5.csv", ROOT / "artifacts" / "predictions_latest.csv")

print("[OK] Daily update finished successfully.")
print(f"Time: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
