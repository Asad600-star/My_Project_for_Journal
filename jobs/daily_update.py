"""Ежедневное обновление: цены → фичи → инференс direction + volatility.

Запуск:
    python -m jobs.daily_update
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

print("🚀 Запуск ежедневного обновления Master_of_Degree...")


def _run(args: list[str], extra_env: dict | None = None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    print(f"▶️  {' '.join(args)}  ({extra_env or ''})")
    result = subprocess.run(args, cwd=ROOT, env=env)
    if result.returncode != 0:
        print(f"❌ Ошибка на шаге: {' '.join(args)}")
        sys.exit(1)


# 1) Загрузка новых цен
_run([sys.executable, "-m", "jobs.ingest_prices"])

# 2) Пересчёт фич
_run([sys.executable, "-m", "jobs.build_features"])

# 3) Инференс по направлению
_run([sys.executable, "-m", "jobs.train_baseline"], {"ACTION": "infer", "TASK": "direction"})

# 4) Инференс по волатильности
_run([sys.executable, "-m", "jobs.train_baseline"], {"ACTION": "infer", "TASK": "volatility"})

print("✅ Ежедневное обновление завершено успешно!")
print(f"Время: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
