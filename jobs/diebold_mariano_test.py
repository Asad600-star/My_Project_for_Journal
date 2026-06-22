"""
Diebold-Mariano test (Diebold & Mariano 1995) + Harvey-Leybourne-Newbold correction.
======================================================================================

Цель: статистически проверить, что прогнозы предлагаемых моделей
(HYBRID_VOTING, HYBRID_STACK, ExtraTrees, HYBRID_STACK_REG) значимо лучше
прогнозов baseline-моделей (LSTM, ARIMA, GARCH, vanilla XGB/LGBM).

Тест:
    H₀: средняя разница потерь = 0 (модели эквивалентны)
    H₁: средняя разница потерь ≠ 0 (одна модель лучше)

DM статистика (Diebold-Mariano 1995):
    DM = mean(d_t) / sqrt(var_hat(mean(d_t)))
    где d_t = L(e1_t) - L(e2_t), L — функция потерь

Harvey-Leybourne-Newbold (HLN) поправка для малой выборки:
    DM_HLN = DM × sqrt((T + 1 - 2h + h(h-1)/T) / T)
    где h — горизонт прогноза, T — количество наблюдений

Запуск:
    python -m jobs.diebold_mariano_test

Результат сохраняется в artifacts/dm_test_results.csv
для прямой вставки в статью (таблица DM значимости).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HORIZON_DAYS = int(os.environ.get("HORIZON_DAYS", "5"))
ARTIFACTS_DIR = Path(os.environ.get("METRICS_DIR", "artifacts"))


# ─────────────────────────────────────────────────────────────────────
#  Diebold-Mariano statistic (с HLN поправкой для малой выборки)
# ─────────────────────────────────────────────────────────────────────
def dm_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    horizon: int = 1,
    alternative: str = "two-sided",
) -> dict:
    """
    Diebold-Mariano test со включённой Harvey-Leybourne-Newbold поправкой.

    Args:
        loss_a: вектор потерь модели A (например, RMSE по каждому fold)
        loss_b: вектор потерь модели B
        horizon: горизонт прогноза (h в HLN формуле)
        alternative: 'two-sided' | 'less' (A < B) | 'greater' (A > B)

    Returns:
        dict с DM статистикой, p-value, выводом, кол-вом наблюдений.
    """
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)

    if loss_a.shape != loss_b.shape:
        raise ValueError(f"Shapes don't match: {loss_a.shape} vs {loss_b.shape}")

    # Удаляем NaN
    mask = np.isfinite(loss_a) & np.isfinite(loss_b)
    loss_a = loss_a[mask]
    loss_b = loss_b[mask]
    T = len(loss_a)

    if T < 4:
        return {
            "T": T,
            "mean_diff": np.nan,
            "DM": np.nan,
            "DM_HLN": np.nan,
            "p_value": np.nan,
            "conclusion": "Недостаточно данных (T < 4)",
        }

    d = loss_a - loss_b
    mean_d = float(np.mean(d))

    # Long-run variance с использованием авто-ковариаций до lag = h-1
    gamma_0 = float(np.var(d, ddof=1))
    var_d = gamma_0
    for k in range(1, horizon):
        if k >= T:
            break
        gamma_k = float(np.mean((d[k:] - mean_d) * (d[:-k] - mean_d)))
        var_d += 2.0 * gamma_k

    var_d = max(var_d, 1e-12)
    se = float(np.sqrt(var_d / T))

    # Стандартная DM статистика
    DM = mean_d / se

    # HLN поправка (Harvey, Leybourne, Newbold 1997)
    hln_correction = np.sqrt((T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T)
    DM_HLN = DM * hln_correction

    # P-value через t-распределение с T-1 степенями свободы (для малых выборок)
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.t.cdf(abs(DM_HLN), df=T - 1))
    elif alternative == "less":
        p_value = stats.t.cdf(DM_HLN, df=T - 1)
    elif alternative == "greater":
        p_value = 1 - stats.t.cdf(DM_HLN, df=T - 1)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Интерпретация
    if p_value < 0.01:
        sig = "*** p<0.01"
    elif p_value < 0.05:
        sig = "** p<0.05"
    elif p_value < 0.10:
        sig = "* p<0.10"
    else:
        sig = "n.s."

    if mean_d < 0:
        conclusion = f"Модель A лучше (mean(L_A - L_B) = {mean_d:+.6f}, {sig})"
    elif mean_d > 0:
        conclusion = f"Модель B лучше (mean(L_A - L_B) = {mean_d:+.6f}, {sig})"
    else:
        conclusion = f"Модели эквивалентны ({sig})"

    return {
        "T": T,
        "mean_diff": mean_d,
        "DM": DM,
        "DM_HLN": DM_HLN,
        "p_value": p_value,
        "significance": sig,
        "conclusion": conclusion,
    }


# ─────────────────────────────────────────────────────────────────────
#  Загрузка fold-level RMSE / AUC из метрик
# ─────────────────────────────────────────────────────────────────────
def load_test_metrics(task: str) -> pd.DataFrame:
    """Загружает test split метрики для walk-forward модели."""
    csv = ARTIFACTS_DIR / f"metrics_walk_{task}_k{HORIZON_DAYS}.csv"
    if not csv.exists():
        print(f"[WARN] Файл не найден: {csv}")
        return pd.DataFrame()
    df = pd.read_csv(csv)
    df = df[df["split"] == "test"].copy()
    return df


def get_fold_metric(
    df: pd.DataFrame, symbol: str, model: str, metric: str
) -> np.ndarray:
    """Извлекает значения метрики (rmse/auc/...) по фолдам для (symbol, model)."""
    sub = df[(df["symbol"] == symbol) & (df["model"] == model)].copy()
    if sub.empty:
        return np.array([])
    sub = sub.sort_values("fold")
    return sub[metric].values


def align_folds(
    a_values: np.ndarray, b_values: np.ndarray, a_folds: np.ndarray, b_folds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Сравнивает только на общих fold (intersection)."""
    common = np.intersect1d(a_folds, b_folds)
    a_aligned = a_values[np.isin(a_folds, common)]
    b_aligned = b_values[np.isin(b_folds, common)]
    return a_aligned, b_aligned


def get_aligned_metric(
    df: pd.DataFrame, symbol: str, model_a: str, model_b: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает метрики двух моделей на общих фолдах."""
    sub_a = df[(df["symbol"] == symbol) & (df["model"] == model_a)].sort_values("fold")
    sub_b = df[(df["symbol"] == symbol) & (df["model"] == model_b)].sort_values("fold")
    if sub_a.empty or sub_b.empty:
        return np.array([]), np.array([])
    folds_a = sub_a["fold"].values
    folds_b = sub_b["fold"].values
    common = np.intersect1d(folds_a, folds_b)
    a_vals = sub_a[sub_a["fold"].isin(common)][metric].values
    b_vals = sub_b[sub_b["fold"].isin(common)][metric].values
    return a_vals, b_vals


# ─────────────────────────────────────────────────────────────────────
#  Сравнения для статьи
# ─────────────────────────────────────────────────────────────────────
#
# Direction (AUC, выше — лучше; используем -AUC как loss для DM):
#   HYBRID_VOTING vs LSTM
#   HYBRID_VOTING vs LOGREG, HGB, XGB, LGBM, RF
#   HYBRID_STACK vs LSTM
#   HYBRID_VOTING vs HYBRID_STACK
#
# Volatility (RMSE, ниже — лучше; используем RMSE напрямую как loss):
#   EXTRATREES vs ARIMA
#   EXTRATREES vs GARCH
#   EXTRATREES vs HYBRID_STACK_REG
#   EXTRATREES vs XGB, LGBM, HGB
# ─────────────────────────────────────────────────────────────────────

SYMBOLS = ["AAPL", "TSLA", "^GSPC", "^IXIC", "^DJI", "^RUT", "GLD", "MSFT"]

DIRECTION_COMPARISONS = [
    # (наша модель, baseline)
    ("HYBRID_VOTING", "LSTM"),
    ("HYBRID_VOTING", "LOGREG"),
    ("HYBRID_VOTING", "HGB"),
    ("HYBRID_VOTING", "XGB"),
    ("HYBRID_VOTING", "LGBM"),
    ("HYBRID_VOTING", "RF"),
    ("HYBRID_STACK", "LSTM"),
    ("HYBRID_STACK", "LOGREG"),
    ("HYBRID_VOTING", "HYBRID_STACK"),
]

VOLATILITY_COMPARISONS = [
    ("EXTRATREES", "ARIMA"),
    ("EXTRATREES", "GARCH"),
    ("EXTRATREES", "XGB"),
    ("EXTRATREES", "LGBM"),
    ("EXTRATREES", "HGB"),
    ("EXTRATREES", "HYBRID_STACK_REG"),
]


def run_direction_dm_tests(df_dir: pd.DataFrame) -> list[dict]:
    """DM тесты для direction (AUC -> loss = -AUC)."""
    results = []
    for our_model, baseline in DIRECTION_COMPARISONS:
        for sym in SYMBOLS:
            our_aucs, base_aucs = get_aligned_metric(df_dir, sym, our_model, baseline, "auc")
            if len(our_aucs) < 4 or len(base_aucs) < 4:
                continue
            # Loss = -AUC (так как чем выше AUC, тем меньше loss)
            our_loss = -our_aucs
            base_loss = -base_aucs
            # Тест: наша модель < baseline по loss (т.е. AUC выше)
            res = dm_test(our_loss, base_loss, horizon=HORIZON_DAYS, alternative="less")
            results.append({
                "task": "direction",
                "symbol": sym,
                "model_A": our_model,
                "model_B": baseline,
                "metric": "AUC",
                "T": res["T"],
                "mean_A": float(np.mean(our_aucs)),
                "mean_B": float(np.mean(base_aucs)),
                "mean_diff_AB": res["mean_diff"],
                "DM_stat": res["DM"],
                "DM_HLN_stat": res["DM_HLN"],
                "p_value": res["p_value"],
                "significance": res["significance"],
                "conclusion": res["conclusion"].replace("Модель A", our_model).replace("Модель B", baseline),
            })
    return results


def run_volatility_dm_tests(df_vol: pd.DataFrame) -> list[dict]:
    """DM тесты для volatility (RMSE -> loss = RMSE)."""
    results = []
    for our_model, baseline in VOLATILITY_COMPARISONS:
        for sym in SYMBOLS:
            our_rmse, base_rmse = get_aligned_metric(df_vol, sym, our_model, baseline, "rmse")
            if len(our_rmse) < 4 or len(base_rmse) < 4:
                continue
            # Тест: наша модель < baseline по loss (т.е. RMSE меньше)
            res = dm_test(our_rmse, base_rmse, horizon=HORIZON_DAYS, alternative="less")
            results.append({
                "task": "volatility",
                "symbol": sym,
                "model_A": our_model,
                "model_B": baseline,
                "metric": "RMSE",
                "T": res["T"],
                "mean_A": float(np.mean(our_rmse)),
                "mean_B": float(np.mean(base_rmse)),
                "mean_diff_AB": res["mean_diff"],
                "DM_stat": res["DM"],
                "DM_HLN_stat": res["DM_HLN"],
                "p_value": res["p_value"],
                "significance": res["significance"],
                "conclusion": res["conclusion"].replace("Модель A", our_model).replace("Модель B", baseline),
            })
    return results


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print(" Diebold-Mariano test (с HLN поправкой)")
    print("=" * 70)
    print(f" Horizon h = {HORIZON_DAYS}")
    print(f" H₀: модели прогнозируют одинаково хорошо")
    print(f" H₁: предлагаемая модель прогнозирует ЛУЧШЕ (one-sided)")
    print("=" * 70)

    df_dir = load_test_metrics("direction")
    df_vol = load_test_metrics("volatility")

    print(f"\n[DATA] Direction CSV: {len(df_dir)} строк test split")
    print(f"[DATA] Direction models: {sorted(df_dir['model'].unique())}")
    print(f"\n[DATA] Volatility CSV: {len(df_vol)} строк test split")
    print(f"[DATA] Volatility models: {sorted(df_vol['model'].unique())}")

    all_results = []
    print("\n" + "=" * 70)
    print(" DM-тесты: DIRECTION (AUC)")
    print("=" * 70)
    dir_results = run_direction_dm_tests(df_dir)
    all_results.extend(dir_results)
    for r in dir_results:
        print(
            f"  {r['model_A']:>15} vs {r['model_B']:<10} | {r['symbol']:<6} | "
            f"T={r['T']:>2} | mean(A)={r['mean_A']:.4f} mean(B)={r['mean_B']:.4f} | "
            f"DM_HLN={r['DM_HLN_stat']:>6.3f} p={r['p_value']:.4f} {r['significance']}"
        )

    print("\n" + "=" * 70)
    print(" DM-тесты: VOLATILITY (RMSE)")
    print("=" * 70)
    vol_results = run_volatility_dm_tests(df_vol)
    all_results.extend(vol_results)
    for r in vol_results:
        print(
            f"  {r['model_A']:>15} vs {r['model_B']:<18} | {r['symbol']:<6} | "
            f"T={r['T']:>2} | mean(A)={r['mean_A']:.6f} mean(B)={r['mean_B']:.6f} | "
            f"DM_HLN={r['DM_HLN_stat']:>6.3f} p={r['p_value']:.4f} {r['significance']}"
        )

    # Сохраняем
    if all_results:
        out_path = ARTIFACTS_DIR / "dm_test_results.csv"
        pd.DataFrame(all_results).to_csv(out_path, index=False)
        print(f"\n[ARTIFACT] Результаты DM-тестов сохранены: {out_path}")

        # Сводка значимости
        df_res = pd.DataFrame(all_results)
        sig_count = sum(df_res["p_value"] < 0.05)
        print(f"\n[SUMMARY] Статистически значимых сравнений (p < 0.05): {sig_count} / {len(df_res)}")
        print(f"[SUMMARY] Очень значимых (p < 0.01): {sum(df_res['p_value'] < 0.01)} / {len(df_res)}")


if __name__ == "__main__":
    main()
