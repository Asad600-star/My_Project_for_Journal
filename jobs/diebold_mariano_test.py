"""
Diebold-Mariano test (Diebold & Mariano 1995) + Harvey-Leybourne-Newbold correction.
======================================================================================

Purpose: statistically test whether the forecasts of the proposed models
(HYBRID_VOTING, HYBRID_STACK, HYBRID_ECON_ET, ExtraTrees) are significantly better than
the baseline models (LSTM, GARCH, HAR, vanilla XGB/LGBM, naive benchmarks).

Test:
    H0: mean loss difference = 0 (models are equivalent)
    H1: mean loss difference != 0 (one model is better)

DM statistic (Diebold-Mariano 1995):
    DM = mean(d_t) / sqrt(var_hat(mean(d_t)))
    where d_t = L(e1_t) - L(e2_t), L is the loss function

Harvey-Leybourne-Newbold (HLN) small-sample correction:
    DM_HLN = DM × sqrt((T + 1 - 2h + h(h-1)/T) / T)
    where h is the forecast horizon, T the number of observations

Run:
    python -m jobs.diebold_mariano_test

Output is saved to artifacts/dm_test_results.csv
for direct inclusion in the paper (DM significance table).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HORIZON_DAYS = int(os.environ.get("HORIZON_DAYS", "5"))
ARTIFACTS_DIR = Path(os.environ.get("METRICS_DIR", "artifacts"))
# The fold-level tests compare one metric per walk-forward fold. Consecutive test
# blocks (126 days, step 63) overlap by half, so the loss differential of the fold
# series is MA(1): the HAC lag of the test is h - 1 = 1, i.e. h = ceil(test / step) = 2.
# (h = HORIZON_DAYS applies to daily forecasts of overlapping k-day targets, as in
# jobs/volatility_models.py.)
FOLD_H = int(np.ceil(int(os.environ.get("WF_TEST_DAYS", "126")) / int(os.environ.get("WF_STEP_DAYS", "63"))))


# ─────────────────────────────────────────────────────────────────────
#  Diebold-Mariano statistic (with HLN small-sample correction)
# ─────────────────────────────────────────────────────────────────────
def dm_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    horizon: int = 1,
    alternative: str = "two-sided",
) -> dict:
    """
    Diebold-Mariano test with the Harvey-Leybourne-Newbold correction applied.

    Args:
        loss_a: loss vector of model A (e.g. RMSE per fold)
        loss_b: loss vector of model B
        horizon: forecast horizon (h in the HLN formula)
        alternative: 'two-sided' | 'less' (A < B) | 'greater' (A > B)

    Returns:
        dict with the DM statistic, p-value, conclusion, and number of observations.
    """
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)

    if loss_a.shape != loss_b.shape:
        raise ValueError(f"Shapes don't match: {loss_a.shape} vs {loss_b.shape}")

    # Drop NaN
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
            "conclusion": "Not enough data (T < 4)",
        }

    d = loss_a - loss_b
    mean_d = float(np.mean(d))

    # Long-run variance using autocovariances up to lag = h-1 (gamma_0 with ddof=1,
    # which makes the statistic slightly conservative in small samples)
    gamma_0 = float(np.var(d, ddof=1))
    var_d = gamma_0
    for k in range(1, horizon):
        if k >= T:
            break
        gamma_k = float(np.mean((d[k:] - mean_d) * (d[:-k] - mean_d)))
        var_d += 2.0 * gamma_k

    var_d = max(var_d, 1e-12)
    se = float(np.sqrt(var_d / T))

    # Standard DM statistic
    DM = mean_d / se

    # HLN correction (Harvey, Leybourne, Newbold 1997)
    hln_correction = np.sqrt((T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T)
    DM_HLN = DM * hln_correction

    # P-value via t-distribution with T-1 degrees of freedom (small samples)
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.t.cdf(abs(DM_HLN), df=T - 1))
    elif alternative == "less":
        p_value = stats.t.cdf(DM_HLN, df=T - 1)
    elif alternative == "greater":
        p_value = 1 - stats.t.cdf(DM_HLN, df=T - 1)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Interpretation
    if p_value < 0.01:
        sig = "*** p<0.01"
    elif p_value < 0.05:
        sig = "** p<0.05"
    elif p_value < 0.10:
        sig = "* p<0.10"
    else:
        sig = "n.s."

    if mean_d < 0:
        conclusion = f"Model A is better (mean(L_A - L_B) = {mean_d:+.6f}, {sig})"
    elif mean_d > 0:
        conclusion = f"Model B is better (mean(L_A - L_B) = {mean_d:+.6f}, {sig})"
    else:
        conclusion = f"Models are equivalent ({sig})"

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
#  Load fold-level RMSE / AUC from the metrics files
# ─────────────────────────────────────────────────────────────────────
def load_test_metrics(task: str) -> pd.DataFrame:
    """Loads the test-split metrics of a walk-forward model."""
    csv = ARTIFACTS_DIR / f"metrics_walk_{task}_k{HORIZON_DAYS}.csv"
    if not csv.exists():
        print(f"[WARN] File not found: {csv}")
        return pd.DataFrame()
    df = pd.read_csv(csv)
    df = df[df["split"] == "test"].copy()
    return df


def get_fold_metric(
    df: pd.DataFrame, symbol: str, model: str, metric: str
) -> np.ndarray:
    """Extracts a metric (rmse/auc/...) per fold for a (symbol, model) pair."""
    sub = df[(df["symbol"] == symbol) & (df["model"] == model)].copy()
    if sub.empty:
        return np.array([])
    sub = sub.sort_values("fold")
    return sub[metric].values


def align_folds(
    a_values: np.ndarray, b_values: np.ndarray, a_folds: np.ndarray, b_folds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compares only on the common folds (intersection)."""
    common = np.intersect1d(a_folds, b_folds)
    a_aligned = a_values[np.isin(a_folds, common)]
    b_aligned = b_values[np.isin(b_folds, common)]
    return a_aligned, b_aligned


def get_aligned_metric(
    df: pd.DataFrame, symbol: str, model_a: str, model_b: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the two models metrics on the common folds."""
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
#  Comparisons for the paper
# ─────────────────────────────────────────────────────────────────────
#
# Direction (AUC, higher is better; we use -AUC as the DM loss):
#   HYBRID_VOTING vs LSTM
#   HYBRID_VOTING vs LOGREG, HGB, XGB, LGBM, RF
#   HYBRID_STACK vs LSTM
#   HYBRID_VOTING vs HYBRID_STACK
#
# Volatility (RMSE, lower is better; we use RMSE directly as the loss):
#   HYBRID_ECON_ET vs GARCH (both directions), GJR_GARCH, ECON, HAR, EXTRATREES
#   GARCH, HAR vs EXTRATREES
#   EXTRATREES vs HYBRID_STACK_REG, XGB, LGBM, HGB
#   HYBRID_ECON_ET, EXTRATREES vs the historical mean; EXTRATREES vs persistence
# (daily-level tests with squared-error and QLIKE losses: jobs/volatility_models.py)
# ─────────────────────────────────────────────────────────────────────

SYMBOLS = ["AAPL", "TSLA", "^GSPC", "^IXIC", "^DJI", "^RUT", "GLD", "MSFT"]

DIRECTION_COMPARISONS = [
    # (our model, baseline)
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
    # the system's volatility model against the econometric baselines and ExtraTrees;
    # GARCH vs the hybrid is tested in both directions because the one-sided test only
    # detects an advantage of model A
    ("HYBRID_ECON_ET", "GARCH"),
    ("GARCH", "HYBRID_ECON_ET"),
    ("HYBRID_ECON_ET", "GJR_GARCH"),
    ("HYBRID_ECON_ET", "ECON"),
    ("HYBRID_ECON_ET", "HAR"),
    ("HYBRID_ECON_ET", "EXTRATREES"),
    ("GARCH", "EXTRATREES"),
    ("HAR", "EXTRATREES"),
    ("EXTRATREES", "XGB"),
    ("EXTRATREES", "LGBM"),
    ("EXTRATREES", "HGB"),
    ("EXTRATREES", "HYBRID_STACK_REG"),
    # feasible constant benchmark: the training-sample mean of the target. It is the
    # reference for the Campbell-Thompson out-of-sample R2 and resolves the apparent
    # conflict between a lower RMSE and a negative R2 measured against the test mean.
    ("HYBRID_ECON_ET", "BASELINE_MEAN_TRAIN"),
    ("EXTRATREES", "BASELINE_MEAN_TRAIN"),
    ("EXTRATREES", "BASELINE_PERSIST"),
]


def run_direction_dm_tests(df_dir: pd.DataFrame) -> list[dict]:
    """DM tests for direction (AUC -> loss = -AUC)."""
    results = []
    for our_model, baseline in DIRECTION_COMPARISONS:
        for sym in SYMBOLS:
            our_aucs, base_aucs = get_aligned_metric(df_dir, sym, our_model, baseline, "auc")
            if len(our_aucs) < 4 or len(base_aucs) < 4:
                continue
            # Loss = -AUC (higher AUC means lower loss)
            our_loss = -our_aucs
            base_loss = -base_aucs
            # Test: our model < baseline in loss (i.e. higher AUC)
            res = dm_test(our_loss, base_loss, horizon=FOLD_H, alternative="less")
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
                "conclusion": res["conclusion"].replace("Model A", our_model).replace("Model B", baseline),
            })
    return results


def run_volatility_dm_tests(df_vol: pd.DataFrame) -> list[dict]:
    """DM tests for volatility (RMSE -> loss = RMSE)."""
    results = []
    for our_model, baseline in VOLATILITY_COMPARISONS:
        for sym in SYMBOLS:
            our_rmse, base_rmse = get_aligned_metric(df_vol, sym, our_model, baseline, "rmse")
            if len(our_rmse) < 4 or len(base_rmse) < 4:
                continue
            # Test: our model < baseline in loss (i.e. lower RMSE)
            res = dm_test(our_rmse, base_rmse, horizon=FOLD_H, alternative="less")
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
                "conclusion": res["conclusion"].replace("Model A", our_model).replace("Model B", baseline),
            })
    return results


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print(" Diebold-Mariano test (with HLN correction)")
    print("=" * 70)
    print(f" Fold-level series: h = {FOLD_H} (consecutive test blocks overlap by half)")
    print(" H0: models forecast equally well")
    print(" H1: the proposed model forecasts BETTER (one-sided)")
    print("=" * 70)

    df_dir = load_test_metrics("direction")
    df_vol = load_test_metrics("volatility")

    print(f"\n[DATA] Direction CSV: {len(df_dir)} test-split rows")
    print(f"[DATA] Direction models: {sorted(df_dir['model'].unique())}")
    print(f"\n[DATA] Volatility CSV: {len(df_vol)} test-split rows")
    print(f"[DATA] Volatility models: {sorted(df_vol['model'].unique())}")

    all_results = []
    print("\n" + "=" * 70)
    print(" DM tests: DIRECTION (AUC)")
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
    print(" DM tests: VOLATILITY (RMSE)")
    print("=" * 70)
    vol_results = run_volatility_dm_tests(df_vol)
    all_results.extend(vol_results)
    for r in vol_results:
        print(
            f"  {r['model_A']:>15} vs {r['model_B']:<18} | {r['symbol']:<6} | "
            f"T={r['T']:>2} | mean(A)={r['mean_A']:.6f} mean(B)={r['mean_B']:.6f} | "
            f"DM_HLN={r['DM_HLN_stat']:>6.3f} p={r['p_value']:.4f} {r['significance']}"
        )

    # Save
    if all_results:
        out_path = ARTIFACTS_DIR / "dm_test_results.csv"
        pd.DataFrame(all_results).to_csv(out_path, index=False)
        print(f"\n[ARTIFACT] DM test results saved: {out_path}")

        # Significance summary
        df_res = pd.DataFrame(all_results)
        sig_count = sum(df_res["p_value"] < 0.05)
        print(f"\n[SUMMARY] Statistically significant comparisons (p < 0.05): {sig_count} / {len(df_res)}")
        print(f"[SUMMARY] Highly significant (p < 0.01): {sum(df_res['p_value'] < 0.01)} / {len(df_res)}")


if __name__ == "__main__":
    main()
