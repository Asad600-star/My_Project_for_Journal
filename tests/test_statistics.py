"""Conformal risk control, the Diebold-Mariano test and the deflated Sharpe ratio."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jobs.diebold_mariano_test import dm_test
from jobs.racc import B, crc_threshold


def _draw(rng, n):
    s = rng.random(n)
    # the loss is larger when the score is small, as for a useful abstention score
    loss = np.clip(rng.exponential(1.0 - 0.8 * s), 0, B)
    return s, loss


def test_crc_threshold_is_smallest_feasible():
    rng = np.random.default_rng(1)
    s, loss = _draw(rng, 2000)
    grid = np.quantile(s, np.linspace(0, 1, 101))
    for alpha in (0.1, 0.3, 0.5):
        t = crc_threshold(s, loss, alpha, grid)
        n = len(loss)
        bound = lambda u: (n * np.mean(loss * (s >= u)) + B) / (n + 1)
        assert bound(t) <= alpha
        earlier = grid[grid < t]
        assert all(bound(u) > alpha for u in earlier)


def test_crc_threshold_abstains_when_no_threshold_is_feasible():
    s = np.linspace(0.0, 0.5, 50)
    loss = np.full(50, B)
    assert crc_threshold(s, loss, 0.01, np.quantile(s, np.linspace(0, 1, 101))) == np.inf


def test_crc_threshold_matches_a_direct_search_with_ties():
    """The vectorised threshold equals a direct search over the fixed grid,
    also for a step-function score with many ties (as after PAV calibration)."""
    from jobs.racc import GRID
    rng = np.random.default_rng(4)
    s = np.round(rng.uniform(0, 0.5, 900), 2)
    loss = np.clip(rng.exponential(0.6, 900), 0, B)
    n = len(loss)
    for alpha in (0.02, 0.1, 0.25, 0.4):
        direct = next((u for u in GRID if (np.sum(loss * (s >= u)) + B) / (n + 1) <= alpha), np.inf)
        assert crc_threshold(s, loss, alpha) == direct


def test_crc_controls_expected_risk_under_exchangeability():
    """Proposition 1: E[L_test] <= alpha when calibration and test are i.i.d."""
    rng = np.random.default_rng(2)
    alpha, risks = 0.3, []
    for _ in range(300):
        s_ca, l_ca = _draw(rng, 300)
        s_te, l_te = _draw(rng, 300)
        t = crc_threshold(s_ca, l_ca, alpha, np.quantile(s_ca, np.linspace(0, 1, 101)))
        risks.append(np.mean(l_te * (s_te >= t)))
    assert np.mean(risks) <= alpha


def test_dm_test_detects_a_clear_difference_and_is_antisymmetric():
    rng = np.random.default_rng(3)
    a = rng.normal(1.0, 0.1, 40)
    b = a + 0.2 + rng.normal(0, 0.05, 40)
    ab, ba = dm_test(a, b), dm_test(b, a)
    assert ab["p_value"] < 0.01
    assert ab["DM"] == pytest.approx(-ba["DM"])
    assert ab["p_value"] == pytest.approx(ba["p_value"])


def test_deflated_sharpe_penalises_many_trials(tmp_path, monkeypatch):
    import jobs.robustness_checks as rc
    monkeypatch.setattr(rc, "A", tmp_path)
    rng = np.random.default_rng(4)
    few = {f"s{i}": rng.normal(0.0005, 0.01, 600) for i in range(2)}
    many = dict(few, **{f"x{i}": rng.normal(0.0, 0.01, 600) for i in range(30)})
    d_few, d_many = rc.deflated_sharpe(few).iloc[0], rc.deflated_sharpe(many).iloc[0]
    # more trials raise the Sharpe ratio expected from luck alone
    assert d_many["expected_max_sharpe_null_ann"] > d_few["expected_max_sharpe_null_ann"]
    assert 0.0 <= d_many["deflated_sharpe_prob"] <= 1.0
    assert (tmp_path / "robustness_dsr.csv").exists()


def test_online_threshold_bound_holds_for_an_adversarial_sequence():
    """Proposition 2: the average daily risk stays below alpha + ((M - t0)/gamma + 2 lag B)/T
    even when the losses switch abruptly between regimes (no exchangeability)."""
    from jobs.racc import online_thresholds
    rng = np.random.default_rng(5)
    T, per_day, gamma, lag, alpha, t0 = 400, 8, 0.05, 6, 0.3, 0.05
    day = np.repeat(np.arange(T), per_day)
    s = rng.random(T * per_day) * 0.5
    crisis = (day // 50) % 2 == 1                    # alternating calm and crisis blocks
    loss = np.where(crisis, np.clip(rng.exponential(2.0, T * per_day), 0, B),
                    np.clip(rng.exponential(0.2, T * per_day), 0, B))
    act, thr = online_thresholds(day, s, loss, alpha, t0, gamma, lag)
    daily = np.array([np.mean(loss[day == j] * act[day == j]) for j in range(T)])
    assert daily.mean() <= alpha + ((0.5 - t0) / gamma + 2 * lag * B) / T
    assert abs(daily.mean() - alpha) < 0.05          # it also tracks alpha closely
    assert thr.max() <= 0.5 + gamma * lag * B        # the threshold stays bounded


def test_online_threshold_uses_only_resolved_labels():
    from jobs.racc import online_thresholds
    rng = np.random.default_rng(6)
    day = np.repeat(np.arange(60), 4)
    s, loss = rng.random(240) * 0.5, rng.random(240)
    _, thr_a = online_thresholds(day, s, loss, 0.2, 0.1, 0.05, 6)
    loss_b = loss.copy()
    loss_b[day >= 40] = 3.0                          # change losses from day 40 onwards
    _, thr_b = online_thresholds(day, s, loss_b, 0.2, 0.1, 0.05, 6)
    assert np.allclose(thr_a[:46], thr_b[:46])        # thresholds up to day 45 are unaffected
    assert not np.allclose(thr_a[46:], thr_b[46:])


def test_qlike_is_zero_only_for_a_perfect_forecast_and_penalises_under_prediction():
    from jobs.volatility_models import qlike
    y = np.array([0.01, 0.02, 0.015])
    assert np.allclose(qlike(y, y), 0.0)
    # QLIKE is asymmetric: under-predicting the variance costs more than over-predicting it
    assert qlike(y, 0.8 * y).mean() > qlike(y, 1.2 * y).mean() > 0


def test_racc_score_does_not_use_the_calibration_labels():
    """Proposition 1 needs a score fixed before the calibration data are seen:
    changing the labels of the calibration block must not change its scores."""
    from jobs.racc import fit_racc
    rng = np.random.default_rng(7)

    def frame(n):
        X = rng.standard_normal((n, 4))
        r = 0.01 * X[:, 0] + 0.02 * rng.standard_normal(n)
        return pd.DataFrame({"f0": X[:, 0], "f1": X[:, 1], "f2": X[:, 2], "f3": X[:, 3],
                             "target_direction": (r >= 0).astype(int), "target_return_kd": r,
                             "target_vol_kd": 0.01 + 0.002 * np.abs(X[:, 1]),
                             "cls3": np.where(r > 0.01, 0, np.where(r < -0.01, 2, 1))})
    tr, tu, ca = frame(1500), frame(400), frame(400)
    feat = ["f0", "f1", "f2", "f3"]
    m = fit_racc(tr, tu, feat)
    s1, _, _ = m.score(ca)
    ca2 = ca.copy()
    ca2["target_direction"] = 1 - ca2["target_direction"]          # flip the calibration labels
    m2 = fit_racc(tr, tu, feat)
    s2, _, _ = m2.score(ca2)
    assert np.allclose(s1, s2) and m.lam == m2.lam
