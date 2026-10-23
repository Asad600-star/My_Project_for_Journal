"""The risk module must implement Equations 25-28 and the banded sizing rule of the paper."""
from __future__ import annotations

from math import sqrt

import pytest

from core.risk.risk_manager import RiskManager

rm = RiskManager()


@pytest.mark.parametrize("p, vol, rec, z, size", [
    (0.642, 0.0148, "Buy", 1.645, 0.10),
    (0.640, 0.0300, "Consider Buying", 1.96, 0.05),     # Buy probability but volatility above the cut-off
    (0.574, 0.0218, "Consider Buying", 1.96, 0.05),
    (0.549, 0.0100, "Do Not Buy", 2.33, 0.0),
    (0.300, 0.0100, "Do Not Buy", 2.33, 0.0),
])
def test_recommendation_var_and_size(p, vol, rec, z, size):
    m = rm.evaluate(p, vol)
    assert m.recommendation_en == rec
    assert m.var_5d_approx == pytest.approx(z * vol * sqrt(5), abs=1e-4)
    assert rm.position_size_pct(p, vol) == size
    assert m.no_trade_zone == (size == 0.0)


def test_expected_return_and_sharpe_scale_with_sqrt5():
    p, vol = 0.62, 0.012
    m = rm.evaluate(p, vol)
    assert m.expected_return_5d == pytest.approx((2 * p - 1) * vol * sqrt(5), abs=1e-4)
    assert m.sharpe_approx == pytest.approx((2 * p - 1) * sqrt(252 / 5), abs=0.01)


def test_inputs_are_clipped():
    m = rm.evaluate(1.7, -0.01)
    assert m.var_5d_approx == 0.0 and m.sharpe_approx == 0.0


@pytest.mark.parametrize("p, vol, level", [(0.65, 0.01, "95%"), (0.57, 0.01, "97.5%"), (0.40, 0.01, "99%")])
def test_risk_summary_states_the_confidence_level_of_its_multiplier(p, vol, level):
    m = rm.evaluate(p, vol)
    assert f"VaR(5d, {level})" in m.risk_summary_en and f"VaR(5д, {level})" in m.risk_summary_ru
