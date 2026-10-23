"""Risk management module.

vol_pred is the forecast realised volatility sigma_{t,5}: the standard deviation
of the next five DAILY log-returns, i.e. a daily-scale quantity. All five-day
quantities are therefore scaled by sqrt(5), as in Equations 25, 27 and 28 of
the paper:

- sigma_5d            = vol_pred * sqrt(5)
- expected_return_5d  = (2 * p_up - 1) * sigma_5d
- VaR (5 days)        = z * sigma_5d, with z = 1.645 / 1.96 / 2.33 for
                        Buy / Consider / Do not buy
- Sharpe (annualised) = (expected_return_5d / sigma_5d) * sqrt(252 / 5)
"""

from dataclasses import dataclass
from math import sqrt


@dataclass
class RiskMetrics:
    recommendation_ru: str
    recommendation_en: str
    confidence_ru: str
    confidence_en: str
    risk_label_ru: str
    risk_label_en: str
    position_size: str
    var_5d_approx: float
    no_trade_zone: bool
    expected_return_5d: float
    sharpe_approx: float
    risk_summary_ru: str
    risk_summary_en: str


# Уровни уверенности
_CONF_RU = {"high": "высокий", "medium": "средний", "low": "низкий"}
_RISK_RU = {"low": "низкий", "medium": "средний", "high": "высокий"}


class RiskManager:
    def __init__(self):
        self.buy_threshold = 0.60       # минимум для "Покупать"
        self.consider_threshold = 0.55  # минимум для "Задуматься"
        self.low_vol_cutoff = 0.022     # «низкая» волатильность для 5д

    def evaluate(self, p_up: float, vol_pred: float) -> RiskMetrics:
        p_up = float(max(0.0, min(1.0, p_up)))
        vol_pred = float(max(0.0, vol_pred))

        # ---- Решение ----
        if p_up >= self.buy_threshold and vol_pred <= self.low_vol_cutoff:
            rec_ru, rec_en = "Покупать", "Buy"
            conf_key = "high" if p_up >= 0.65 else "medium"
            risk_key = "low"
            position = "10% капитала"
            no_trade = False
            var_mult = 1.645
        elif p_up >= self.consider_threshold:
            rec_ru, rec_en = "Задуматься о покупке", "Consider Buying"
            conf_key = "medium" if p_up >= 0.58 else "low"
            risk_key = "medium"
            position = "5% капитала"
            no_trade = False
            var_mult = 1.96
        else:
            rec_ru, rec_en = "Не покупать", "Do Not Buy"
            conf_key = "low"
            risk_key = "high"
            position = "0% (избегать)"
            no_trade = True
            var_mult = 2.33

        # ---- Metrics (vol_pred is daily-scale; five-day quantities scale with sqrt(5)) ----
        sigma_5d = vol_pred * sqrt(5.0)
        expected_return_5d = (2.0 * p_up - 1.0) * sigma_5d
        var_5d = round(var_mult * sigma_5d, 4)
        if sigma_5d > 1e-9:
            sharpe = float(round((expected_return_5d / sigma_5d) * sqrt(252.0 / 5.0), 2))
        else:
            sharpe = 0.0

        level = {1.645: "95%", 1.96: "97.5%", 2.33: "99%"}[var_mult]      # confidence level of z
        risk_summary_ru = f"Риск: {_RISK_RU[risk_key]} • Позиция: {position} • VaR(5д, {level}): {var_5d:.2%}"
        risk_summary_en = (
            f"Risk: {risk_key} • Position: {position.replace('капитала', 'of capital').replace('избегать','avoid')} "
            f"• VaR(5d, {level}): {var_5d:.2%}"
        )

        return RiskMetrics(
            recommendation_ru=rec_ru,
            recommendation_en=rec_en,
            confidence_ru=_CONF_RU[conf_key],
            confidence_en=conf_key,
            risk_label_ru=_RISK_RU[risk_key],
            risk_label_en=risk_key,
            position_size=position,
            var_5d_approx=var_5d,
            no_trade_zone=no_trade,
            expected_return_5d=round(expected_return_5d, 4),
            sharpe_approx=sharpe,
            risk_summary_ru=risk_summary_ru,
            risk_summary_en=risk_summary_en,
        )

    def add_to_prediction(self, pred: dict) -> dict:
        m = self.evaluate(pred["p_up"], pred["vol_pred"])
        pred.update({
            "recommendation_ru": m.recommendation_ru,
            "recommendation_en": m.recommendation_en,
            "confidence": m.confidence_ru,         # обратная совместимость
            "confidence_ru": m.confidence_ru,
            "confidence_en": m.confidence_en,
            "risk_label_ru": m.risk_label_ru,
            "risk_label_en": m.risk_label_en,
            "position_size": m.position_size,
            "var_5d_approx": m.var_5d_approx,
            "no_trade_zone": m.no_trade_zone,
            "expected_return_5d": m.expected_return_5d,
            "sharpe_approx": m.sharpe_approx,
            "risk_summary_ru": m.risk_summary_ru,
            "risk_summary_en": m.risk_summary_en,
        })
        return pred

    def position_size_pct(self, p_up: float, vol_pred: float) -> float:
        """Возвращает рекомендуемый размер позиции в долях капитала (для бэктеста).

        - 0.10 если "Покупать"
        - 0.05 если "Задуматься"
        - 0.0 если "Не покупать"
        """
        m = self.evaluate(p_up, vol_pred)
        if m.no_trade_zone:
            return 0.0
        if m.recommendation_en == "Buy":
            return 0.10
        return 0.05
