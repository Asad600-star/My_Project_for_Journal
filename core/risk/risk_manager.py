"""Risk management module.

Корректные формулы:
- expected_return_5d = p_up * upside - (1 - p_up) * downside,
  где upside ≈ vol_pred (1 sigma), downside ≈ vol_pred.
- VaR (95%): z * sigma_5d = 1.645 * vol_pred (нормальное приближение).
- Sharpe (приведённый к году): (mu_daily / sigma_daily) * sqrt(252),
  где mu_daily = expected_return_5d / 5, sigma_daily = vol_pred / sqrt(5).
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
            position = "8-12% капитала"
            no_trade = False
            var_mult = 1.645
        elif p_up >= self.consider_threshold:
            rec_ru, rec_en = "Задуматься о покупке", "Consider Buying"
            conf_key = "medium" if p_up >= 0.58 else "low"
            risk_key = "medium"
            position = "4-6% капитала"
            no_trade = False
            var_mult = 1.96
        else:
            rec_ru, rec_en = "Не покупать", "Do Not Buy"
            conf_key = "low"
            risk_key = "high"
            position = "0% (избегать)"
            no_trade = True
            var_mult = 2.33

        # ---- Метрики ----
        # Ожидаемый возврат за 5 дней (грубая оценка через 1-sigma)
        expected_return_5d = p_up * vol_pred - (1.0 - p_up) * vol_pred  # = (2*p_up - 1) * vol_pred

        # VaR: z * sigma на горизонте 5 дней
        var_5d = round(var_mult * vol_pred, 4)

        # Sharpe (annualized)
        if vol_pred > 1e-9:
            mu_daily = expected_return_5d / 5.0
            sigma_daily = vol_pred / sqrt(5.0)
            sharpe = float(round((mu_daily / sigma_daily) * sqrt(252.0), 2))
        else:
            sharpe = 0.0

        risk_summary_ru = f"Риск: {_RISK_RU[risk_key]} • Позиция: {position} • VaR(5д, 95%): {var_5d:.2%}"
        risk_summary_en = (
            f"Risk: {risk_key} • Position: {position.replace('капитала', 'of capital').replace('избегать','avoid')} "
            f"• VaR(5d, 95%): {var_5d:.2%}"
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
