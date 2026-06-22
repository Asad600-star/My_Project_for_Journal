import os
import sys
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from services.predict import get_prediction
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg://stock:stockpass@localhost:5432/stockdb")

st.set_page_config(page_title="Stock Forecast", page_icon="📈", layout="wide")

st.title("📈 Прогноз направления и волатильности акций")
st.markdown("**Гибридная ML-модель** • горизонт 5 / 10 / 20 дней • Реальное время")

lang = st.sidebar.radio("Язык / Language", ["🇷🇺 Русский", "🇬🇧 English"], horizontal=True)
is_ru = lang.startswith("🇷🇺")

symbols = {
    "AAPL": "Apple Inc. (AAPL)",
    "TSLA": "Tesla Inc. (TSLA)",
    "MSFT": "Microsoft Corp. (MSFT)",
    "GLD": "SPDR Gold Trust (GLD)",
    "^GSPC": "S&P 500 (^GSPC)",
    "^IXIC": "Nasdaq Composite (^IXIC)",
    "^DJI": "Dow Jones (^DJI)",
    "^RUT": "Russell 2000 (^RUT)",
}

symbol = st.sidebar.selectbox(
    "Выберите инструмент" if is_ru else "Select instrument",
    options=list(symbols.keys()),
    format_func=lambda x: symbols[x]
)

horizon = st.sidebar.selectbox(
    "Горизонт прогноза" if is_ru else "Forecast horizon",
    options=[5, 10, 20],
    format_func=lambda k: (f"{k} дней вперёд" if is_ru else f"{k} days ahead"),
)

if st.sidebar.button("🔄 Обновить данные и прогноз" if is_ru else "🔄 Refresh", width='stretch'):
    with st.spinner("Обновление..." if is_ru else "Refreshing..."):
        result = get_prediction(symbol, horizon=horizon, refresh=True)
    st.success("✅ Обновлено!" if is_ru else "✅ Done!")
else:
    result = get_prediction(symbol, horizon=horizon, refresh=False)

st.subheader(f"{symbols[symbol]} • {result['asof_date']}")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Рекомендация" if is_ru else "Recommendation",
              result["recommendation_ru"] if is_ru else result["recommendation_en"])
with c2:
    st.metric("Уверенность" if is_ru else "Confidence",
              (result["confidence_ru"] if is_ru else result["confidence_en"]).capitalize())
with c3:
    st.metric("Риск" if is_ru else "Risk",
              result["risk_label_ru"] if is_ru else result["risk_label_en"])

if is_ru:
    st.info(f"**Вероятность роста ({horizon} дней):** {result['p_up']:.1%} | **Ожидаемая волатильность ({horizon} дней):** {result['vol_pred']:.2%}")
else:
    st.info(f"**Probability of rise ({horizon}d):** {result['p_up']:.1%} | **Expected {horizon}d volatility:** {result['vol_pred']:.2%}")

st.subheader("🛡️ Risk Management")
st.success(result["risk_summary_ru"] if is_ru else result["risk_summary_en"])

# ==================== ГРАФИК ЦЕНЫ ====================
st.subheader("📈 График цены + прогнозный коридор" if is_ru else "📈 Price + forecast band")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
df_price = pd.read_sql_query(
    text("SELECT date, close FROM market_ohlcv WHERE symbol = :sym ORDER BY date ASC"),
    engine, params={"sym": symbol}
)
df_price = df_price.tail(60).reset_index(drop=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_price["date"], y=df_price["close"],
    name="Историческая цена" if is_ru else "Historical price",
    line=dict(color="#22c55e")
))

last_close = float(df_price["close"].iloc[-1])
dates_future = pd.date_range(start=df_price["date"].iloc[-1], periods=horizon + 1, freq="B")[1:]
# Корридор расширяется как sqrt(t) (классическая стохастика) на выбранный горизонт
import numpy as np
sigma = float(result["vol_pred"])
upper = [last_close * (1 + sigma * np.sqrt(i)) for i in range(1, horizon + 1)]
lower = [last_close * (1 - sigma * np.sqrt(i)) for i in range(1, horizon + 1)]

fig.add_trace(go.Scatter(x=dates_future, y=upper, mode="lines",
                         line=dict(color="rgba(34,197,94,0.4)"),
                         name="Верхний коридор" if is_ru else "Upper band"))
fig.add_trace(go.Scatter(x=dates_future, y=lower, mode="lines",
                         line=dict(color="rgba(234,179,8,0.4)"),
                         name="Нижний коридор" if is_ru else "Lower band",
                         fill="tonexty"))

fig.update_layout(height=500, template="plotly_dark",
                  title=f"{symbol} — {'Последние 60 дней + прогноз' if is_ru else 'Last 60 days + forecast'}")
st.plotly_chart(fig, width='stretch')

# ==================== SHAP ====================
st.subheader("🔍 Почему модель решила именно так? (SHAP)" if is_ru else "🔍 Why did the model decide so? (SHAP)")

shap_file = ROOT / "artifacts" / f"shap_{symbol}_direction.json"
if shap_file.exists():
    with open(shap_file, encoding="utf-8") as f:
        data = json.load(f)

    shap_values = data["shap_values"]
    if isinstance(shap_values, list) and len(shap_values) > 0 and isinstance(shap_values[0], list):
        shap_values = shap_values[0]

    top = sorted(zip(data["feature_names"], shap_values), key=lambda x: abs(x[1]), reverse=True)[:12]
    names = [name for name, _ in top]
    values = [val for _, val in top]

    colors = ["#22c55e" if v > 0 else "#ef4444" for v in values]
    fig_shap = go.Figure(go.Bar(
        y=names[::-1],
        x=values[::-1],
        orientation='h',
        marker_color=colors[::-1],
        text=[f"{v:+.4f}" for v in values[::-1]],
        textposition="auto"
    ))
    fig_shap.update_layout(
        height=500,
        template="plotly_dark",
        title="Топ-12 факторов, которые повлияли на решение модели" if is_ru else "Top-12 factors driving the prediction",
        xaxis_title="Вклад в вероятность роста" if is_ru else "Contribution to P(up)",
        yaxis_title=""
    )
    st.plotly_chart(fig_shap, width='stretch')

    st.write("**Топ факторов (по силе влияния):**" if is_ru else "**Top factors (by impact):**")
    table_data = {("Фактор" if is_ru else "Feature"): names,
                  ("Вклад" if is_ru else "Contribution"): [f"{v:+.4f}" for v in values]}
    st.dataframe(pd.DataFrame(table_data), width='stretch')

else:
    st.info("SHAP-график будет доступен после следующего полного обновления модели."
            if is_ru else "SHAP plot will be available after the next full model update.")

st.caption(f"{'Последнее обновление' if is_ru else 'Last refresh'}: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
