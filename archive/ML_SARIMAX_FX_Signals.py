# FILE: ml_fx_signals.py
# --------------------------------------------------
# SARIMAX-exogenous FX signal module for your ES strategy
# (RandomForest replaced, structure preserved)
# --------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.special import expit
import statsmodels.api as sm


# ========================================================
# STREAMLIT CONFIG + GLOBAL STYLES
# ========================================================
st.set_page_config(
    page_title="SARIMAX ES + ML Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAJOR_FX_PAIRS = [
    "EURJPY=X", "USDJPY=X", "EURUSD=X", "GBPUSD=X",
    "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
]


# ========================================================
# 1. CORE HELPERS (UNCHANGED)
# ========================================================
@st.cache_data
def load_fx(symbol, timeframe, start_date, end_date):
    interval = {
        "Daily": "1d",
        "Weekly": "1wk",
        "Monthly": "1mo",
    }[timeframe]

    df = yf.download(
        symbol,
        interval=interval,
        start=start_date,
        end=end_date,
        progress=False,
    )

    df = df[["Open", "High", "Low", "Close"]].dropna()
    return df.astype(float)


def exp_smooth(series, lam):
    es = np.zeros(len(series))
    es[0] = series.iloc[0]
    for t in range(1, len(series)):
        es[t] = lam * series.iloc[t] + (1 - lam) * es[t - 1]
    return es


def generate_signals(es_alpha, es_beta, index, x=0.0, exit_rule="cross", theta=0.0):
    d = es_beta - es_alpha
    raw = np.zeros(len(d))

    for t in range(1, len(d)):
        if (d[t - 1] < 0) and (d[t] > x):
            raw[t] = 1
        elif (d[t - 1] > 0) and (d[t] < -x):
            raw[t] = -1

        if exit_rule == "deceleration":
            slope = d[t] - d[t - 1]
            if slope < -theta:
                raw[t] = 0

    raw = pd.Series(raw, index=index)
    pos = raw.replace(0, np.nan).ffill().fillna(0)
    return raw, pos


def backtest(prices, raw_signals, positions, cost=0.0001, starting_capital=10000):
    ret = prices.pct_change().fillna(0)
    strat = positions.shift(1) * ret
    strat -= positions.diff().abs().fillna(0) * cost

    equity = (1 + strat).cumprod() * starting_capital
    drawdown = equity.cummax() - equity

    metrics = {
        "Total Return %": (equity.iloc[-1] / starting_capital - 1) * 100,
        "Sharpe Ratio": (strat.mean() * 252) / (strat.std() * np.sqrt(252) + 1e-9),
        "Max Drawdown": drawdown.max(),
        "Number of Trades (signals)": int((raw_signals != 0).sum()),
    }

    return strat, equity, metrics, drawdown


# ========================================================
# 2. FEATURE ENGINEERING (UNCHANGED)
# ========================================================
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def build_feature_matrix(df, alpha, beta):
    close, high, low = df["Close"], df["High"], df["Low"]
    out = pd.DataFrame(index=df.index)

    out["price"] = close
    out["ret_1"] = close.pct_change()
    out["ret_5"] = close.pct_change(5)
    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["vol_20"] = out["ret_1"].rolling(20).std()

    out["rsi_14"] = compute_rsi(close)
    out["ma_10"] = close.rolling(10).mean()
    out["ma_30"] = close.rolling(30).mean()
    out["ma_diff"] = out["ma_10"] - out["ma_30"]

    es_a = exp_smooth(close, alpha)
    es_b = exp_smooth(close, beta)
    out["es_diff"] = es_b - es_a
    out["es_slope"] = out["es_diff"].diff()

    out["z_price_20"] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-9)

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    out["future_ret"] = out["ret_1"].shift(-1)
    out["y"] = (out["future_ret"] > 0).astype(int)

    out = out.iloc[65:-1]
    return out


# ========================================================
# 3. EXOG FEATURE DIAGNOSTICS (NEW, LOCAL ADDITION)
# ========================================================
def get_exog_candidates(feat_df):
    return [c for c in feat_df.columns if c not in {"price", "future_ret", "y"}]


def exog_feature_diagnostics(feat_df, prices):
    future_ret = np.log(prices).diff().shift(-1)
    results = []

    for col in get_exog_candidates(feat_df):
        x = feat_df[col]
        y = future_ret

        df = pd.concat([x, y], axis=1).dropna()
        if len(df) < 100:
            continue

        corr = df.iloc[:, 0].corr(df.iloc[:, 1])

        X = sm.add_constant(df.iloc[:, 0])
        model = sm.OLS(df.iloc[:, 1], X).fit()
        tstat = model.tvalues.iloc[1]

        hit = np.mean(np.sign(df.iloc[:, 0]) == np.sign(df.iloc[:, 1]))

        results.append({
            "Feature": col,
            "Correlation": corr,
            "T-Stat": tstat,
            "Hit Rate": hit,
            "N": len(df),
        })

    return pd.DataFrame(results).sort_values(
        by="T-Stat", key=lambda s: s.abs(), ascending=False
    )


# ========================================================
# 4. SIDEBAR – CONTROLS (UNCHANGED)
# ========================================================
with st.sidebar:
    st.markdown("### 📈 SARIMAX Exogenous Model")

    selected_pair = st.selectbox("FX Pair", MAJOR_FX_PAIRS)
    timeframe = st.selectbox("Timeframe", ["Daily", "Weekly"])
    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.date_input("End Date", datetime.now())

    alpha = st.slider("α (slow)", 0.01, 0.5, 0.1, 0.01)
    beta = st.slider("β (fast)", alpha + 0.01, 0.9, 0.3, 0.01)

    train_frac = st.slider("Train Fraction", 0.4, 0.9, 0.7, 0.05)

    ml_long_cutoff = st.slider("Min P(up) LONG", 0.5, 0.8, 0.55, 0.01)
    ml_short_cutoff = st.slider("Min P(down) SHORT", 0.5, 0.8, 0.55, 0.01)

    starting_capital = st.number_input("Capital", 1000, 1_000_000, 10000)
    cost = st.number_input("Transaction Cost", value=0.0001, step=0.0001)

    run_button = st.button("🚀 Run SARIMAX Strategy")


# ========================================================
# 5. MAIN LOGIC (RF → SARIMAX REPLACEMENT)
# ========================================================
if not run_button:
    st.stop()

df_px = load_fx(selected_pair, timeframe, start_date, end_date)
prices = df_px["Close"]

feat_df = build_feature_matrix(df_px, alpha, beta)

# ---- Exogenous diagnostics (NEW, additive) ----
st.markdown("## 🧪 Exogenous Feature Diagnostics")
exog_diag = exog_feature_diagnostics(feat_df, prices)

st.dataframe(
    exog_diag.style.format({
        "Correlation": "{:.3f}",
        "T-Stat": "{:.2f}",
        "Hit Rate": "{:.2%}",
    }),
    use_container_width=True,
)

default_exog = exog_diag.loc[
    exog_diag["T-Stat"].abs() > 0.5, "Feature"
].head(5).tolist()

selected_exog = st.multiselect(
    "Select exogenous features for SARIMAX",
    exog_diag["Feature"].tolist(),
    default=default_exog,
)

if len(selected_exog) == 0:
    st.error("Please select at least one exogenous feature.")
    st.stop()

# ---- SARIMAX training (replaces RF block) ----
log_ret = np.log(prices).diff().dropna()
exog_full = feat_df[selected_exog].replace([np.inf, -np.inf], np.nan).dropna()

common_idx = log_ret.index.intersection(exog_full.index)
log_ret = log_ret.loc[common_idx]
exog_full = exog_full.loc[common_idx]

train_size = int(len(log_ret) * train_frac)

model = SARIMAX(
    log_ret.iloc[:train_size],
    exog=exog_full.iloc[:train_size],
    order=(1, 0, 1),
    enforce_stationarity=False,
    enforce_invertibility=False,
)

res = model.fit(disp=False)

forecast = res.get_forecast(
    steps=len(log_ret) - train_size,
    exog=exog_full.iloc[train_size:]
)

mu = forecast.predicted_mean
sigma = np.sqrt(forecast.var_pred_mean)
p_up = pd.Series(expit(mu / (sigma + 1e-6)), index=mu.index)

# ========================================================
# 6. HYBRID ES + SARIMAX STRATEGY (UNCHANGED)
# ========================================================
es_a = exp_smooth(prices, alpha)
es_b = exp_smooth(prices, beta)
raw_es, pos_es = generate_signals(es_a, es_b, prices.index)

p_up_aligned = p_up.reindex(prices.index).ffill()

raw_hybrid = raw_es.copy()
for i in range(len(raw_hybrid)):
    if raw_hybrid.iloc[i] == 1 and p_up_aligned.iloc[i] < ml_long_cutoff:
        raw_hybrid.iloc[i] = 0
    if raw_hybrid.iloc[i] == -1 and (1 - p_up_aligned.iloc[i]) < ml_short_cutoff:
        raw_hybrid.iloc[i] = 0

pos_hybrid = raw_hybrid.replace(0, np.nan).ffill().fillna(0)

_, equity_es, metrics_es, _ = backtest(prices, raw_es, pos_es, cost, starting_capital)
_, equity_h, metrics_h, _ = backtest(prices, raw_hybrid, pos_hybrid, cost, starting_capital)

# ========================================================
# 7. CHARTS (UNCHANGED)
# ========================================================
eq = pd.DataFrame({"ES": equity_es, "Hybrid": equity_h})
st.markdown("## 📉 Equity Curve")
st.plotly_chart(px.line(eq, template="plotly_dark"), use_container_width=True)

st.markdown("## 📊 P(up)")
st.plotly_chart(px.line(p_up_aligned, template="plotly_dark"), use_container_width=True)
