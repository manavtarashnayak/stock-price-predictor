import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import binomtest
import ta

st.set_page_config(
    page_title="Easy Stock Price Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header with Beginner Guide
st.title("🤖 Beginner-Friendly AI Stock Predictor")
st.caption("Simple, data-backed next-day stock price predictions using Machine Learning.")

# Sidebar Configuration
st.sidebar.header("1️⃣ Select Stock")

# Quick Buttons or Selectbox
preset_stocks = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "NVIDIA (NVDA)": "NVDA",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "S&P 500 ETF (SPY)": "SPY"
}

selected_name = st.sidebar.selectbox("Choose a popular stock:", list(preset_stocks.keys()))
custom_input = st.sidebar.text_input("...or type any ticker symbol:", "").strip().upper()

if custom_input:
    ticker = custom_input
else:
    ticker = preset_stocks[selected_name]

st.sidebar.header("2️⃣ Advanced Options (Optional)")
user_mode = st.sidebar.radio("Display Mode:", ["🌱 Beginner Mode (Simplified)", "🔬 Expert Mode (Detailed Technicals)"])

test_ratio = 0.2

@st.cache_data(ttl=3600)
def load_and_prep_data(ticker_symbol):
    df = yf.download(ticker_symbol, start="2018-01-01")
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Technical Indicators
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(close=df['Close']).macd()
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    
    df = df.dropna()
    return df

with st.spinner(f"Analyzing {ticker} historical market data..."):
    df = load_and_prep_data(ticker)

if df is None or len(df) < 100:
    st.error(f"⚠️ Could not load data for **'{ticker}'**. Please double-check the ticker symbol (e.g. AAPL, TSLA, MSFT).")
    st.stop()

# Chronological Train / Test Split
split_idx = int(len(df) * (1 - test_ratio))
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

features_raw = ['Open', 'High', 'Low', 'Close', 'Volume']
features_eng = features_raw + ['rsi', 'macd', 'sma_20']

# Scaling
scaler_eng = StandardScaler().fit(train[features_eng])
X_train_eng = scaler_eng.transform(train[features_eng])
X_test_eng = scaler_eng.transform(test[features_eng])

# Model Training (Random Forest)
model_rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42).fit(X_train_eng, train['target'])
pred_rf = model_rf.predict(X_test_eng)

# Baselines
majority_class = train['target'].mode()[0]
majority_pred = np.full(len(test), majority_class)
majority_acc = accuracy_score(test['target'], majority_pred)
model_acc = accuracy_score(test['target'], pred_rf)

# Next-Day Forecast
latest_features = scaler_eng.transform(df[features_eng].iloc[[-1]])
latest_prob = model_rf.predict_proba(latest_features)[0]
latest_pred = model_rf.predict(latest_features)[0]
latest_close = float(df['Close'].iloc[-1])

# ==============================================================================
# BEGINNER VIEW vs EXPERT VIEW
# ==============================================================================

if user_mode == "🌱 Beginner Mode (Simplified)":

    # Big Banner Signal
    st.markdown("### 🎯 Tomorrow's Price Forecast")
    
    col_card, col_explain = st.columns([1.2, 2])
    
    with col_card:
        if latest_pred == 1:
            st.success(f"### 🟢 BULLISH / UP\n**Price likely to go higher tomorrow.**")
            confidence = latest_prob[1] * 100
        else:
            st.error(f"### 🔴 BEARISH / DOWN\n**Price likely to go lower tomorrow.**")
            confidence = latest_prob[0] * 100
            
        st.metric(label="AI Model Confidence", value=f"{confidence:.1f}%")
        st.progress(confidence / 100.0)

    with col_explain:
        st.info("💡 **How to read this prediction:**")
        st.write(f"- Current Price of **{ticker}**: **${latest_close:.2f}**")
        st.write(f"- Based on past price patterns, our AI model estimates a **{confidence:.1f}% probability** that **{ticker}** will close **{'higher' if latest_pred == 1 else 'lower'}** on the next trading session.")
        st.write(f"- **Historical Accuracy:** On unseen past data, this model was correct **{model_acc:.1%}** of the time (compared to simple guessing at **{majority_acc:.1%}**).")

    st.markdown("---")

    # Simplified Price Chart
    st.markdown(f"### 📉 {ticker} Price History (Last 1 Year)")
    one_year_df = df.iloc[-252:]
    
    fig_simple = go.Figure()
    fig_simple.add_trace(go.Scatter(x=one_year_df.index, y=one_year_df['Close'], name='Stock Price', line=dict(color='#00F0FF', width=2.5)))
    fig_simple.add_trace(go.Scatter(x=one_year_df.index, y=one_year_df['sma_20'], name='20-Day Trend (Average)', line=dict(color='#FF007F', dash='dash')))
    fig_simple.update_layout(height=400, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_simple, use_container_width=True)

    with st.expander("❓ What is a 20-Day Trend Line?"):
        st.write("The dashed line shows the average stock price over the last 20 days. When the stock price is above this line, the stock is generally in an uptrend.")

else:
    # EXPERT MODE (Detailed)
    st.subheader(f"📊 Detailed Technical Analysis & Model Audit ({ticker})")
    
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    col_e1.metric("Current Price", f"${latest_close:.2f}")
    col_e2.metric("Predicted Signal", "UP (1)" if latest_pred == 1 else "DOWN (0)")
    col_e3.metric("Up Probability", f"{latest_prob[1]*100:.1f}%")
    col_e4.metric("Test Set Accuracy", f"{model_acc:.2%}")
    
    st.markdown("---")

    tab1, tab2 = st.tabs(["📉 Indicator Charts", "🔬 Leak-Free Benchmark Audit"])
    
    with tab1:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=(f'{ticker} Closing Price & SMA 20', 'RSI (14)', 'MACD'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='#00F0FF')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', line=dict(color='#FF007F', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='#FFD700')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='#00FF66')), row=3, col=1)
        fig.update_layout(height=600, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        b_test = binomtest(int(round(model_acc * len(test))), len(test), p=majority_acc, alternative='greater')
        st.write(f"- **Random Forest Accuracy:** {model_acc:.2%}")
        st.write(f"- **Majority Class Baseline:** {majority_acc:.2%}")
        st.write(f"- **Statistical p-value:** {b_test.pvalue:.4f}")
        if b_test.pvalue < 0.05:
            st.success("✅ Statistically significant improvement over baseline (p < 0.05).")
        else:
            st.warning("⚠️ Improvement over baseline is not statistically significant (p >= 0.05).")

st.markdown("---")
st.caption("⚠️ **Disclaimer:** For educational & demonstration purposes only. Not financial advice.")
