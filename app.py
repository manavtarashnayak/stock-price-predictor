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

# Page configuration - Institutional Financial Theme
st.set_page_config(
    page_title="Stock Movement Predictor | Quantitative Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for institutional, professional UI without gradients or pill buttons
st.markdown("""
    <style>
    /* Base theme adjustments */
    .stApp {
        background-color: #0e1117;
        color: #d1d4dc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px;
    }
    
    /* Card Containers */
    .metric-card {
        background-color: #1e222d;
        border: 1px solid #2a2e39;
        border-radius: 4px;
        padding: 20px;
        margin-bottom: 15px;
    }
    
    .signal-box-up {
        background-color: #089981;
        color: #ffffff;
        padding: 16px 20px;
        border-radius: 4px;
        font-size: 18px;
        font-weight: 600;
        text-align: center;
    }
    
    .signal-box-down {
        background-color: #f23645;
        color: #ffffff;
        padding: 16px 20px;
        border-radius: 4px;
        font-size: 18px;
        font-weight: 600;
        text-align: center;
    }
    
    /* Progress bar override */
    .stProgress > div > div > div > div {
        background-color: #2962ff;
        border-radius: 0px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #131722;
        border-right: 1px solid #2a2e39;
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 1px solid #2a2e39;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title Section
st.title("Stock Price Movement Predictor")
st.markdown("Quantitative directional probability model using historical market data and technical indicators.")
st.markdown("---")

# Sidebar - Selection & Configuration
st.sidebar.markdown("### Market Selection")

ticker_list = {
    "Apple Inc. (AAPL)": "AAPL",
    "Tesla Inc. (TSLA)": "TSLA",
    "NVIDIA Corp. (NVDA)": "NVDA",
    "Microsoft Corp. (MSFT)": "MSFT",
    "Amazon.com Inc. (AMZN)": "AMZN",
    "Bitcoin USD (BTC-USD)": "BTC-USD",
    "S&P 500 ETF (SPY)": "SPY"
}

selected_ticker_name = st.sidebar.selectbox("Select Asset", list(ticker_list.keys()))
custom_ticker = st.sidebar.text_input("Custom Ticker Symbol", "").strip().upper()

if custom_ticker:
    ticker = custom_ticker
else:
    ticker = ticker_list[selected_ticker_name]

st.sidebar.markdown("---")
st.sidebar.markdown("### View Mode")
view_mode = st.sidebar.radio("Select Interface Mode", ["Standard View", "Technical Audit View"])

test_ratio = 0.2

@st.cache_data(ttl=3600)
def fetch_and_process_market_data(symbol):
    data = yf.download(symbol, start="2018-01-01")
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Technical Indicators
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(close=df['Close']).macd()
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    
    df = df.dropna()
    return df

with st.spinner(f"Retrieving market dataset for {ticker}..."):
    df = fetch_and_process_market_data(ticker)

if df is None or len(df) < 100:
    st.error(f"Unable to retrieve market dataset for ticker '{ticker}'. Please verify the symbol.")
    st.stop()

# Time-series Split
split_index = int(len(df) * (1 - test_ratio))
train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

features_raw = ['Open', 'High', 'Low', 'Close', 'Volume']
features_eng = features_raw + ['rsi', 'macd', 'sma_20']

# Fit scaler strictly on training window
scaler = StandardScaler().fit(train[features_eng])
X_train = scaler.transform(train[features_eng])
X_test = scaler.transform(test[features_eng])

# Model Training
model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42).fit(X_train, train['target'])
test_predictions = model.predict(X_test)

# Evaluation
majority_baseline = train['target'].mode()[0]
majority_predictions = np.full(len(test), majority_baseline)
majority_accuracy = accuracy_score(test['target'], majority_predictions)
model_accuracy = accuracy_score(test['target'], test_predictions)

# Next-day Forecast
current_features = scaler.transform(df[features_eng].iloc[[-1]])
forecast_probability = model.predict_proba(current_features)[0]
forecast_direction = model.predict(current_features)[0]
last_close_price = float(df['Close'].iloc[-1])

# ==============================================================================
# STANDARD USER INTERFACE (Beginner & General Users)
# ==============================================================================

if view_mode == "Standard View":
    st.markdown("### Next Trading Session Forecast")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Asset Symbol", ticker)
    with col2:
        st.metric("Latest Closing Price", f"${last_close_price:.2f}")
    with col3:
        direction_text = "UPWARD" if forecast_direction == 1 else "DOWNWARD"
        st.metric("Directional Forecast", direction_text)
    with col4:
        model_prob_value = forecast_probability[1] if forecast_direction == 1 else forecast_probability[0]
        st.metric("Model Probability", f"{model_prob_value * 100:.1f}%")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_signal, col_explanation = st.columns([1, 2])
    
    with col_signal:
        if forecast_direction == 1:
            st.markdown('<div class="signal-box-up">FORECAST: UPWARD MOVEMENT</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="signal-box-down">FORECAST: DOWNWARD MOVEMENT</div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Forecast Probability Confidence:** {model_prob_value * 100:.1f}%")
        st.progress(float(model_prob_value))
        
    with col_explanation:
        st.markdown("#### How to Interpret This Result")
        st.write(f"The quantitative model analyzes recent price patterns and technical momentum indicators for **{ticker}**.")
        st.write(f"For the upcoming market session, the model estimates a **{model_prob_value * 100:.1f}% likelihood** of **{'upward' if forecast_direction == 1 else 'downward'}** closing price movement relative to today's close of **${last_close_price:.2f}**.")
        st.write(f"On unseen historical evaluation data, this predictive model achieved a classification accuracy of **{model_accuracy:.1%}**, compared to the uninformative baseline accuracy of **{majority_accuracy:.1%}**.")

    st.markdown("---")

    # Price & Moving Average Chart
    st.markdown(f"### Historical Price Chart ({ticker})")
    
    chart_df = df.iloc[-252:]  # Past 1 year
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_df.index, 
        y=chart_df['Close'], 
        name='Closing Price', 
        line=dict(color='#2962ff', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=chart_df.index, 
        y=chart_df['sma_20'], 
        name='20-Day Moving Average', 
        line=dict(color='#ff9800', width=1.5)
    ))
    
    fig.update_layout(
        height=420,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1e222d",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=True, gridcolor="#2a2e39"),
        yaxis=dict(showgrid=True, gridcolor="#2a2e39"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TECHNICAL AUDIT VIEW (For Advanced Users & Evaluators)
# ==============================================================================

else:
    st.markdown("### Technical Audit & Methodology")
    
    tab_overview, tab_charts, tab_metrics = st.tabs(["Overview & Verification", "Technical Indicators", "Performance Metrics"])
    
    with tab_overview:
        st.markdown("#### Methodology Summary")
        st.write("This application implements a leakage-free machine learning pipeline for binary stock movement classification.")
        st.markdown("""
        - **Target Variable Definition:** Binary indicator $y_t = 1$ if $\\text{Close}_{t+1} > \\text{Close}_t$, else $0$.
        - **Data Splitting:** Temporal chronological split (80% training window, 20% unseen test window).
        - **Normalization:** `StandardScaler` fitted strictly on training data to prevent out-of-fold leakage.
        - **Model Architecture:** Random Forest Classifier (200 estimators, max depth 5).
        """)

    with tab_charts:
        st.markdown("#### Detailed Indicator Analysis")
        
        fig_multi = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.06,
            subplot_titles=('Closing Price and 20-Day Simple Moving Average', 'Relative Strength Index (RSI 14)', 'MACD Oscillator')
        )
        
        fig_multi.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='#2962ff', width=1.5)), row=1, col=1)
        fig_multi.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', line=dict(color='#ff9800', width=1.5)), row=1, col=1)
        
        fig_multi.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='#089981', width=1.5)), row=2, col=1)
        fig_multi.add_hline(y=70, line_color="#f23645", row=2, col=1)
        fig_multi.add_hline(y=30, line_color="#089981", row=2, col=1)
        
        fig_multi.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='#ab47bc', width=1.5)), row=3, col=1)
        
        fig_multi.update_layout(
            height=650, 
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1e222d",
            xaxis3=dict(showgrid=True, gridcolor="#2a2e39"),
            yaxis=dict(showgrid=True, gridcolor="#2a2e39"),
            yaxis2=dict(showgrid=True, gridcolor="#2a2e39"),
            yaxis3=dict(showgrid=True, gridcolor="#2a2e39")
        )
        st.plotly_chart(fig_multi, use_container_width=True)

    with tab_metrics:
        st.markdown("#### Statistical Model Evaluation")
        
        prec = precision_score(test['target'], test_predictions, zero_division=0)
        rec = recall_score(test['target'], test_predictions, zero_division=0)
        f1 = f1_score(test['target'], test_predictions, zero_division=0)
        
        metrics_table = pd.DataFrame([
            {"Model": "Random Forest (Engineered Features)", "Accuracy": f"{model_accuracy:.2%}", "Precision": f"{prec:.2%}", "Recall": f"{rec:.2%}", "F1 Score": f"{f1:.2%}"},
            {"Model": "Majority Class Baseline", "Accuracy": f"{majority_accuracy:.2%}", "Precision": "N/A", "Recall": "N/A", "F1 Score": "N/A"}
        ])
        st.table(metrics_table)
        
        n_test = len(test)
        n_correct = int(round(model_accuracy * n_test))
        b_test = binomtest(n_correct, n_test, p=majority_accuracy, alternative='greater')
        
        st.markdown("##### Binomial Hypothesis Test")
        st.write(f"- Test Set Observations: {n_test}")
        st.write(f"- Correct Predictions: {n_correct} ({model_accuracy:.2%})")
        st.write(f"- Null Hypothesis Rate (Majority Baseline): {majority_accuracy:.2%}")
        st.write(f"- One-sided p-value: {b_test.pvalue:.4f}")
        
        if b_test.pvalue < 0.05:
            st.success("Result: The predictive model demonstrates statistically significant improvement over baseline (p < 0.05).")
        else:
            st.warning("Result: Performance difference relative to baseline is not statistically significant (p >= 0.05).")

st.markdown("---")
st.caption("Disclaimer: This tool is strictly for academic and demonstration purposes. It does not constitute financial advice or investment recommendations.")
