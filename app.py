# pyrefly: ignore [missing-import]
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
from plotly.subplots import make_subplots
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

# Custom CSS for clean institutional UI
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #d1d4dc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px;
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
    
    .stProgress > div > div > div > div {
        background-color: #2962ff;
        border-radius: 0px;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #131722;
        border-right: 1px solid #2a2e39;
    }
    
    .stDataFrame {
        border: 1px solid #2a2e39;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Main Header
st.title("Stock Price Movement Predictor")
st.markdown("Quantitative directional probability model using historical market data and technical indicators.")
st.markdown("---")

# Sidebar Controls
st.sidebar.markdown("### Asset Selection")

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

# Sidebar - Graph Customization Controls
st.sidebar.markdown("---")
st.sidebar.markdown("### Graph Customization")

chart_type = st.sidebar.selectbox("Chart Style", ["Line Chart", "Candlestick Chart"])

timeframe_map = {
    "1 Month": 21,
    "3 Months": 63,
    "6 Months": 126,
    "1 Year": 252,
    "3 Years": 756,
    "5 Years": 1260,
    "All Available Data": None
}
selected_timeframe = st.sidebar.selectbox("History Horizon", list(timeframe_map.keys()), index=3)

st.sidebar.markdown("**Technical Overlays**")
show_sma20 = st.sidebar.checkbox("20-Day Simple Moving Average (SMA 20)", value=True)
show_ema50 = st.sidebar.checkbox("50-Day Exponential Moving Average (EMA 50)", value=False)
show_rsi = st.sidebar.checkbox("Relative Strength Index (RSI 14)", value=False)
show_macd = st.sidebar.checkbox("MACD Oscillator", value=False)
show_volume = st.sidebar.checkbox("Trading Volume Subplot", value=False)

st.sidebar.markdown("---")
show_technical_audit = st.sidebar.checkbox("Show Technical Audit & Methodology", value=False)

test_ratio = 0.2

@st.cache_data(ttl=3600)
def fetch_and_process_market_data(symbol):
    try:
        data = yf.download(symbol, start="2018-01-01", progress=False)
        if data is None or data.empty:
            return None
            
        if isinstance(data.columns, pd.MultiIndex):
            if symbol in data.columns.levels[1]:
                data = data.xs(symbol, axis=1, level=1)
            else:
                data.columns = data.columns.get_level_values(0)
                
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            return None
            
        df = data[required_cols].copy()
        
        # Ensure 1D Series for all price columns
        for col in required_cols:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]
                
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Technical Indicators
        df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        df['macd'] = ta.trend.MACD(close=df['Close']).macd()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        df = df.dropna()
        return df
    except Exception:
        return None

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

scaler = StandardScaler().fit(train[features_eng])
X_train = scaler.transform(train[features_eng])
X_test = scaler.transform(test[features_eng])

model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42).fit(X_train, train['target'])
test_predictions = model.predict(X_test)

majority_baseline = train['target'].mode()[0]
majority_predictions = np.full(len(test), majority_baseline)
majority_accuracy = accuracy_score(test['target'], majority_predictions)
model_accuracy = accuracy_score(test['target'], test_predictions)

current_features = scaler.transform(df[features_eng].iloc[[-1]])
forecast_probability = model.predict_proba(current_features)[0]
forecast_direction = model.predict(current_features)[0]
last_close_price = float(df['Close'].iloc[-1])

# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================

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
    st.markdown(f"**Forecast Confidence Level:** {model_prob_value * 100:.1f}%")
    st.progress(float(model_prob_value))
    
with col_explanation:
    st.markdown("#### How to Interpret This Result")
    st.write(f"The quantitative model analyzes historical price patterns and momentum indicators for **{ticker}**.")
    st.write(f"For the upcoming session, the model calculates a **{model_prob_value * 100:.1f}% likelihood** of **{'upward' if forecast_direction == 1 else 'downward'}** closing price movement relative to today's close of **${last_close_price:.2f}**.")
    st.write(f"On unseen historical evaluation data, this predictive model achieved a classification accuracy of **{model_accuracy:.1%}**, compared to the uninformative baseline rate of **{majority_accuracy:.1%}**.")

st.markdown("---")

# ==============================================================================
# CUSTOMIZABLE DYNAMIC CHART
# ==============================================================================

st.markdown(f"### Historical Price & Technical Analysis ({ticker})")

# Slice dataframe according to selected timeframe
max_rows = timeframe_map[selected_timeframe]
if max_rows is not None and max_rows < len(df):
    chart_df = df.iloc[-max_rows:].copy()
else:
    chart_df = df.copy()

# Determine number of subplots dynamically based on user selections
subplots = ["Price"]
if show_rsi:
    subplots.append("RSI")
if show_macd:
    subplots.append("MACD")
if show_volume:
    subplots.append("Volume")

num_rows = len(subplots)

row_heights = [0.5] + [0.5 / (num_rows - 1)] * (num_rows - 1) if num_rows > 1 else [1.0]

fig = make_subplots(
    rows=num_rows, 
    cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.04,
    row_heights=row_heights,
    subplot_titles=subplots
)

# Row 1: Price (Candlestick or Line)
if chart_type == "Candlestick Chart":
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df['Open'],
        high=chart_df['High'],
        low=chart_df['Low'],
        close=chart_df['Close'],
        name='Price (OHLC)',
        increasing_line_color='#089981',
        decreasing_line_color='#f23645'
    ), row=1, col=1)
else:
    fig.add_trace(go.Scatter(
        x=chart_df.index, 
        y=chart_df['Close'], 
        name='Closing Price', 
        line=dict(color='#2962ff', width=2)
    ), row=1, col=1)

# Overlays on Row 1
if show_sma20:
    fig.add_trace(go.Scatter(
        x=chart_df.index, 
        y=chart_df['sma_20'], 
        name='SMA 20', 
        line=dict(color='#ff9800', width=1.5)
    ), row=1, col=1)

if show_ema50:
    fig.add_trace(go.Scatter(
        x=chart_df.index, 
        y=chart_df['ema_50'], 
        name='EMA 50', 
        line=dict(color='#e91e63', width=1.5)
    ), row=1, col=1)

# Secondary Subplots
current_row = 2

if show_rsi:
    fig.add_trace(go.Scatter(
        x=chart_df.index, 
        y=chart_df['rsi'], 
        name='RSI 14', 
        line=dict(color='#089981', width=1.5)
    ), row=current_row, col=1)
    fig.add_hline(y=70, line_color="#f23645", row=current_row, col=1)
    fig.add_hline(y=30, line_color="#089981", row=current_row, col=1)
    current_row += 1

if show_macd:
    fig.add_trace(go.Scatter(
        x=chart_df.index, 
        y=chart_df['macd'], 
        name='MACD', 
        line=dict(color='#ab47bc', width=1.5)
    ), row=current_row, col=1)
    current_row += 1

if show_volume:
    close_arr = chart_df['Close'].to_numpy().ravel()
    open_arr = chart_df['Open'].to_numpy().ravel()
    colors = ['#089981' if c >= o else '#f23645' for c, o in zip(close_arr, open_arr)]
    fig.add_trace(go.Bar(
        x=chart_df.index, 
        y=chart_df['Volume'], 
        name='Volume', 
        marker_color=colors
    ), row=current_row, col=1)

# Chart Layout Styling
fig.update_layout(
    height=320 + (num_rows * 140),
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#1e222d",
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(showgrid=True, gridcolor="#2a2e39", rangeslider=dict(visible=False)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# Apply grid to all subplots
for r in range(1, num_rows + 1):
    fig.update_yaxes(showgrid=True, gridcolor="#2a2e39", row=r, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#2a2e39", row=r, col=1)

st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# OPTIONAL TECHNICAL AUDIT SECTION
# ==============================================================================

if show_technical_audit:
    st.markdown("---")
    st.markdown("### Technical Audit & Model Methodology")
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("#### Model Architecture & Validation")
        st.markdown("""
        - **Target Formulation:** $y_t = 1$ if $\\text{Close}_{t+1} > \\text{Close}_t$, else $0$.
        - **Validation Strategy:** Chronological split (80% training window, 20% unseen test window).
        - **Out-of-Fold Scaling:** `StandardScaler` fitted strictly on training observations to prevent data leakage.
        - **Classifier:** Random Forest Classifier (200 trees, max depth 5).
        """)
        
    with col_t2:
        st.markdown("#### Performance Metrics vs Baseline")
        prec = precision_score(test['target'], test_predictions, zero_division=0)
        rec = recall_score(test['target'], test_predictions, zero_division=0)
        f1 = f1_score(test['target'], test_predictions, zero_division=0)
        
        metrics_table = pd.DataFrame([
            {"Model": "Random Forest", "Accuracy": f"{model_accuracy:.2%}", "Precision": f"{prec:.2%}", "Recall": f"{rec:.2%}", "F1 Score": f"{f1:.2%}"},
            {"Model": "Majority Class Baseline", "Accuracy": f"{majority_accuracy:.2%}", "Precision": "N/A", "Recall": "N/A", "F1 Score": "N/A"}
        ])
        st.table(metrics_table)
        
        n_test = len(test)
        n_correct = int(round(model_accuracy * n_test))
        b_test = binomtest(n_correct, n_test, p=majority_accuracy, alternative='greater')
        st.caption(f"Binomial Test p-value: {b_test.pvalue:.4f}")

st.markdown("---")
st.caption("Disclaimer: This tool is strictly for academic and demonstration purposes. It does not constitute financial advice or investment recommendations.")
