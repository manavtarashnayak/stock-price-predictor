import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import binomtest
import ta

st.set_page_config(
    page_title="Stock Price Movement Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title & Header
st.title("📈 Leak-Free Stock Price Movement Predictor")
st.markdown("""
Predicting next-day stock price directional movement (Up / Down) using Machine Learning with strict leak-free temporal evaluation and dual baselines.
""")

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")
ticker_option = st.sidebar.selectbox("Select Stock Ticker", ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "BTC-USD", "Custom..."])
if ticker_option == "Custom...":
    ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL").upper()
else:
    ticker = ticker_option

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
test_ratio = st.sidebar.slider("Test Set Split Ratio", 0.1, 0.3, 0.2, step=0.05)

@st.cache_data(ttl=3600)
def load_and_prep_data(ticker_symbol, start, end):
    df = yf.download(ticker_symbol, start=start, end=end)
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

with st.spinner(f"Fetching market data for {ticker}..."):
    df = load_and_prep_data(ticker, start_date, end_date)

if df is None or len(df) < 100:
    st.error(f"Not enough historical data found for '{ticker}'. Please check the ticker symbol or expand the date range.")
    st.stop()

# Train / Test Split
split_idx = int(len(df) * (1 - test_ratio))
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

features_raw = ['Open', 'High', 'Low', 'Close', 'Volume']
features_eng = features_raw + ['rsi', 'macd', 'sma_20']

# Scaling (Fit on Train ONLY to avoid temporal data leakage)
scaler_raw = StandardScaler().fit(train[features_raw])
X_train_raw = scaler_raw.transform(train[features_raw])
X_test_raw = scaler_raw.transform(test[features_raw])

scaler_eng = StandardScaler().fit(train[features_eng])
X_train_eng = scaler_eng.transform(train[features_eng])
X_test_eng = scaler_eng.transform(test[features_eng])

# Model Training
model_lr_raw = LogisticRegression(max_iter=1000).fit(X_train_raw, train['target'])
model_lr_eng = LogisticRegression(max_iter=1000).fit(X_train_eng, train['target'])

model_rf_raw = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42).fit(X_train_raw, train['target'])
model_rf_eng = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42).fit(X_train_eng, train['target'])

# Predictions
pred_lr_raw = model_lr_raw.predict(X_test_raw)
pred_lr_eng = model_lr_eng.predict(X_test_eng)
pred_rf_raw = model_rf_raw.predict(X_test_raw)
pred_rf_eng = model_rf_eng.predict(X_test_eng)

# Baselines
persistence_pred = test['target'].shift(1).bfill()
majority_class = train['target'].mode()[0]
majority_pred = np.full(len(test), majority_class)

# Metrics calculation
def get_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }

metrics_df = pd.DataFrame([
    {'Model': 'Persistence Baseline', 'Feature Set': 'Shifted Target', **get_metrics(test['target'], persistence_pred)},
    {'Model': 'Majority Class Baseline', 'Feature Set': 'None (Train Mode)', **get_metrics(test['target'], majority_pred)},
    {'Model': 'Logistic Regression', 'Feature Set': 'Raw (OHLCV)', **get_metrics(test['target'], pred_lr_raw)},
    {'Model': 'Logistic Regression', 'Feature Set': 'Engineered (OHLCV+Ind)', **get_metrics(test['target'], pred_lr_eng)},
    {'Model': 'Random Forest', 'Feature Set': 'Raw (OHLCV)', **get_metrics(test['target'], pred_rf_raw)},
    {'Model': 'Random Forest', 'Feature Set': 'Engineered (OHLCV+Ind)', **get_metrics(test['target'], pred_rf_eng)},
]).sort_values('Accuracy', ascending=False).reset_index(drop=True)

# Next-Day Live Forecast for Latest Market Day
latest_features_eng = scaler_eng.transform(df[features_eng].iloc[[-1]])
latest_pred_prob = model_rf_eng.predict_proba(latest_features_eng)[0]
latest_pred = model_rf_eng.predict(latest_features_eng)[0]

# Display Live Prediction Banner
st.subheader(f"⚡ Live Forecast for Next Market Session ({ticker})")
col_b1, col_b2, col_b3 = st.columns(3)
with col_b1:
    signal = "🟢 BULLISH (UP)" if latest_pred == 1 else "🔴 BEARISH (DOWN)"
    st.metric("Predicted Direction", signal)
with col_b2:
    st.metric("Up Probability", f"{latest_pred_prob[1]*100:.1f}%")
with col_b3:
    st.metric("Down Probability", f"{latest_pred_prob[0]*100:.1f}%")

st.markdown("---")

# Main Layout Tabs
tab1, tab2, tab3 = st.tabs(["📊 Stock & Indicators Chart", "🏆 Model Performance & Evaluation", "🔬 Leak-Free Methodology"])

with tab1:
    st.subheader(f"{ticker} Price & Technical Indicators")
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=(f'{ticker} Closing Price & SMA 20', 'RSI (14)', 'MACD'))
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='#00F0FF')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', line=dict(color='#FF007F', dash='dash')), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='#FFD700')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='#00FF66')), row=3, col=1)
    
    fig.update_layout(height=650, template="plotly_dark", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Model Comparison against Dual Baselines")
    st.dataframe(metrics_df.style.format({
        'Accuracy': '{:.2%}',
        'Precision': '{:.2%}',
        'Recall': '{:.2%}',
        'F1-Score': '{:.2%}'
    }).highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='#1e4620'), use_container_width=True)
    
    # Statistical significance
    best_model_row = metrics_df[~metrics_df['Model'].str.contains('Baseline')].iloc[0]
    best_acc = best_model_row['Accuracy']
    majority_acc = metrics_df[metrics_df['Model'].str.contains('Majority')]['Accuracy'].values[0]
    
    n_test = len(test)
    n_correct = int(round(best_acc * n_test))
    b_test = binomtest(n_correct, n_test, p=majority_acc, alternative='greater')
    
    st.markdown("### 🧪 Statistical Significance Test (Binomial Test)")
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Best ML Model Accuracy", f"{best_acc:.2%}")
    col_s2.metric("Majority Baseline Rate", f"{majority_acc:.2%}")
    col_s3.metric("p-value", f"{b_test.pvalue:.4f}")
    
    if b_test.pvalue < 0.05:
        st.success(f"✅ The best model ({best_model_row['Model']} - {best_model_row['Feature Set']}) is **statistically significantly better** than majority baseline (p < 0.05).")
    else:
        st.warning(f"⚠️ The performance difference over majority baseline is **not statistically significant** (p = {b_test.pvalue:.4f} >= 0.05). This confirms the difficulty of beating baseline random walk dynamics on daily stock returns.")

with tab3:
    st.subheader("🛡️ Leakage Prevention Safeguards")
    st.markdown("""
    - **No Look-Ahead Target Leakage**: Target $y_t = \\mathbb{I}(\\text{Close}_{t+1} > \\text{Close}_t)$ is computed before any feature aggregation or splitting.
    - **Temporal Chronological Split**: Strictly uses past data for training and future unseen data for testing (80% / 20%). No standard random `train_test_split` is used.
    - **Out-of-Fold Feature Scaling**: `StandardScaler` is fitted *only* on the training set to prevent statistics from the test set leaking into model normalization.
    - **Dual Baselines Included**: Compares ML models against both Persistence (Naive shift) and Majority Class baselines to prevent illusory high accuracy metrics.
    """)

st.caption("Disclaimer: This tool is for educational and research purposes only. Not financial advice.")
