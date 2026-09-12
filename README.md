# Stock Price Movement Predictor

**Quantitative Machine Learning Pipeline & Interactive Web Application**

---

## 1. Overview
This repository contains a reproducible, end-to-end Machine Learning pipeline and web application for stock price movement prediction. The project predicts the next-day price movement direction (**Up / Down**) for liquid equities and crypto assets using daily historical Open-High-Low-Close-Volume (OHLCV) data.

The project prioritizes **technical rigor, strict data leakage prevention, naive baseline comparisons, chronological evaluation, and honest statistical analysis**. It includes an interactive **Streamlit Web Dashboard** (`app.py`) for real-time predictions, customizable technical charts, and model inspection.

---

## 2. Dataset
- **Source**: Yahoo Finance (`yfinance` API).
- **Supported Assets**: Liquid US Equities (`AAPL`, `TSLA`, `NVDA`, `MSFT`, `AMZN`), Crypto (`BTC-USD`), and ETFs (`SPY`).
- **Period**: 2018-01-01 to Present (Daily resolution).
- **Features**: `Open`, `High`, `Low`, `Close`, `Volume`.
- **Liquidity Rationale**: Liquid securities exhibit continuous trading, tight bid-ask spreads, and active price discovery, minimizing execution noise.

---

## 3. Methodology
The pipeline executes linearly as follows:
1. **Configurable Data Ingestion**: Historical price download and column standardizations.
2. **Leak-Free Target Engineering**: Generating $y_t$ via `.shift(-1)` and dropping incomplete trailing observation.
3. **Class Balance Reporting**: Calculating class distributions and saving distribution plot to `outputs/class_balance.png`.
4. **Baseline Setup**: Defining Majority Class and Persistence baselines.
5. **Feature Engineering**: Computing technical indicators (SMA-20, EMA-50, RSI-14, MACD).
6. **Chronological Splitting & Scaler Fitting**: Partitioning data 80/20 without shuffling and fitting scalers ONLY on training features.
7. **Model Training & Multi-Metric Evaluation**: Training Logistic Regression and Random Forest classifiers on both raw and engineered feature sets.
8. **Statistical Significance Testing**: Conducting a Binomial test against baseline performance.
9. **Visualization & Generalization Analysis**: Saving `outputs/prediction_plot.png` and reporting out-of-sample findings.

---

## 4. Leakage Prevention
Data leakage is explicitly prevented across three levels:

1. **Target Construction Safeguard**:
   The target uses `shift(-1)` to map tomorrow's outcome to today's row $t$. We verify row-by-row that future prices are NEVER passed into model feature matrices $X_t$.

   *Verification Table:*
   | Date | Today Close | Next-Day Close | Target (Label) | Expected Target | Match |
   | :--- | :--- | :--- | :--- | :--- | :--- |
   | 2018-01-02 | 40.23 | 40.23 | 0 | 0 | True |
   | 2018-01-03 | 40.23 | 40.41 | 1 | 1 | True |
   | 2018-01-04 | 40.41 | 40.87 | 1 | 1 | True |
   | 2018-01-05 | 40.87 | 40.72 | 0 | 0 | True |
   | 2018-01-08 | 40.72 | 40.72 | 0 | 0 | True |

2. **Look-Ahead Bias Elimination**:
   Rolling features (e.g., 20-day SMA) look strictly backward ($t-19$ to $t$).

3. **Preprocessing Leakage Prevention**:
   `StandardScaler` is fitted ONLY on the training partition (`scaler.fit(X_train)`). Test features are transformed using parameters derived exclusively from training history.

---

## 5. Baselines
1. **Majority Class Baseline**: Predicts the dominant class learned from the training set for all test days (~52.90% accuracy).
2. **Persistence Baseline**: Predicts tomorrow's price direction will match today's direction (~51.28% accuracy).

---

## 6. Feature Sets
- **Feature Set A (Raw Features)**: `Open`, `High`, `Low`, `Close`, `Volume`.
- **Feature Set B (Engineered Features)**: Adds technical indicators:
  - **SMA-20**: 20-day Simple Moving Average (trend smoothing).
  - **EMA-50**: 50-day Exponential Moving Average (trend tracking).
  - **RSI-14**: Relative Strength Index over 14 days (momentum oscillator).
  - **MACD**: Moving Average Convergence Divergence (trend momentum).

---

## 7. Models
1. **Logistic Regression**: Linear binary classifier providing probabilistic outputs and interpretable feature coefficients.
2. **Random Forest Classifier**: Non-linear ensemble model using 200 decision trees (`max_depth=5`) to evaluate feature interactions without assuming linear separability.

---

## 8. Evaluation
The project uses strict chronological evaluation:
- **Train Period**: 2018-02-07 to 2024-12-09 (1,721 rows / 80%).
- **Test Period**: 2024-12-10 to 2026-08-31 (431 rows / 20%).
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Binomial $p$-value.

---

## 9. Results

### Model Comparison Table

| Approach / Model | Feature Set | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | Previous Direction | 0.5128 | 0.5395 | 0.5395 | 0.5395 |
| **Majority Class Baseline** | None (Train Mode) | 0.5290 | 0.5290 | 1.0000 | 0.6920 |
| **Logistic Regression** | Raw (OHLCV) | **0.5383** | 0.5343 | 0.9912 | 0.6943 |
| **Logistic Regression** | Engineered (OHLCV+Ind) | 0.5267 | 0.5300 | 0.9298 | 0.6752 |
| **Random Forest** | Raw (OHLCV) | 0.5104 | 0.5208 | 0.9342 | 0.6688 |
| **Random Forest** | Engineered (OHLCV+Ind) | 0.5174 | 0.5238 | 0.9649 | 0.6790 |

---

## 10. Limitations
1. **Market Efficiency**: Daily OHLCV price history is rapidly digested by market participants, leaving minimal standalone predictive signal for next-day close.
2. **Feature Scope**: The model relies solely on price and volume without macro indicators, order book liquidity depth, or sentiment analysis.
3. **Execution Costs Excluded**: Signal evaluation does not incorporate trading fees, slippage, or spread costs.

---

## 11. Future Improvements
1. **Walk-Forward Validation**: Implement continuous daily model updating.
2. **Returns-Based Features**: Incorporate log returns and realized volatility metrics.
3. **Multi-Asset Signals**: Integrate cross-asset correlations (e.g., Crude Oil, USD/INR, US Indices).
4. **Transaction-Cost Backtesting**: Simulate net strategy profitability after broker fees and slippage.

---

## 12. Local Execution & Deployment

### Run Web App Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Launch interactive Streamlit web dashboard
streamlit run app.py
```

### Run Jupyter Notebook
```bash
jupyter notebook stock_price_predictor.ipynb
```

---

## Output Artifacts
- **Class Balance Plot**: [`outputs/class_balance.png`](outputs/class_balance.png)
- **Prediction Plot**: [`outputs/prediction_plot.png`](outputs/prediction_plot.png)
