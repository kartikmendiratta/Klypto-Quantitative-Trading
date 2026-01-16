# NIFTY 50 Quantitative Trading System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/XGBoost-2.0+-006400?style=for-the-badge" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/HMM-Regime%20Detection-9B59B6?style=for-the-badge" alt="HMM"/>
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge" alt="Status"/>
</p>

<p align="center">
  <strong>Complete end-to-end quantitative trading system for NIFTY 50 derivatives with HMM regime detection, sequential filtering architecture, and deep learning-based trade validation.</strong>
</p>

<p align="center">
  <em>Testing Period: September - December 2025 | ~19,500 5-minute bars</em>
</p>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Results](#-key-results)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Technical Details](#-technical-details)
- [Future Improvements](#-future-improvements)

---

## 🎯 Project Overview

This project implements a **Sequential Filtering Architecture** for algorithmic trading on NIFTY 50:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SEQUENTIAL FILTERING PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   📊 DATA        →   🔧 FEATURES    →   📈 REGIME     →   📉 SIGNALS       │
│   Spot/Futures       70+ Features       HMM 3-State       EMA Crossover    │
│   Options            Greeks, IV         Detection         + Regime Filter  │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│   🤖 ML FILTER   →   ✅ EXECUTION                                          │
│   XGBoost/LSTM       High-Confidence                                       │
│   Validation         Trades Only                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Description |
|-----------|-------------|
| **Data Pipeline** | 5-minute OHLCV for Spot, Futures (with rollover), and Options (ATM ± 2 strikes) |
| **Feature Engineering** | 70+ features: Greeks (Black-Scholes), EMA, RSI, ATR, IV metrics, PCR |
| **Regime Detection** | Hidden Markov Model classifying 3 states: Uptrend (+1), Sideways (0), Downtrend (-1) |
| **Trading Strategy** | 5/15 EMA Crossover with regime filter (LONG in uptrend, SHORT in downtrend) |
| **ML Enhancement** | XGBoost and LSTM models filter signals, taking only high-confidence trades |

---

## 📈 Key Results

### Performance Comparison

| Metric | Baseline | XGBoost | **LSTM** |
|--------|----------|---------|----------|
| **Total Trades** | 140 | 27 | **9** |
| **Win Rate (%)** | 29.3% | 22.2% | **44.4%** ✅ |
| **Total Return (%)** | -0.54% | -0.50% | **+0.18%** ✅ |
| **Sharpe Ratio** | -5.72 | -34.4 | **+30.7** ✅ |
| **Sortino Ratio** | -19.5 | -98.6 | **+97.6** ✅ |
| **Max Drawdown (%)** | 1.21% | 0.55% | **0.15%** ✅ |
| **Profit Factor** | 0.90 | 0.54 | **1.85** ✅ |

### Key Insights

- **LSTM is the only profitable configuration** with positive returns and profit factor > 1
- Win rate improved by **+15.2 percentage points** (29.3% → 44.4%)
- Sharpe ratio transformed from **-5.72 to +30.7** (risk-adjusted returns massively improved)
- Max drawdown reduced by **88%** (1.21% → 0.15%)
- Trade quality over quantity: **93% fewer trades** (140 → 9), but significantly higher conviction
- Outlier trades (Z-score > 3) represent **2.4%** of profitable trades but contribute **~10%** of total profits

### Outlier Trade Characteristics

| Metric | Outlier Trades | Normal Profitable |
|--------|----------------|-------------------|
| Average P&L | ₹461.92 | ₹107.42 |
| Average Duration | 1,430 minutes | 658 minutes |
| Regime Distribution | 100% Downtrend | 75% Downtrend, 25% Uptrend |
| Average IV | 0.1399 | 0.1253 |
| Entry Time | Late session (15:00) | Distributed throughout day |

---

## 🔧 Installation

### Prerequisites

- **Python 3.11+**
- **8GB RAM** (recommended for LSTM training)
- **pip** package manager

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/kartikmendiratta/nifty-quant-trading.git
cd nifty-quant-trading

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥2.0.0 | Data manipulation |
| `numpy` | ≥1.24.0 | Numerical computing |
| `yfinance` | ≥0.2.36 | Market data (spot) |
| `mibian` | ≥0.1.3 | Black-Scholes Greeks |
| `hmmlearn` | ≥0.3.0 | Hidden Markov Model |
| `xgboost` | ≥2.0.0 | Gradient boosting |
| `tensorflow` | ≥2.15.0 | LSTM neural network |
| `scikit-learn` | ≥1.3.0 | ML utilities |
| `matplotlib` | ≥3.8.0 | Visualization |
| `python-pptx` | ≥0.6.21 | PowerPoint generation |

---

## 🚀 How to Run

### Option 1: Run All Notebooks Sequentially (Recommended)

Execute notebooks in order for complete pipeline understanding:

```bash
# Start Jupyter
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

**Execution Order:**

| # | Notebook | Purpose | Runtime | Output |
|---|----------|---------|---------|--------|
| 1 | `01_data_acquisition.ipynb` | Fetch/load market data | ~2 min | `data/*.csv` files |
| 2 | `02_data_cleaning.ipynb` | Clean, merge, validate | ~1 min | `nifty_merged_5min.csv` |
| 3 | `03_features.ipynb` | Engineer 70+ features | ~3 min | `nifty_features_5min.csv` |
| 4 | `04_regime_detection.ipynb` | Train HMM, detect regimes | ~5 min | `hmm_regime_model.joblib` |
| 5 | `05_baseline_strategy.ipynb` | Backtest EMA crossover | ~2 min | `baseline_trades.csv`, metrics |
| 6 | `06_ml_models.ipynb` | Train XGBoost & LSTM | ~15 min | All ML models, comparison |
| 7 | `07_outlier_analysis.ipynb` | Analyze high-performance trades | ~1 min | `outlier_insights.csv` |

**Total Runtime:** ~30 minutes on standard laptop

### Option 2: Quick Start with Python Scripts

```python
# Complete pipeline in Python
import pandas as pd
from src import data_utils, features, regime, strategy, ml_models

# 1. Load and prepare data
df_merged = pd.read_csv('data/nifty_merged_5min.csv', parse_dates=['timestamp'])

# 2. Engineer features
df_features = features.engineer_all_features(df_merged)

# 3. Detect regimes with HMM
hmm_model = regime.train_hmm(df_features)
df_features['regime'] = hmm_model.predict(df_features)

# 4. Run baseline strategy
baseline_trades = strategy.run_baseline(df_features)
print(f"Baseline trades: {len(baseline_trades)}, Win rate: {baseline_trades['pnl'].gt(0).mean():.1%}")

# 5. Apply LSTM filter
lstm_model = ml_models.LSTMModel.load('models/lstm_model.joblib')
lstm_trades = strategy.run_with_ml_filter(df_features, lstm_model)
print(f"LSTM trades: {len(lstm_trades)}, Win rate: {lstm_trades['pnl'].gt(0).mean():.1%}")
```

### Option 3: Load Pre-trained Models

```python
# Use existing models without retraining
from src.ml_models import XGBoostModel, LSTMModel
import joblib

# Load models
hmm = joblib.load('models/hmm_regime_model.joblib')
xgb = XGBoostModel.load('models/xgboost_model.joblib')
lstm = LSTMModel.load('models/lstm_model.joblib')

# Load processed data
df = pd.read_csv('data/nifty_features_5min.csv', parse_dates=['timestamp'])

# Predict regimes
regimes = hmm.predict(df[['avg_iv', 'pcr_oi', 'call_delta', 'futures_basis', 'returns']])

# Get ML predictions
lstm_proba = lstm.predict_proba(df)
high_confidence_signals = lstm_proba > 0.5
```

---

## 📁 Project Structure

```
nifty-quant-trading/
│
├── 📂 data/                              # Market data files (~19,500 rows each)
│   ├── nifty_spot_5min.csv              # NIFTY 50 Spot OHLCV
│   ├── nifty_futures_5min.csv           # Futures with OI & rollover
│   ├── nifty_options_5min.csv           # Options chain (ATM ± 2 strikes)
│   ├── nifty_merged_5min.csv            # Merged & cleaned dataset
│   ├── nifty_features_5min.csv          # 70+ engineered features
│   └── data_cleaning_report.txt         # Data quality report
│
├── 📂 notebooks/                         # 7 Jupyter notebooks (sequential)
│   ├── 01_data_acquisition.ipynb        # Data sourcing & loading
│   ├── 02_data_cleaning.ipynb           # Cleaning & merging
│   ├── 03_features.ipynb                # Feature engineering (Greeks, IV, etc.)
│   ├── 04_regime_detection.ipynb        # HMM training & regime labeling
│   ├── 05_baseline_strategy.ipynb       # EMA crossover backtest
│   ├── 06_ml_models.ipynb               # XGBoost & LSTM training
│   └── 07_outlier_analysis.ipynb        # High-performance trade analysis
│
├── 📂 src/                               # Production-ready Python modules
│   ├── __init__.py                      # Package initialization
│   ├── data_utils.py                    # Data loading, cleaning, validation
│   ├── features.py                      # Feature engineering functions
│   ├── greeks.py                        # Black-Scholes Greeks calculator
│   ├── regime.py                        # HMM regime detection class
│   ├── strategy.py                      # Trading strategy logic
│   ├── backtest.py                      # Backtesting engine
│   └── ml_models.py                     # XGBoost & LSTM wrapper classes
│
├── 📂 models/                            # Trained models (joblib & h5)
│   ├── hmm_regime_model.joblib          # HMM (3 states, Gaussian)
│   ├── xgboost_model.joblib             # XGBoost classifier
│   ├── lstm_model.joblib                # LSTM metadata + scaler
│   └── lstm_model_lstm.h5               # Keras LSTM weights
│
├── 📂 results/                           # All backtest results & metrics
│   ├── baseline_metrics.csv             # Baseline strategy performance
│   ├── baseline_trades.csv              # 140 baseline trades
│   ├── xgb_trades.csv                   # 27 XGBoost-filtered trades
│   ├── lstm_trades.csv                  # 9 LSTM-filtered trades
│   ├── ml_comparison.csv                # Model comparison table
│   ├── outlier_insights.csv             # Outlier analysis summary
│   └── outlier_trades.csv               # High-performance trade details
│
├── 📂 plots/                             # All visualizations (PNG, 150 DPI)
│   ├── 01_data_overview.png             # Data distribution & coverage
│   ├── 02_data_quality.png              # Missing values & outliers
│   ├── 03_feature_engineering.png       # Feature correlations
│   ├── 04_regime_detection.png          # HMM regime timeline
│   ├── 04_regime_statistics.png         # Regime distribution
│   ├── 05_baseline_strategy.png         # Baseline equity curve
│   ├── 06_ml_comparison.png             # Model performance comparison
│   ├── 06_xgb_importance.png            # XGBoost feature importance
│   ├── 06_xgb_confusion_matrix.png      # XGBoost confusion matrix
│   └── 07_outlier_analysis.png          # Outlier trade analysis (9 subplots)
│
├── 📄 .gitignore                         # Git ignore rules
├── 📄 requirements.txt                   # Python dependencies
└── 📄 README.md                          # This file
```

**Total Files:** 40+ files organized in 5 directories

---

## 📊 Methodology

### 1. Data Pipeline

**Data Sources:**
- **NIFTY 50 Spot:** 5-minute OHLCV bars from yfinance
- **NIFTY Futures:** Current month contract with automatic rollover handling
- **NIFTY Options:** ATM ± 2 strikes for both Calls and Puts

**Processing:**
- Timestamp alignment across all sources
- Forward-fill for gaps < 15 minutes
- Outlier detection using IQR method
- Train/Test split: 70% / 30% (time-series order, no shuffling)

### 2. Feature Engineering (70+ Features)

| Category | Features | Purpose |
|----------|----------|---------|
| **Technical** | EMA(5), EMA(15), RSI(14), ATR(14), MACD, Bollinger Bands | Trend & momentum |
| **Options Greeks** | Delta, Gamma, Theta, Vega (Black-Scholes, r=6.5%) | Risk exposure |
| **Volatility** | Avg IV, IV Spread (Call - Put), IV Skew | Market sentiment |
| **Market Structure** | PCR (OI), PCR (Volume), Futures Basis | Positioning |
| **Derived** | EMA Distance %, Net Delta Exposure, Gamma Exposure | Custom indicators |
| **Temporal** | Hour, Day of Week, Session (opening/closing) | Time patterns |

**Stationarity:** All features converted to returns, percentages, or ratios to avoid lookahead bias.

### 3. Hidden Markov Model (Regime Detection)

**Configuration:**
- **Library:** `hmmlearn.GaussianHMM`
- **States:** 3 (Uptrend = +1, Sideways = 0, Downtrend = -1)
- **Input Features:** `['avg_iv', 'iv_spread', 'pcr_oi', 'call_delta', 'call_gamma', 'futures_basis', 'returns']`
- **Training:** First 70% of data, EM algorithm with 100 iterations
- **Transition Matrix:** >99% self-persistence (stable regimes)

**Regime Interpretation:**
| State | Characteristics | Trading Action |
|-------|----------------|----------------|
| **+1 (Uptrend)** | Positive momentum, lower IV, bullish PCR | LONG entries allowed |
| **0 (Sideways)** | Range-bound, elevated IV, neutral PCR | **NO TRADES** |
| **-1 (Downtrend)** | Negative momentum, higher IV, bearish PCR | SHORT entries allowed |

### 4. Baseline Strategy (EMA Crossover + Regime Filter)

**Rules:**
```python
# LONG Entry
if (EMA_5 > EMA_15) and (regime == +1):
    enter_long()

# SHORT Entry  
if (EMA_5 < EMA_15) and (regime == -1):
    enter_short()

# Exit
if opposite_crossover or regime_change_to_sideways:
    exit_position()
```

**Entry Execution:** Next candle OPEN (no lookahead bias)

### 5. Machine Learning Enhancement

#### XGBoost Classifier
- **Algorithm:** Gradient Boosting Decision Trees
- **Hyperparameters:** 100 trees, max_depth=3, learning_rate=0.1
- **Validation:** Time-Series Cross-Validation (5 folds)
- **Features:** All stationary features at signal point
- **Output:** Binary probability (profitable = 1, loss = 0)

#### LSTM Neural Network
**Architecture:**
```
Input Layer (sequence_length=10, n_features)
    ↓
LSTM(64 units, return_sequences=True) + L2(0.001)
    ↓
Dropout(0.2)
    ↓
LSTM(32 units, return_sequences=False) + L2(0.001)
    ↓
BatchNormalization()
    ↓
Dense(64, activation='relu') + L2(0.001)
    ↓
Dropout(0.2)
    ↓
Dense(32, activation='relu')
    ↓
Dense(1, activation='sigmoid')  # Probability output
```

**Training Configuration:**
- **Optimizer:** Adam (initial_lr=0.0005)
- **Loss:** Binary Crossentropy
- **Callbacks:** 
  - `EarlyStopping(patience=15, restore_best_weights=True)`
  - `ReduceLROnPlateau(factor=0.5, patience=5)`
- **Epochs:** 100 (typically converges ~40-50)
- **Batch Size:** 32
- **Sequence Length:** 10 time steps (50 minutes of lookback)

**Target Alignment:** LSTM predicts profitability for the **last row** in each sequence (no future peeking).

**Warmup Period:** First 10 bars default to 0.5 probability (neutral, no bias).

### 6. ML Trade Filter

**Decision Rule:**
```python
if ml_probability >= 0.5:
    execute_trade()  # High confidence
else:
    skip_trade()     # Low confidence
```

**Result:** Only 9 high-confidence trades executed (vs 140 baseline), but 44.4% win rate.

---

## 🔬 Technical Details

### Preventing Lookahead Bias

**Critical Design Decisions:**

1. **Signal Generation:** All signals generated at time `t`, entry at `t+1` open price
2. **Feature Engineering:** Only use data available at signal time (no future peeking)
3. **HMM Training:** Trained only on first 70% of data, tested on remaining 30%
4. **ML Models:** Time-series cross-validation (no data shuffling)
5. **Regime Labels:** Not used during signal generation, only for post-analysis

### Stationary Features (No Raw Prices)

All ML model inputs are stationary to ensure valid predictions:

| Feature Type | Examples | Transformation |
|--------------|----------|----------------|
| **Returns** | `spot_returns`, `futures_returns` | `log(price_t / price_t-1)` |
| **Distances** | `ema_diff_pct` | `(price - EMA) / EMA * 100` |
| **Ratios** | `pcr_oi`, `iv_spread`, `delta_ratio` | Already stationary |
| **Normalized** | `rsi`, `atr_pct`, `normalized_gamma` | 0-100 scale or % of price |
| **Greeks** | All Greeks normalized by spot price | `delta / spot_price` |

**Why Stationary?** Prevents model from memorizing price levels instead of learning patterns.

### Sequence Creation for LSTM

```python
# Example: Creating 10-step sequences
sequences = []
for i in range(10, len(df)):
    seq = df.iloc[i-10:i][feature_cols].values  # 10 rows
    target = df.iloc[i]['is_profitable']        # Target for last row
    sequences.append((seq, target))
```

**Warmup Handling:** First 10 bars cannot form full sequence → default probability = 0.5 (neutral).

### Train/Test Split

```python
train_size = int(0.7 * len(df))
train_df = df.iloc[:train_size]   # First 70%
test_df = df.iloc[train_size:]    # Last 30%

# No shuffling! Preserve temporal order
```

### Performance Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Win Rate** | `winning_trades / total_trades` | Higher is better (>50% ideal) |
| **Profit Factor** | `gross_profit / gross_loss` | >1 = profitable, >2 = excellent |
| **Sharpe Ratio** | `(returns - rf) / std(returns)` | Risk-adjusted returns, >1 good |
| **Sortino Ratio** | `(returns - rf) / downside_std` | Penalizes downside volatility only |
| **Max Drawdown** | `max(cumulative_peak - current)` | Largest peak-to-trough decline |
| **Calmar Ratio** | `annual_return / max_drawdown` | Return per unit of drawdown risk |

---

## 🚧 Future Improvements

### 1. Live Trading Integration
- **Broker APIs:** Zerodha Kite, ICICI Breeze, Upstox, Groww
- **Execution:** Automated order placement with SL/TP
- **Monitoring:** Real-time P&L tracking and alerts

### 2. Advanced ML Techniques
- **Ensemble Models:** Weighted voting (XGBoost + LSTM + Random Forest)
- **Transformer Models:** Replace LSTM with attention-based architecture
- **Reinforcement Learning:** Deep Q-Network (DQN) for dynamic action selection
- **AutoML:** Hyperparameter optimization with Optuna/Ray Tune

### 3. Risk Management Enhancements
- **Dynamic Position Sizing:** Kelly Criterion based on ML confidence
- **Adaptive Stop-Loss:** ATR-based trailing stops
- **Take-Profit Levels:** Fibonacci retracements, support/resistance
- **Max Drawdown Control:** Halt trading when drawdown exceeds threshold

### 4. Feature Engineering Expansion
- **Order Book Data:** Bid-ask spread, market depth, order imbalance
- **Sentiment Analysis:** News sentiment, social media (Twitter/Reddit)
- **Alternative Data:** VIX, global indices correlation, macroeconomic indicators
- **Higher Timeframes:** 15-min, 1-hour regime alignment

### 5. Multi-Asset Strategy
- **Bank Nifty:** Apply same pipeline to Bank Nifty futures/options
- **Sector Indices:** Nifty IT, Pharma, Auto, Financial Services
- **Stock-Specific:** Large-cap stocks with liquid options
- **Pairs Trading:** Relative value between correlated assets

### 6. Performance Optimization
- **Parallel Processing:** Multiprocessing for feature engineering
- **Database Integration:** PostgreSQL/MongoDB for faster data access
- **Cloud Deployment:** AWS/GCP for scalable backtesting
- **Real-time Inference:** Optimize LSTM for <100ms prediction latency

### 7. Robustness Testing
- **Walk-Forward Analysis:** Rolling train/test windows
- **Monte Carlo Simulation:** Test strategy on synthetic data
- **Stress Testing:** Performance during market crashes/rallies
- **Out-of-Sample Validation:** Test on 2026 data (future validation)

### 8. Explainability & Monitoring
- **SHAP Values:** Understand feature importance per trade
- **Trade Autopsy:** Detailed post-mortem of each trade
- **Dashboard:** Streamlit/Plotly Dash for live monitoring
- **Alerts:** Telegram/Email notifications for signals and errors

---

## ⚠️ Disclaimer

**This project is for educational and research purposes only.**

- ⚠️ **Not Financial Advice:** Do not use this system for real trading without thorough testing
- 📉 **Past Performance ≠ Future Results:** Backtest results do not guarantee live performance
- 💰 **Risk of Loss:** Trading involves substantial risk of loss
- 🧪 **Paper Trade First:** Always validate on paper trading before deploying real capital
- 📚 **Educational Use:** Designed to demonstrate quantitative trading concepts and ML techniques

**Use at your own risk. The author assumes no liability for financial losses.**

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 👨‍💻 Author

**Kartik Mendiratta**
- GitHub: [@kartikmendiratta](https://github.com/kartikmendiratta)
- LinkedIn: [Kartik Mendiratta](https://linkedin.com/in/kartikmendiratta)

---

## 🙏 Acknowledgments

- **Data Sources:** Yahoo Finance (yfinance)
- **Libraries:** TensorFlow, XGBoost, hmmlearn, scikit-learn
- **Inspiration:** Quantitative finance community and academic research

---

## 📧 Contact

For questions, suggestions, or collaboration:
- **Open an issue:** [GitHub Issues](https://github.com/kartikmendiratta/nifty-quant-trading/issues)
- **Discussions:** [GitHub Discussions](https://github.com/kartikmendiratta/nifty-quant-trading/discussions)

---

## 📊 Project Stats

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-orange.svg)

---

<p align="center">
  <strong>⭐ Star this repository if you found it helpful!</strong>
</p>

<p align="center">
  <em>Last Updated: January 17, 2026</em>
</p>
