# NIFTY 50 Quantitative Trading System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/XGBoost-2.0+-006400?style=for-the-badge" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge" alt="Status"/>
</p>

<p align="center">
  <strong>A complete end-to-end quantitative trading pipeline for NIFTY 50 derivatives featuring HMM regime detection, EMA crossover signals, and ML-based trade filtering.</strong>
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
- Sharpe ratio improved from **-5.72 to +30.7** (massive risk-adjusted improvement)
- Max drawdown reduced by **88%** (1.21% → 0.15%)
- Trade filtering: 93% reduction in trades, but significantly higher quality

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

### Option 1: Jupyter Notebooks (Recommended for Learning)

Execute notebooks in order to understand each step:

```bash
cd notebooks
jupyter notebook
```

| Notebook | Purpose | Output |
|----------|---------|--------|
| `01_data_acquisition.ipynb` | Fetch/generate data | `data/*.csv` |
| `02_data_cleaning.ipynb` | Clean & merge data | `nifty_merged_5min.csv` |
| `03_features.ipynb` | Feature engineering | `nifty_features_5min.csv` |
| `04_regime_detection.ipynb` | Train HMM | `hmm_regime_model.joblib` |
| `05_baseline_strategy.ipynb` | Backtest EMA strategy | `baseline_trades.csv` |
| `06_ml_models.ipynb` | Train XGBoost & LSTM | `ml_comparison.csv` |
| `07_outlier_analysis.ipynb` | Analyze outliers | `outlier_insights.csv` |

### Option 2: Python Scripts

```python
# Load data and run strategy
import pandas as pd
from src.strategy import MLFilteredStrategy
from src.ml_models import LSTMModel

# Load preprocessed data
df = pd.read_csv('data/nifty_regime_5min.csv', parse_dates=['timestamp'])

# Load trained LSTM model
lstm = LSTMModel.load('models/lstm_model.joblib')

# Run ML-filtered strategy
strategy = MLFilteredStrategy(model=lstm, confidence_threshold=0.5)
trades = strategy.run_with_ml(df)

print(f"LSTM Filtered Trades: {len(trades)}")
```

### Option 3: Generate Presentation

```bash
python generate_presentation.py
# Output: NIFTY_Trading_Strategy_Final.pptx
```

---

## 📁 Project Structure

```
nifty-quant-trading/
│
├── 📂 data/                           # Data files
│   ├── nifty_spot_5min.csv           # Spot OHLCV data
│   ├── nifty_futures_5min.csv        # Futures with OI
│   ├── nifty_options_5min.csv        # Options chain
│   ├── nifty_merged_5min.csv         # Merged dataset
│   ├── nifty_features_5min.csv       # Feature-engineered
│   └── nifty_regime_5min.csv         # With regime labels
│
├── 📂 notebooks/                      # Jupyter notebooks
│   ├── 01_data_acquisition.ipynb     # Data fetching
│   ├── 02_data_cleaning.ipynb        # Data preprocessing
│   ├── 03_features.ipynb             # Feature engineering
│   ├── 04_regime_detection.ipynb     # HMM training
│   ├── 05_baseline_strategy.ipynb    # EMA strategy backtest
│   ├── 06_ml_models.ipynb            # ML model training
│   └── 07_outlier_analysis.ipynb     # Outlier analysis
│
├── 📂 src/                            # Python modules
│   ├── __init__.py
│   ├── data_utils.py                 # Data fetching & cleaning
│   ├── features.py                   # Feature engineering
│   ├── greeks.py                     # Black-Scholes Greeks
│   ├── regime.py                     # HMM regime detection
│   ├── strategy.py                   # Trading strategy
│   ├── backtest.py                   # Backtesting engine
│   └── ml_models.py                  # XGBoost & LSTM models
│
├── 📂 models/                         # Saved models
│   ├── hmm_regime_model.joblib       # Trained HMM
│   ├── xgboost_model.joblib          # Trained XGBoost
│   └── lstm_model.h5                 # Trained LSTM
│
├── 📂 results/                        # Output files
│   ├── baseline_metrics.csv          # Baseline performance
│   ├── baseline_trades.csv           # Trade records
│   ├── ml_comparison.csv             # Model comparison
│   ├── xgb_trades.csv                # XGBoost filtered trades
│   └── lstm_trades.csv               # LSTM filtered trades
│
├── 📂 plots/                          # Visualizations
│   ├── 01_data_overview.png
│   ├── 04_regime_detection.png
│   ├── 05_baseline_strategy.png
│   └── 06_ml_comparison.png
│
├── generate_presentation.py          # PowerPoint generator
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

---

## 📊 Methodology

### 1. Hidden Markov Model (Regime Detection)

**Configuration:**
- **States:** 3 (Uptrend, Sideways, Downtrend)
- **Input Features:** Avg IV, IV Spread, PCR, ATM Greeks, Futures Basis, Returns
- **Training:** First 70% of data

**Interpretation:**
| State | Label | Trading Action |
|-------|-------|----------------|
| +1 | Uptrend | LONG entries allowed |
| 0 | Sideways | NO trades |
| -1 | Downtrend | SHORT entries allowed |

### 2. LSTM Model (Trade Filter)

**Architecture:**
```
Input (sequence_length=10, n_features)
    ↓
LSTM(64, return_sequences=True) + L2 regularization
    ↓
Dropout(0.2)
    ↓
LSTM(32, return_sequences=False) + L2 regularization
    ↓
BatchNormalization
    ↓
Dense(64, ReLU) + L2 regularization
    ↓
Dropout(0.2)
    ↓
Dense(32, ReLU)
    ↓
Dense(1, Sigmoid) → Probability of profitable trade
```

**Training:**
- **Epochs:** 100 (with EarlyStopping, patience=15)
- **Optimizer:** Adam (lr=0.0005 with ReduceLROnPlateau)
- **Target:** Binary (1 = profitable trade, 0 = loss)

---

## 🔬 Technical Details

### Stationary Features

All ML features are stationary to prevent lookahead bias:

| Feature Type | Examples |
|--------------|----------|
| **Returns** | Log returns, momentum (not raw prices) |
| **Distances** | Distance from EMA as % (not raw EMA values) |
| **Ratios** | PCR, Delta ratios, IV spreads |
| **Normalized** | ATR%, RSI, normalized Greeks |

### No Lookahead Bias

- Signals generated at time T, entry at T+1 open
- HMM trained only on past data
- ML models use time-series cross-validation
- Regime labels not peeked during signal generation

---

## 🚧 Future Improvements

1. **Live Integration:** Connect to Zerodha Kite, ICICI Breeze, or Groww API
2. **Ensemble Models:** Combine XGBoost + LSTM voting
3. **Position Sizing:** Kelly Criterion based on ML confidence
4. **Risk Management:** Dynamic stop-loss and take-profit levels
5. **Multi-Asset:** Extend to Bank Nifty, sector indices
6. **Transformer Models:** Replace LSTM with attention-based models

---

## 📝 License

This project is for **educational purposes only**. Always paper trade before deploying with real capital.

---

## 👨‍💻 Author

**Kartik Mendiratta**

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

*Last Updated: January 2026*
