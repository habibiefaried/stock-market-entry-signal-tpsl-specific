# Stock Market Entry Signal — TP/SL Specific

Predicts whether to go **LONG or SHORT** on a given stock to maximize the probability of hitting Take Profit (1.5x ATR) before Stop Loss (1x ATR).

## Quick Start

```bash
pyenv local 3.11.9
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# For GPU (Windows with NVIDIA):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 1. Fetch data (9 years default)
python fetch_stock_data.py --ticker AAPL

# 2. Train model
python train_xgboost.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv

# 3. Train with Genetic Algorithm + LSTM/CNN features (best results)
python train_xgboost_ga.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv

# 4. Generate HTML report with backtest + Monte Carlo
python generate_report.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv

# 5. Live trade decision
python current.py --ticker AAPL --price 302.50
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  fetch_stock_data.py                                                │
│  Yahoo Finance → OHLCV → 170+ indicators → TP/SL labels            │
│  Output: data/{TICKER}_tpsl_data_{YYYYMMDD}.csv                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌───────────────────┐ ┌─────────────────────┐ ┌──────────────────────┐
│ train_xgboost.py  │ │ train_xgboost_ga.py │ │ generate_report.py   │
│                   │ │                     │ │                      │
│ Standard XGBoost  │ │ GA + LSTM/CNN +     │ │ HTML report:         │
│ + Optuna tuning   │ │ XGBoost (best)      │ │ - Verdict            │
│                   │ │                     │ │ - Equity curve       │
│ Output: models/   │ │ Output: models/     │ │ - Monte Carlo        │
└───────────────────┘ └─────────────────────┘ │ - Backtest stats     │
                                              │ Output: output/      │
                                              └──────────────────────┘
                                │
                                ▼
                ┌───────────────────────────┐
                │ current.py                │
                │                           │
                │ Enter current price →     │
                │ Get LONG/SHORT + TP/SL %  │
                │ + position sizing         │
                └───────────────────────────┘
```

## How It Works

### The Problem
Given today's market conditions, should I enter LONG or SHORT to maximize my chance of hitting TP before SL?

### The Data (fetch_stock_data.py)
- Fetches 9 years of daily OHLCV from Yahoo Finance
- Computes 170+ technical indicators (momentum, trend, volatility, volume, candlestick patterns)
- For each historical candle, looks forward 10 days and labels:
  - LONG: did price go UP to hit +1.5×ATR before going DOWN to hit -1×ATR?
  - SHORT: did price go DOWN to hit -1.5×ATR before going UP to hit +1×ATR?
- Outputs clean CSV with zero NaN values

### The Model (train_xgboost.py)
- XGBoost binary classifier: predicts LONG (1) vs SHORT (0)
- Only uses scale-invariant features (no absolute prices that change over time)
- Chronological train/test split (90/10) — never peeks at future
- Auto-detects NVIDIA GPU (`device='cuda'`)
- Optional Optuna Bayesian hyperparameter tuning (`--tune`)

### The GA Model (train_xgboost_ga.py) — Best Results
- **Genetic Algorithm** evolves both feature selection AND hyperparameters
- **LSTM + CNN** (PyTorch) extract learned temporal features from indicator sequences
- GA decides which of the 170+ features (traditional + deep) maximize walk-forward win rate
- Auto-detects GPU for both XGBoost (CUDA) and PyTorch (CUDA/MPS)
- Use `--no-deep` to skip LSTM/CNN if PyTorch has issues

### The Report (generate_report.py)
- Trains model, runs backtest, runs 10,000 Monte Carlo simulations
- Generates interactive HTML with Chart.js visualizations
- Shows: verdict, equity curve, MC probability distribution, feature importance

### Live Decision (current.py)
- Loads latest trained model
- Fetches recent market data for indicator computation
- Predicts direction from last completed candle's indicators
- Calculates TP/SL levels and % based on your current entry price + ATR
- Shows position sizing table

## Key Concepts

### R (Risk Unit)
R = dollars you risk per trade = (Entry - StopLoss) × shares.
- Win = +1.5R (TP hit)
- Loss = -1.0R (SL hit)
- Edge = (WinRate × 1.5) - (LossRate × 1.0)
- Break-even win rate at 1.5:1 R:R = 40%

### Expected Edge
```
Edge = WinRate × 1.5R - LossRate × 1.0R
51% WR → +0.27R/trade → +68R/year (252 trades)
55% WR → +0.37R/trade → +93R/year
```
If R = $100: +$6,800 to +$9,300 expected annual return.

### GPU Support
Both XGBoost and PyTorch auto-detect NVIDIA CUDA:
- XGBoost: `device='cuda'` speeds up tree building 2-4x
- PyTorch LSTM/CNN: trains on GPU automatically
- No manual flags needed — just have CUDA toolkit installed

## File Organization

```
fetch_stock_data.py          # Data pipeline
train_xgboost.py             # Standard XGBoost training
train_xgboost_ga.py          # GA + LSTM/CNN + XGBoost (best)
generate_report.py           # HTML report generator
current.py                   # Live trade decision

data/                        # CSV files (gitignored)
models/                      # Trained models (gitignored)
output/                      # HTML reports (gitignored)
```

All outputs use `{TICKER}_{YYYYMMDD}` prefix for versioning.

## Requirements

- Python 3.11.9
- yfinance, pandas, numpy, xgboost, scikit-learn, joblib, optuna, torch
