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

# 2. Train model (Optuna runs automatically, 100 trials by default)
python train_xgboost.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv

# 2b. More Optuna trials for better tuning
python train_xgboost.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv --n-trials 200

# 3. Generate HTML report with backtest + Monte Carlo (Optuna runs inside)
python generate_report.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv

# 4. Live trade decision
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
┌───────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│ train_xgboost.py  │ │ generate_report.py   │ │ current.py           │
│                   │ │                      │ │                      │
│ XGBoost            │ │ HTML report:         │ │ Live trade decision  │
│ + Optuna tuning   │ │ - Verdict            │ │ - LONG/SHORT + TP/SL │
│                   │ │ - Equity curve       │ │ - Position sizing    │
│ Output: models/   │ │ - Monte Carlo        │ │                      │
└───────────────────┘ │ - Backtest stats     │ └──────────────────────┘
                      │ Output: output/      │
                      └──────────────────────┘

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
- **Optuna Bayesian hyperparameter tuning is mandatory** — always runs, no manual params needed
- **MedianPruner** skips unpromising trials early, saving time
- **Decision threshold optimization** finds the best LONG/SHORT boundary (not locked at 0.50)

#### What is Optuna?
Optuna automatically finds the best hyperparameters by running many trials and learning from each result. It uses Bayesian optimization: each trial is informed by all previous trials, so it narrows in on good values faster than random or grid search. The MedianPruner stops trials that are in the bottom half at each step, allowing more effective trials in less time.

Each trial trains XGBoost with a different combination of parameters and scores it using walk-forward win rate. After training, the optimal decision threshold is found by testing thresholds from 0.30 to 0.70 and picking the one that maximizes actual TP hit rate.

**Parameters Optuna tunes (you never set these manually):**

| Parameter | Search Range | What It Controls |
|-----------|-------------|-----------------|
| `n_estimators` | 500 – 5000 | Number of trees |
| `learning_rate` | 0.005 – 0.1 | How much each tree contributes |
| `max_depth` | 3 – 10 | Complexity of each tree |
| `subsample` | 0.6 – 1.0 | Fraction of rows each tree sees |
| `colsample_bytree` | 0.5 – 1.0 | Fraction of features each tree sees |
| `min_child_weight` | 1 – 10 | Min samples required to split a node |
| `gamma` | 0 – 5 | Min gain required to make a split |
| `reg_alpha` | 0 – 10 | L1 regularization (sparsity) |
| `reg_lambda` | 0 – 10 | L2 regularization (shrinkage) |

**Usage:**
```bash
# Default: 100 trials (~5 min on CPU, ~1 min on GPU)
python train_xgboost.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv

# More trials for better results (~10 min on CPU)
python train_xgboost.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv --n-trials 200
```

### The Report (generate_report.py)
- Trains model with Optuna (same mandatory tuning as `train_xgboost.py`)
- Runs backtest on test set, then runs 10,000 Monte Carlo simulations
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
XGBoost auto-detects NVIDIA CUDA via PyTorch:
- XGBoost: `device='cuda'` speeds up tree building 2-4x
- No manual flags needed — just have CUDA toolkit installed

## File Organization

```
fetch_stock_data.py          # Data pipeline
train_xgboost.py             # XGBoost + Optuna training
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
