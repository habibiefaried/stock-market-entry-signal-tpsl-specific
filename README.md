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
- **Binary prediction with noise filtering**: LONG (1) or SHORT (0). Drops samples where both directions hit SL (choppy noise). Use confidence threshold at inference to skip uncertain signals.
- **Market regime features**: ATR percentile, return percentile, volatility regime, trend strength regime (all 0-100 scale, tells model "what kind of market is this?")
- **Normalized features**: MACD_Hist/ATR, log-volume changes, CCI/KST clipped to bounded ranges — prevents scale drift over time
- **Interaction features**: 14 pre-computed pattern combinations (e.g., RSI_oversold + Bullish_Engulfing, BB_Squeeze + above SMA) — gives XGBoost direct access to multi-indicator signals
- **Calendar features**: day-of-week, month, quarter, is_monday, is_friday, month start/end — captures recurring temporal patterns
- Chronological train/test split (90/10) — never peeks at future
- Auto-detects NVIDIA GPU (`device='cuda'`)
- **Optuna Bayesian hyperparameter tuning is mandatory** — always runs, no manual params needed
- **XGBoostPruningCallback** kills bad trials mid-training (per-tree level)
- **MedianPruner** skips unpromising trials early, saving time
- **Decision threshold optimization** finds the best LONG/SHORT boundary (not locked at 0.50)

### Experimental: LSTM+CNN Deep Features (`train_xgboost_cnn_lstm_experimental.py`)

An experimental variant that adds 4 learned features from a deep neural network:

| Component | Spec | Purpose |
|-----------|------|---------|
| LSTM | 2-layer 64-hidden → fc32 → fc8 → **2 output** | Regime encoding (bullish/bearish phase) |
| CNN | 3-layer conv 64→32→16 → fc8 → **2 output** | Pattern encoding (non-linear signals) |
| Input | 15-day × ~60 key indicators | Key oscillators/ratios only |
| Training | 30 epochs, CUDA, Adam, CrossEntropyLoss | ~2 min on GPU |
| XGBoost | Same config as `train_xgboost.py` | Optuna + XGBoostPruningCallback, 500-15000 trees, lr 0.0005-0.2 |

**Total deep features: 4** (LSTM_0, LSTM_1, CNN_0, CNN_1) — deeper network compresses temporal patterns into a tight bottleneck rather than passing through 24 noisy activations.

**Why it underperforms in backtest/Monte Carlo despite higher Optuna WR:**

1. **Overfitting to walk-forward signal**: The LSTM+CNN is trained on the *same* data that Optuna evaluates on. It learns to produce features that score well in walk-forward CV, but these features don't generalize to the held-out test set. This is a form of **information leakage** — the neural net sees the full sequence distribution during training.

2. **Feature dominance / co-adaptation**: LSTM features dominate the top-20 importance (often 10-15 of top 20 slots). This creates co-adaptation between the deep features and XGBoost, where the model relies heavily on a few high-signal deep features that may be noise on unseen data.

3. **Small dataset (1995 samples)**: Neural networks typically need much more data than tree-based models. With only ~2000 samples, the LSTM+CNN cannot learn robust representations, leading to high-variance features that hurt test performance.

4. **Sequence trimming**: The first 15 rows are dropped (seq_len), losing 15 data points — negligible for training but symbolically the model sees slightly less history.

**The Optuna WR paradox**: Walk-forward CV uses overlapping windows within the training set. The deep features, having been trained on this distribution, score artificially high in CV but fail on the true chronological test split. This is classic **train/test mismatch** for time series.

**Usage:**
```bash
# Train standalone
python train_xgboost_cnn_lstm_experimental.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv

# Use in report (instead of standard XGBoost)
python generate_report.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv --deep-experimental
```

#### How Confidence Is Calculated

XGBoost outputs a probability for each class via `predict_proba()`. For binary classification:
- `P(LONG)` = probability the model assigns to class LONG
- `P(SHORT)` = 1 - P(LONG)
- `Confidence` = whichever is higher (the predicted class's probability)

Example:
```
predict_proba() → [0.38, 0.62]
                    ↑        ↑
                P(SHORT)  P(LONG)

Prediction: LONG (higher probability)
Confidence: 62%
```

Internally, XGBoost applies sigmoid to its raw output (log-odds):
```
raw_score = sum of all tree leaf values for this sample
P(LONG) = 1 / (1 + exp(-raw_score))
P(SHORT) = 1 - P(LONG)
```

**Why confidence matters for trading:**

| Confidence | Meaning | Action |
|-----------|---------|--------|
| 50-55% | Barely above coin flip | Skip or reduce size |
| 55-60% | Moderate signal | Trade with normal size |
| 60-70% | Strong signal | Trade with confidence |
| 70%+ | Very strong signal | Maximum conviction |

The confidence analysis section in training output shows win rate at each threshold. Example from AAPL:
```
Confidence >= 50%: 145 trades, Win Rate: 63.4%
Confidence >= 55%: 112 trades, Win Rate: 67.0%
Confidence >= 60%:  91 trades, Win Rate: 72.5%
Confidence >= 70%:  50 trades, Win Rate: 76.0%
```

Higher confidence = fewer trades but higher win rate. You choose your tradeoff.

#### What is Optuna?
Optuna automatically finds the best hyperparameters by running many trials and learning from each result. It uses Bayesian optimization: each trial is informed by all previous trials, so it narrows in on good values faster than random or grid search. The MedianPruner stops trials that are in the bottom half at each step, allowing more effective trials in less time.

Each trial trains XGBoost with a different combination of parameters and scores it using walk-forward win rate. After training, the optimal decision threshold is found by testing thresholds from 0.30 to 0.70 and picking the one that maximizes actual TP hit rate.

**Parameters Optuna tunes (you never set these manually):**

| Parameter | Search Range | What It Controls |
|-----------|-------------|-----------------|
| `n_estimators` | 500 – 15000 | Number of trees |
| `learning_rate` | 0.0005 – 0.2 | How much each tree contributes |
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

# Trending-regime filter: only train on candles where ADX is in top N% of its 60d history
# Tested values: 0 (default, all data), 50 (top half), 66 (top third)
# Warning: cutting 50% of data hurts more than it helps on stocks with <2000 samples
python train_xgboost.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv --min-adx-pctile 50
```

#### Validated Experiments: What Actually Improves Win Rate

Tested on AAPL, NFLX, AMD (9 years data, 100 Optuna trials each):

| Config | AAPL WR | NFLX WR | AMD WR | Verdict |
|--------|---------|---------|--------|---------|
| Baseline (lh=10, no filter) | **60.0%** | 51.4% | **50.3%** | Best overall |
| Lookahead=5, no filter | 59.3% | **55.0%** | 37.4% | Worse on AMD |
| Lookahead=10, ADX>=50 | 59.5% | 44.0% | 50.0% | Hurts NFLX |
| Lookahead=5, ADX>=50 | 53.7% | 47.2% | 45.6% | Worst overall |

**Conclusion:** The ADX regime filter removes ~50% of training data (500-700 samples), which hurts more than the signal quality gain. The baseline (10-day lookahead, all samples) remains optimal. Confidence of 60-70%+ is achievable through the confidence threshold analysis at inference time — only trade signals where model confidence ≥ 60%.

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
fetch_stock_data.py                        # Data pipeline
train_xgboost.py                           # XGBoost + Optuna training
train_xgboost_cnn_lstm_experimental.py     # Experimental: LSTM+CNN deep features
generate_report.py                         # HTML report generator
current.py                                 # Live trade decision

data/                        # CSV files (gitignored)
models/                      # Trained models (gitignored)
output/                      # HTML reports (gitignored)
```

All outputs use `{TICKER}_{YYYYMMDD}` prefix for versioning.

## Requirements

- Python 3.11.9
- yfinance, pandas, numpy, xgboost, scikit-learn, joblib, optuna, torch
