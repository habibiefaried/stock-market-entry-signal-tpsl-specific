# Stock Market Entry Signal — TP/SL Specific

Predicts whether to go **LONG or SHORT** on a given stock to maximize the probability of hitting Take Profit (1.5x ATR) before Stop Loss (1x ATR).

## Quick Start

```bash
pyenv local 3.11.9
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

# For GPU (Windows with NVIDIA):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# ── Single stock workflow ──────────────────────────────────────────────
# 1. Fetch 9 years of data
python fetch_stock_data.py --ticker AAPL

# 2. Train model — GPU auto-detected:
#    GPU found  →  250 Optuna trials + LSTM/CNN deep features ON
#    CPU only   →  100 Optuna trials + LSTM/CNN deep features OFF
python train.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv

# Override: force no deep (useful on GPU but want faster run)
python train.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv --no-deep

# Override: custom trial count
python train.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv --n-trials 200

# 3. Live trade decision (enter your current price)
python current.py --ticker AAPL --price 292.45

# 4. Generate HTML report (backtest + Monte Carlo)
python generate_report.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv

# ── Multi-stock ranking workflow ───────────────────────────────────────
# Edit target_stocks.txt with your tickers, then:
python ranking.py                          # train all, rank by confidence × win rate
python ranking.py --n-trials 200          # more Optuna trials (better tuning)
python ranking.py --deep-learning         # use LSTM+CNN experimental model

# ── Optional flags ─────────────────────────────────────────────────────
# ADX trending filter (not recommended as default — see experiments below)
python train.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv --min-adx-pctile 50
python ranking.py --min-adx-pctile 50

# Tighter TP/SL label window (default 10 days, try 5 for faster signals)
python fetch_stock_data.py --ticker AAPL --lookahead 5
python ranking.py --lookahead 5

# Deep learning model for current.py
python current.py --ticker AAPL --price 292.45 --deep
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
┌────────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│ train.py               │ │ generate_report.py   │ │ current.py           │
│                        │ │                      │ │                      │
│ XGBoost + Optuna       │ │ HTML report:         │ │ Live trade decision  │
│ --deep-learning for    │ │ - Verdict            │ │ - LONG/SHORT + TP/SL │
│ LSTM+CNN features      │ │ - Equity curve       │ │ - Position sizing    │
│ Output: models/        │ │ - Monte Carlo        │ │                      │
└────────────────────────┘ │ - Backtest stats     │ └──────────────────────┘
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

### Data Split Strategy — Full 9 Years Train, Last 3 Months Test

```
Full 9 years of data (~1446 clean samples):
├── 90% of pre-test data → TRAIN  (2018 – 2025, ~1258 samples)
├── 10% of pre-test data → VALID  (most recent pre-test, ~140 samples)
└── Last 3 months        → TEST   (Mar – Jun 2026, ~48 samples) ← live trading
```

**WHY FULL 9 YEARS FOR TRAINING:**
Tested: 3-year window (448 train samples) → 38% test WR (underfitting).
With 9 years (1258 train samples) → 84% test WR. The model needs volume
of data to learn robust patterns. Recency weighting was removed — XGBoost
with heavy regularization (gamma=3.8, lambda=7.7) naturally discounts
noisy old patterns while keeping broadly useful ones.

**WHY LAST 3 MONTHS FOR TEST:**
Directly answers the only question that matters for live trading:
"Would this model have made money in the last 3 months?"
If yes → trade it. If no → reject it regardless of historical performance.

**WHY 90/10 TRAIN/VALID (not 80/20):**
More training data = less underfitting. Validation set (10%) is only used
for early stopping and threshold tuning — doesn't need to be large.

**Example output (AAPL, 200 trials):**
```
Full dataset: 1446 samples (2018-10-16 → 2026-06-08)
Split (9yr train+valid, last 3mo test):
  Train: 1258 (2018-10-16 → 2025-05-23)
  Valid: 140 (2025-06-03 → 2026-03-06)
  Test:  48 (2026-03-09 → 2026-06-08) ← live trading period
  WR: 84.2% | Edge: +1.105R | Passed 55% WR gate
```

**Timeout-based training (find best model within time budget):**

Instead of a single Optuna run, the trainer loops for 5 minutes (configurable via `--timeout`):
1. Each loop runs a fresh Optuna study with different random seeds
2. Trains a model, evaluates test-set WR (backtest on last 3 months)
3. Keeps the best model (highest backtest WR) across all attempts
4. Only saves if new best WR > existing model on disk

```bash
python train.py --csv data/AAPL_*.csv              # 5 min default
python train.py --csv data/AAPL_*.csv --timeout 600  # 10 min for harder stocks
```

Example output:
```
Training (timeout 5min, finding best backtest WR)...
  Attempt 1: WR=49.1% | Trees=7479 LR=0.1486 Depth=8 [0s] ★ NEW BEST
  Attempt 2: WR=54.7% | Trees=6941 LR=0.0892 Depth=3 [32s] ★ NEW BEST
  Attempt 3: WR=54.7% | Trees=22669 LR=0.0862 Depth=6 [65s]
  ...
Best model found in 4 attempts (137s): WR=54.7%
```

WHY: Optuna's random exploration means each run finds different params.
Running multiple attempts within a timeout guarantees you find a good
model without manual re-running. For ranking.py with 20 stocks at 5min
each = 1.5 hours total (reasonable).

**PRIMARY OBJECTIVE: WIN RATE (TP hit before SL)**

Every part of the pipeline optimises for one thing — "does price hit TP before SL?"

| Stage | What it optimises | NOT optimising |
|-------|-------------------|----------------|
| Optuna walk-forward CV | Actual TP-hit win rate | Accuracy, logloss |
| Timeout retry loop | Highest test-set WR across attempts | Number of trials |
| Model save logic | Edge (derived from WR) vs existing model | Training loss |
| HTML report | Win Rate + Edge displayed | Accuracy/precision removed |
| current.py | TRADE if conf > threshold that maximised WR | - |

Accuracy (did model match TARGET label?) is misleading because a "wrong"
prediction can still hit TP (both directions sometimes win). Win Rate
directly measures: "if you follow this signal, do you make money?"

**Minimum WR requirement:**
- Default: **55% WR** (configurable via `--min-wr 0.55`)
- At 1.5:1 R:R, 55% WR = +0.475R edge — meaningfully profitable
- Below 55% = model rejected, not saved, user told to retrain
- Use `--force-save` to bypass the gate (e.g. for research purposes)

```bash
# Standard: rejects if < 55% WR in majority of last 3 years
python train.py --csv data/AAPL_*.csv

# Custom threshold: require 60% WR
python train.py --csv data/AAPL_*.csv --min-wr 0.60

# Override gate for research
python train.py --csv data/AAPL_*.csv --force-save
```

### The Model (train.py)
- XGBoost binary classifier: predicts LONG (1) vs SHORT (0)
- Only uses scale-invariant features (no absolute prices that change over time)
- **Binary prediction with noise filtering**: LONG (1) or SHORT (0). Drops samples where both directions hit SL (choppy noise).
- **Optuna-tuned confidence threshold**: The decision threshold is optimised on the validation set using Optuna. Scoring = `edge × sqrt(n_trades/20)` — penalises lucky streaks on small sample sizes (requires ≥20 trades). Saved to `models/{TICKER}_{date}_xgboost_threshold.txt`. Used automatically by `current.py`.
- **Trader-style backtest**: Simulates a real trader — only enters trades where confidence ≥ threshold. Reports full correlation table showing how win rate changes at each confidence level.
- **Market regime features**: ATR percentile, return percentile, volatility regime, trend strength regime (all 0-100 scale, tells model "what kind of market is this?")
- **Normalized features**: MACD_Hist/ATR, log-volume changes, CCI/KST clipped to bounded ranges — prevents scale drift over time
- **Interaction features**: 14 pre-computed pattern combinations (e.g., RSI_oversold + Bullish_Engulfing, BB_Squeeze + above SMA) — gives XGBoost direct access to multi-indicator signals
- **Calendar features**: day-of-week, month, quarter, is_monday, is_friday, month start/end — captures recurring temporal patterns
- Chronological train/test split (90/10) — never peeks at future
- Auto-detects NVIDIA GPU (`device='cuda'`)
- **Optuna Bayesian hyperparameter tuning is mandatory** — always runs, no manual params needed
- **MedianPruner** kills bottom-half trials at fold-level, saving ~30% tuning time
- **MedianPruner** skips unpromising trials early, saving time
- **Decision threshold optimization** finds the best LONG/SHORT boundary (not locked at 0.50)

### Experimental: LSTM+CNN Deep Features (`train_xgboost_cnn_lstm_experimental.py`)

An experimental variant that adds 4 learned features from a deep neural network:

| Component | Spec | Purpose |
|-----------|------|---------|
| LSTM | 1-layer hidden=8 → **2 output** | Regime encoding (bullish/bearish phase) |
| CNN | 1-layer conv 4 filters → pool → **2 output** | Pattern encoding (non-linear signals) |
| Input | 15-day × ~60 key indicators | Key oscillators/ratios only |
| Training | 20 epochs, CUDA, Adam, CrossEntropyLoss | ~30s on GPU |
| Total params | ~500 | Ultra-light, no overfitting risk |

**Total deep features: 4** (LSTM_0, LSTM_1, CNN_0, CNN_1) — minimal bottleneck compresses temporal patterns without overfitting.

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

#### Best Model Persistence

`train.py` compares the new model's test-set edge against the previously saved model. It only overwrites if the new run is better:

```
New run edge: +0.350R > Saved edge: +0.221R → Save new model ✔
New run edge: +0.172R < Saved edge: +0.221R → Keep saved model ✔
```

This means you can run `train.py` multiple times (e.g., to try more Optuna trials) and the best result is always preserved. `generate_report.py` and `current.py` always use the saved (best) model — they never retrain.

Score stored in `models/{TICKER}_{date}_xgboost_perf.txt`.
Use `--force-save` to overwrite regardless: `python train.py --csv ... --force-save`

#### Confidence Threshold — How It Works

The threshold is the minimum confidence required to enter a trade. Instead of guessing (e.g., always use 60%), we let Optuna find the sweet spot on the validation set:

```
Optuna searches threshold from 0.45 → 0.80
For each candidate threshold:
  → Simulate trading on validation set (skip any trade below threshold)
  → Measure edge = WinRate × 1.5R - LossRate × 1.0R
  → Score = edge (higher is better)
Best threshold saved to models/{ticker}_threshold.txt
```

**Example output — AAPL (after new indicators + fixed threshold):**
```
Correlation table on TEST set:
  Threshold  Trades   Skip%  Win Rate     Edge
  ─────────  ──────  ──────  ────────  ───────
       0.50     145    0.0%     61.4%  +0.534R   ← Optuna-tuned (≥20 trades enforced)
       0.52     108   25.5%     63.9%  +0.597R
       0.55      75   48.3%     66.7%  +0.667R   ← sweet spot
       0.57      60   58.6%     66.7%  +0.667R
       0.60      31   78.6%     51.6%  +0.290R
```

Key insight: the previous threshold overfitting (picking 0.62 with only 5 test trades) is now corrected.
The model correctly identifies 0.50 as having sufficient trades AND 61.4% WR.
```

Trade-off: higher threshold → fewer trades but higher win rate. The Optuna threshold maximises **edge** (quality × quantity), not just win rate.

`current.py` loads the saved threshold and shows whether today's signal is above it:
```
VERDICT:    ▼ SHORT
Confidence: 51.4%  →  ✘ SKIP (below threshold 0.62)
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
| `n_estimators` | 2000 – 30000 | Number of trees |
| `learning_rate` | 0.0001 – 0.5 | How much each tree contributes |
| `max_depth` | 3 – 8 | Complexity of each tree |
| `subsample` | 0.6 – 1.0 | Fraction of rows each tree sees |
| `colsample_bytree` | 0.5 – 1.0 | Fraction of features each tree sees |
| `min_child_weight` | 1 – 10 | Min samples required to split a node |
| `gamma` | 0 – 5 | Min gain required to make a split |
| `reg_alpha` | 0 – 10 | L1 regularization (sparsity) |
| `reg_lambda` | 0 – 10 | L2 regularization (shrinkage) |

**Usage:**
```bash
# Default: 100 trials (~5 min on CPU, ~1 min on GPU)
python train.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv

# More trials for better results (~10 min on CPU)
python train.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv --n-trials 200

# Trending-regime filter: only train on candles where ADX is in top N% of its 60d history
# Tested values: 0 (default, all data), 50 (top half), 66 (top third)
# Warning: cutting 50% of data hurts more than it helps on stocks with <2000 samples
python train.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv --min-adx-pctile 50
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

### Model Improvements: Options 1-3

#### Option 1: Contextual multi-candle interaction features
XGBoost sees each row independently with no sequential memory. Simply having `Three_Black_Crows = 1` tells it the pattern occurred — but not whether it happened after a huge volume day, or at an extreme RSI level. New contextual features encode compound signals:

| Feature | What it encodes |
|---------|----------------|
| `Bounce_After_3_Red` | Green candle after 3 consecutive red candles (the TSLA scenario) |
| `Bounce_After_5_Red` | Green candle after 5 consecutive red candles (deeper rebound) |
| `Failed_Bounce` | Green candle after drop, but close < prior high (weak relief rally) |
| `Drop_Depth_5d` | How far price fell in last 5 days, normalised by ATR (-10 to 0) |
| `Rally_Depth_5d` | How far price rose in last 5 days, normalised by ATR (0 to 10) |
| `Bullish/Bearish_Engulf_HighVol` | Engulfing pattern confirmed by volume spike (>1.5× avg) |
| `Three_Black/White_*_HighVol` | 3-candle patterns with volume confirmation |
| `Bullish/Bearish_Exhaustion` | Big candle after extended run = likely reversal |
| `Doji_In_Downtrend/Uptrend` | Doji within established trend = more meaningful indecision |
| `Inside_Bar_After_Big_Move` | Compression (inside bar) right after a large candle = breakout setup |

#### Option 2: Pattern strength as continuous value (not binary 0/1)
Previously `Three_Black_Crows = 1` whether the 3 candles fell 1% or 15%. Now each pattern includes a magnitude feature:

| Feature | Formula | Meaning |
|---------|---------|---------|
| `Three_Black_Crows_Magnitude` | sum(3 bodies) / ATR | Higher = stronger bearish signal |
| `Three_White_Soldiers_Magnitude` | sum(3 bodies) / ATR | Higher = stronger bullish signal |
| `Bullish/Bearish_Engulf_Magnitude` | engulf_body / prior_body | How much bigger the engulfing candle is |
| `Morning/Evening_Star_Strength` | reversal close distance / ATR | How far reversal candle penetrates |
| `Hammer/Shooting_Star_Strength` | wick_length / ATR | Longer wick = stronger rejection |
| `Bear/Bull_Run_Strength` | consecutive_candles × avg_body / ATR | Momentum of the run |

#### Option 3: Recency weighting
XGBoost treats AAPL data from 2017 equally to 2025 data. But stocks change behaviour over time (different products, different market regime, different volatility). We apply exponential decay weighting:

```
weight[i] = 0.9995^(n - 1 - i) × class_weight[i]

decay = 0.9995 → most recent sample = weight 1.0
                  sample from 1400 rows back ≈ half weight
                  sample from 2800 rows back ≈ quarter weight
```

Class balance weight is multiplied on top, so both recency AND class imbalance are handled simultaneously. Applied in both Optuna walk-forward CV and final model training.

### New Indicators Added (from TT library audit)

| Indicator | Formula summary | Why it helps |
|-----------|----------------|-------------|
| **Supertrend** | ATR-based dynamic band, +1/-1 trend signal | Widely proven trend-following; tells model if trend is established |
| **Supertrend_Dist** | (Close - active band) / Close % | How far price is from the support/resistance line |
| **STC (Schaff Trend Cycle)** | MACD through double Stochastic smoothing (0-100) | Faster and less noisy than MACD; >75 overbought, <25 oversold |
| **STC_Signal** | +1/>75, -1/<25, 0=neutral | Discrete overbought/oversold state |
| **RVI (Relative Vigor Index)** | body/range rolling average ratio | Measures buying vigor vs selling pressure |
| **RVI_Signal** | 4-period EMA of RVI | Smoothed version for crossover signals |
| **RVI_Cross** | sign(RVI - RVI_Signal) | +1 = bullish cross, -1 = bearish cross |
| **Ulcer_Index** | sqrt(mean(pct_drawdown²)) over 14d | Measures downside-only volatility; high = bearish stress |
| **Elder_Impulse** | EMA slope × MACD_Hist slope combined | +1 = both rising (green bar), -1 = both falling (red bar), 0 = mixed |
| **KVO (Klinger Volume Oscillator)** | Volume-weighted trend, EMA34-EMA55 | Better at volume divergences than OBV |
| **KVO_Signal** | 13-period MA of KVO | Trigger line |
| **RWI_High/Low** | (H-L[n]) / (ATR × √n) | >1 = trending stronger than a random walk |
| **RWI_Trend** | sign(RWI_High - RWI_Low) | +1 = uptrend confirmed, -1 = downtrend confirmed |
| **RVI_Vol (Relative Volatility Index)** | RSI applied to std deviation | High = volatile days dominate; helps regime detection |

### The Report (generate_report.py)
- Uses same 3-way split as `train.py` (80% train / 10% valid / 10% test, no leakage)
- Optuna tunes hyperparameters on train, confidence threshold on valid
- Backtest runs at Optuna-tuned threshold (trader-style: skip signals below threshold)
- Runs 10,000 Monte Carlo simulations on filtered trades
- HTML verdict card shows ✔ TRADE or ✘ SKIP based on threshold
- Shows: threshold-aware verdict, equity curve, MC, feature importance

### Live Decision (current.py)
**Never retrains.** Loads saved model artifacts from `models/`. If no model exists for the ticker, it exits with a clear error message and tells you to run `train.py` first.

Loads the saved confidence threshold from `models/{TICKER}_*_threshold.txt` (written by `train.py` / `generate_report.py`) and shows whether today's signal clears it:
```
VERDICT:    ▲ LONG
Confidence: 52.3%  →  ✔ TRADE
Threshold:  52% (Optuna-tuned on validation set)
```
or:
```
VERDICT:    ▼ SHORT
Confidence: 48.1%  →  ✘ SKIP (below threshold 0.62)
```

### Live Decision (current.py) — Details

**Why you only need to supply a current price:**

The model was trained on *completed* daily candles — Open, High, Low, Close are all known. When you enter the traderoom, today's candle is still forming so it can't be used as a feature. Instead `current.py` does this:

```
Step 1: Load model
        ↓ reads models/{TICKER}_{date}_xgboost_model.json
        ↓ reads matching _scaler.pkl and _features.txt

Step 2: Fetch recent historical data (14 months)
        ↓ Yahoo Finance → OHLCV for last 14 months
        ↓ Compute 170+ technical indicators
        ↓ Run same load_and_prepare() pipeline as training
          (lag features, regime percentiles, interactions, calendar)
        ↓ The LAST COMPLETED candle is the most recent full trading day

Step 3: Predict direction using last completed candle's indicators
        ↓ model.predict_proba(last_candle_features) → [P(SHORT), P(LONG)]
        ↓ direction = argmax(probabilities)
        ↓ confidence = max probability
        Note: the prediction is "if I enter NOW, should I go LONG or SHORT?"
              It uses yesterday's completed indicators as signal

Step 4: Calculate TP/SL from YOUR current price (not last close)
        ↓ ATR comes from last completed candle (the most recent valid ATR)
        ↓ LONG: TP = current_price + 1.5×ATR, SL = current_price - 1.0×ATR
        ↓ SHORT: TP = current_price - 1.5×ATR, SL = current_price + 1.0×ATR
        Note: we use YOUR price for levels because that's your actual entry,
              not yesterday's close price

Step 5: Position sizing table
        ↓ risk_per_share = |current_price - sl_price|
        ↓ shares = risk_budget / risk_per_share
        ↓ printed for 1%, 2%, 3%, 5% of $10,000 capital
```

**Example:**
```
Last completed candle: 2026-06-09 | Close: $290.55
ATR(14): $7.51
↓ model sees June 9's indicators → predicts SHORT (58% confidence)

You enter: current price = $292.45
↓ SHORT TP = 292.45 - 1.5×7.51 = $281.19  (-3.8%)
↓ SHORT SL = 292.45 + 1.0×7.51 = $299.96  (+2.6%)
```

The key insight: **indicators tell you the direction, your current price determines the exact dollar levels.**

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

### GPU Auto-Detection

When NVIDIA CUDA is detected, the pipeline automatically upgrades:

| Setting | CPU default | GPU auto |
|---------|------------|---------|
| Optuna trials | 200 | **400** |
| LSTM/CNN deep features | OFF | **ON** |
| XGBoost device | cpu | **cuda** |

No manual flags needed — just have CUDA + PyTorch installed. To override:
```bash
python train.py --csv data/AAPL_*.csv --no-deep       # GPU but no deep
python train.py --csv data/AAPL_*.csv --n-trials 100  # GPU but fewer trials
```

## File Organization

```
fetch_stock_data.py                        # Data pipeline
train.py                           # XGBoost + Optuna (--deep-learning for LSTM+CNN)
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
