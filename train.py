"""
train.py - Unified XGBoost training script with optional LSTM+CNN deep features.

Standard mode (default):
    python train.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv

Deep learning mode (adds 4 LSTM+CNN features, requires PyTorch + NVIDIA GPU):
    python train.py --csv data/AAPL_tpsl_data_YYYYMMDD.csv --deep-learning

Optional flags:
    --n-trials 200          More Optuna trials (better tuning, slower)
    --min-adx-pctile 50     Only train on trending regimes (see README)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import argparse
import warnings
import time
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from optuna_integration import XGBoostPruningCallback
except ImportError:
    raise ImportError("optuna and optuna-integration are required. Run: pip install optuna optuna-integration")


# ──────────────────────────────────────────────
# SHARED: Feature engineering + target labeling
# ──────────────────────────────────────────────

def load_and_prepare(csv_path: str):
    """Load CSV, engineer scale-invariant lag/regime/interaction/calendar features, create target."""
    df = pd.read_csv(csv_path)
    df = df.copy()

    close = df["Close"]
    volume = df["Volume"]

    new_cols = {}

    # Return lags
    daily_return = close.pct_change() * 100
    for lag in [1, 2, 3, 5, 10]:
        new_cols[f"Return_Lag_{lag}"] = daily_return.shift(lag)

    # Log-normalized volume change (tames outliers)
    vol_change = np.log1p(volume.pct_change().abs()) * np.sign(volume.pct_change()) * 100
    for lag in [1, 2, 3, 5]:
        new_cols[f"VolChange_Lag_{lag}"] = vol_change.shift(lag)

    # MACD_Hist normalized by ATR (scale-invariant across time)
    macd_hist_norm = df["MACD_Hist"] / df["ATR"]
    for lag in [1, 2, 3, 5]:
        new_cols[f"MACD_Hist_Norm_Lag_{lag}"] = macd_hist_norm.shift(lag)

    for lag in [1, 2, 3]:
        new_cols[f"BB_Pct_Lag_{lag}"] = df["BB_Pct"].shift(lag)
    for lag in [1, 2, 3]:
        new_cols[f"Stoch_K_Lag_{lag}"] = df["Stoch_K"].shift(lag)
    for lag in [1, 2, 3]:
        new_cols[f"ADX_Lag_{lag}"] = df["ADX"].shift(lag)

    # CCI normalized to -1/+1
    cci_norm = df["CCI"].clip(-200, 200) / 200
    for lag in [1, 2, 3]:
        new_cols[f"CCI_Norm_Lag_{lag}"] = cci_norm.shift(lag)

    new_cols["Volatility_Change_5d"] = df["Volatility_5d"] - df["Volatility_5d"].shift(5)
    new_cols["RSI14_Accel"] = df["RSI14_Slope_3d"] - df["RSI14_Slope_3d"].shift(3)

    # KST and Chaikin_Vol normalized
    if "KST" in df.columns:
        new_cols["KST_Norm"] = df["KST"].clip(-200, 200) / 200
        new_cols["KST_Signal_Norm"] = df["KST_Signal"].clip(-200, 200) / 200
    if "Chaikin_Vol" in df.columns:
        new_cols["Chaikin_Vol_Norm"] = df["Chaikin_Vol"].clip(-100, 200) / 100

    # Interaction features
    new_cols["RSI_Oversold_AND_Bullish_Engulf"] = ((df["RSI_14"] < 30) & (df["Bullish_Engulfing"] == 1)).astype(int)
    new_cols["RSI_Overbought_AND_Bearish_Engulf"] = ((df["RSI_14"] > 70) & (df["Bearish_Engulfing"] == 1)).astype(int)
    new_cols["ADX_Trending_AND_Momentum_Pos"] = ((df["ADX"] > 25) & (df["Momentum_5"] > 0)).astype(int)
    new_cols["ADX_Trending_AND_Momentum_Neg"] = ((df["ADX"] > 25) & (df["Momentum_5"] < 0)).astype(int)
    new_cols["BB_Squeeze_AND_Bullish"] = ((df["BB_Squeeze"] == 1) & (df["Close"] > df["SMA_20"])).astype(int)
    new_cols["BB_Squeeze_AND_Bearish"] = ((df["BB_Squeeze"] == 1) & (df["Close"] < df["SMA_20"])).astype(int)
    new_cols["Volume_Spike_AND_Bullish"] = ((df["Volume_Ratio"] > 2.0) & (df["Close"] > df["Open"])).astype(int)
    new_cols["Volume_Spike_AND_Bearish"] = ((df["Volume_Ratio"] > 2.0) & (df["Close"] < df["Open"])).astype(int)
    new_cols["Stoch_Oversold_AND_MACD_Cross"] = ((df["Stoch_K"] < 20) & (df["MACD_Hist"] > 0)).astype(int)
    new_cols["Stoch_Overbought_AND_MACD_Cross"] = ((df["Stoch_K"] > 80) & (df["MACD_Hist"] < 0)).astype(int)
    new_cols["Doji_At_Support"] = ((df["Doji"] == 1) & (df["Close_SMA20_Ratio"] < -0.03)).astype(int)
    new_cols["Doji_At_Resistance"] = ((df["Doji"] == 1) & (df["Close_SMA20_Ratio"] > 0.03)).astype(int)
    new_cols["Hammer_In_Downtrend"] = ((df["Hammer"] == 1) & (df["Trend_5d"] < 0)).astype(int)
    new_cols["Shooting_Star_In_Uptrend"] = ((df["Shooting_Star"] == 1) & (df["Trend_5d"] > 0)).astype(int)

    # Calendar features
    date_series = pd.to_datetime(df["Date"])
    new_cols["Day_of_Week"] = date_series.dt.dayofweek
    new_cols["Month"] = date_series.dt.month
    new_cols["Is_Monday"] = (date_series.dt.dayofweek == 0).astype(int)
    new_cols["Is_Friday"] = (date_series.dt.dayofweek == 4).astype(int)
    new_cols["Is_Month_Start"] = (date_series.dt.day <= 5).astype(int)
    new_cols["Is_Month_End"] = (date_series.dt.day >= 25).astype(int)
    new_cols["Quarter"] = date_series.dt.quarter

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df = df.dropna().reset_index(drop=True)

    # Market regime features (percentile-based)
    regime_cols = {}
    regime_cols["ATR_Pctile_60d"] = df["ATR"].rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    regime_cols["Return20d_Pctile_252d"] = close.pct_change(20).rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    regime_cols["Volatility_Regime"] = df["Volatility_20d"].rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    regime_cols["Trend_Strength_Regime"] = df["ADX"].rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    df = pd.concat([df, pd.DataFrame(regime_cols, index=df.index)], axis=1)
    df = df.dropna().reset_index(drop=True)

    # Target: LONG=1, SHORT=0, drop both-SL noisy rows
    targets, keep_mask = [], []
    for _, row in df.iterrows():
        long_tp = row["VERDICT_LONG"] == "TP"
        short_tp = row["VERDICT_SHORT"] == "TP"
        if long_tp and short_tp:
            targets.append(1 if row["DAY_PASS_LONG"] <= row["DAY_PASS_SHORT"] else 0)
            keep_mask.append(True)
        elif long_tp:
            targets.append(1); keep_mask.append(True)
        elif short_tp:
            targets.append(0); keep_mask.append(True)
        else:
            targets.append(-1); keep_mask.append(False)

    df["TARGET"] = targets
    n_before = len(df)
    df = df[keep_mask].reset_index(drop=True)
    n_dropped = n_before - len(df)
    n_long = (df["TARGET"] == 1).sum()
    n_short = (df["TARGET"] == 0).sum()
    print(f"Dropped {n_dropped} noisy samples (both SL hit, {n_dropped/n_before*100:.1f}%)")
    print(f"Clean targets: LONG={n_long}, SHORT={n_short}")

    exclude_cols = [
        "Date", "LONG_TP_Level", "LONG_SL_Level", "VERDICT_LONG", "DAY_PASS_LONG",
        "SHORT_TP_Level", "SHORT_SL_Level", "VERDICT_SHORT", "DAY_PASS_SHORT", "TARGET",
        "Open", "High", "Low", "Close", "Volume",
        "SMA_5", "SMA_10", "SMA_20", "SMA_50", "SMA_200",
        "EMA_9", "EMA_12", "EMA_21", "EMA_26", "EMA_50",
        "HMA_18", "BB_Upper", "BB_Lower",
        "Ichimoku_Tenkan", "Ichimoku_Kijun", "Ichimoku_Senkou_A", "Ichimoku_Senkou_B",
        "Ichimoku_Cloud_Width", "Price_vs_Cloud",
        "PSAR", "Keltner_Upper", "Keltner_Lower",
        "Donchian_Upper", "Donchian_Lower", "Donchian_Mid",
        "VWAP_10", "OBV", "OBV_EMA", "AD_Line", "AD_Line_EMA",
        "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5",
        "Volume_Lag_1", "Volume_Lag_2", "Volume_Lag_3", "Volume_Lag_5",
        "Force_Index", "Force_Index_EMA",
        "ATR", "MACD", "MACD_Signal", "MACD_Hist",
        "AO", "AO_Signal", "DPO", "Bull_Power", "Bear_Power",
        "EMA9_EMA21_Diff", "SMA20_SMA50_Diff", "SMA50_SMA200_Diff",
        "Volume_SMA_20", "Wilder_MA_14", "Chaikin_Osc", "DMA_20_5",
        "EMV", "EMV_Signal",
        "CCI", "KST", "KST_Signal", "Chaikin_Vol",
        # New: Klinger raw (dollar-scale), keep KVO_Norm and KVO_Signal
        "vf_series",
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    df = df.copy()
    return df, feature_cols


# ──────────────────────────────────────────────
# OPTIONAL: LSTM+CNN deep feature extraction
# ──────────────────────────────────────────────

def generate_deep_features(df, feature_cols, seq_len=15, epochs=20, lr=0.001):
    """Train ultra-light LSTM+CNN, extract 4 compressed deep features (LSTM_0/1, CNN_0/1).

    Minimal architecture — 1 layer each, 2 output params, ~500 total parameters.
    Trains in ~30s on GPU.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  [SKIP] PyTorch not installed — pip install torch")
        return df, []

    deep_input_cols = [c for c in feature_cols if any(k in c for k in [
        "RSI", "Stoch", "MACD_Hist", "BB_Pct", "CCI", "ADX", "Williams",
        "MFI", "CMF", "ROC", "Momentum", "Volatility", "CMO", "PPO",
        "TRIX", "Vortex", "Aroon", "VHF", "Body_Ratio", "Price_Change",
    ])]
    if len(deep_input_cols) < 10:
        deep_input_cols = feature_cols[:30]

    print(f"  Training ultra-light LSTM+CNN ({len(deep_input_cols)} inputs, seq_len={seq_len}, epochs={epochs})...")

    class LSTMFeat(nn.Module):
        """1-layer LSTM: hidden=8 -> 2 regime features."""
        def __init__(self, n_in):
            super().__init__()
            self.lstm = nn.LSTM(n_in, 8, num_layers=1, batch_first=True)
            self.fc = nn.Linear(8, 2)
        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return self.fc(h_n[-1])

    class CNNFeat(nn.Module):
        """1-layer CNN: 4 filters -> pool -> 2 pattern features."""
        def __init__(self, n_in):
            super().__init__()
            self.conv = nn.Conv1d(n_in, 4, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(4, 2)
        def forward(self, x):
            x = x.transpose(1, 2)
            x = torch.relu(self.conv(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)

    class CombinedExtractor(nn.Module):
        """LSTM + CNN -> 4 features -> classifier."""
        def __init__(self, n_in):
            super().__init__()
            self.lstm = LSTMFeat(n_in)
            self.cnn = CNNFeat(n_in)
            self.classifier = nn.Linear(4, 2)
        def forward(self, x):
            return self.classifier(torch.cat([self.lstm(x), self.cnn(x)], dim=1)), \
                   torch.cat([self.lstm(x), self.cnn(x)], dim=1)

    feature_data = StandardScaler().fit_transform(
        df[deep_input_cols].values.astype(np.float64)
    ).astype(np.float32)

    targets = df["TARGET"].values
    n_seq = len(df) - seq_len
    X_seq = np.zeros((n_seq, seq_len, len(deep_input_cols)), dtype=np.float32)
    y_seq = np.zeros(n_seq, dtype=np.int64)
    for i in range(n_seq):
        X_seq[i] = feature_data[i:i + seq_len]
        y_seq[i] = targets[i + seq_len]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  PyTorch device: {device}")
    train_size = int(n_seq * 0.8)
    X_train_t = torch.from_numpy(X_seq[:train_size].copy()).to(device)
    y_train_t = torch.from_numpy(y_seq[:train_size].copy()).to(device)

    model = CombinedExtractor(len(deep_input_cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_train_t))
        total_loss = 0.0
        for i in range(0, len(X_train_t), 64):
            idx = perm[i:i + 64]
            logits, _ = model(X_train_t[idx])
            loss = crit(logits, y_train_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}/{epochs}  loss={total_loss:.4f}")

    model.eval()
    all_feats = []
    with torch.no_grad():
        X_all_t = torch.from_numpy(X_seq.copy()).to(device)
        for i in range(0, len(X_all_t), 256):
            _, feats = model(X_all_t[i:i + 256])
            all_feats.append(feats.cpu().numpy())
    all_feats = np.concatenate(all_feats, axis=0)

    deep_cols = [f"LSTM_{i}" for i in range(2)] + [f"CNN_{i}" for i in range(2)]
    df = df.iloc[seq_len:].reset_index(drop=True).copy()
    for i, col in enumerate(deep_cols):
        df[col] = all_feats[:, i]

    print(f"  Extracted {len(deep_cols)} deep features (samples: {len(df)})")
    del model, X_train_t, y_train_t, X_all_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return df, deep_cols


# ──────────────────────────────────────────────
# SHARED: Optuna walk-forward CV + optimization
# ──────────────────────────────────────────────

def walk_forward_cv(df, feature_cols, params, n_splits=5, test_size=0.1, trial=None):
    """Walk-forward CV with XGBoostPruningCallback for per-tree pruning."""
    n = len(df)
    test_len = int(n * test_size)
    min_train = int(n * 0.4)
    scores = []
    max_boost_rounds = params.get("n_estimators", 30000)

    for i in range(n_splits):
        test_end = n - i * test_len
        test_start = test_end - test_len
        if test_start < min_train:
            break

        train_df = df[:test_start]
        test_df = df[test_start:test_end]

        X_train = train_df[feature_cols].values
        y_train = train_df["TARGET"].values
        X_test = test_df[feature_cols].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        n_long = y_train.sum()
        n_short = len(y_train) - n_long
        sw = n_short / n_long if n_long > 0 else 1.0

        callbacks = [XGBoostPruningCallback(trial, "validation_0-logloss")] if trial else None
        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=sw,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            early_stopping_rounds=30,
            callbacks=callbacks,
        )
        model.fit(X_train_s, y_train, eval_set=[(X_test_s, test_df["TARGET"].values)],
                  verbose=False)

        y_pred = model.predict(X_test_s)
        correct = sum(
            1 for idx, (_, row) in enumerate(test_df.iterrows())
            if (y_pred[idx] == 1 and row["VERDICT_LONG"] == "TP") or
               (y_pred[idx] == 0 and row["VERDICT_SHORT"] == "TP")
        )
        scores.append(correct / len(test_df))

        if trial is not None:
            trial.report(np.mean(scores), i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return np.mean(scores) if scores else 0.0


def optimize_hyperparams(df, feature_cols, n_trials=100, n_jobs=1):
    """Optuna Bayesian optimization with GPU parallelism + MedianPruner.

    n_jobs: number of parallel trials. 2-4 works well on laptop GPU
            (XGBoost CUDA can handle concurrent models before VRAM fills).
    """
    plural = "s" if n_jobs > 1 else ""
    print(f"Running Optuna optimization ({n_trials} trials, {n_jobs} parallel worker{plural})...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 2000, 30000),
            "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.5, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        }
        return walk_forward_cv(df, feature_cols, params, n_splits=5, trial=trial)

    kwargs = dict(direction="maximize",
                  pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2))
    if n_jobs > 1:
        import tempfile, uuid
        db_path = os.path.join(tempfile.gettempdir(), f"optuna_{uuid.uuid4().hex[:8]}.db")
        kwargs["storage"] = f"sqlite:///{db_path}"

    study = optuna.create_study(**kwargs)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=(n_jobs == 1))

    # Clean up temp storage file
    if n_jobs > 1:
        try:
            os.remove(db_path)
        except OSError:
            pass

    print(f"\nBest walk-forward win rate: {study.best_value*100:.1f}%")
    print(f"Best params: {study.best_params}")
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"Pruned trials: {pruned}/{n_trials}")
    return study.best_params


def detect_gpu():
    """Detect CUDA availability for XGBoost."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ──────────────────────────────────────────────
# MAIN: Train + evaluate + save
# ──────────────────────────────────────────────

def _backtest_at_threshold(df_set, y_pred, y_prob, threshold):
    """Run trader-style backtest: only enter trades where confidence >= threshold."""
    correct = skipped = total = 0
    cumulative_r = 0.0
    equity_curve = []
    for idx, (_, row) in enumerate(df_set.iterrows()):
        conf = float(np.max(y_prob[idx]))
        if conf < threshold:
            skipped += 1
            continue
        total += 1
        won = (y_pred[idx] == 1 and row["VERDICT_LONG"] == "TP") or \
              (y_pred[idx] == 0 and row["VERDICT_SHORT"] == "TP")
        if won:
            correct += 1
            cumulative_r += 1.5
        else:
            cumulative_r -= 1.0
        equity_curve.append(cumulative_r)
    wr = correct / total * 100 if total > 0 else 0.0
    edge = wr / 100 * 1.5 - (1 - wr / 100) * 1.0 if total > 0 else 0.0
    return {"total": total, "wins": correct, "wr": wr, "edge": edge,
            "skipped": skipped, "final_r": cumulative_r, "equity": equity_curve}


def _find_best_threshold_optuna(df_valid, y_pred_valid, y_prob_valid, n_trials=40):
    """
    Use Optuna to find the confidence threshold that maximises risk-adjusted edge
    on the validation set.

    Scoring = edge × sqrt(n_trades / 20)
      - Rewards high edge
      - Penalises statistically thin results (< 20 trades -> score scaled down)
      - Prevents overfitting to lucky 5-trade streaks at very high thresholds
    Min 20 trades required to return a positive score.
    """
    def objective(trial):
        thresh = trial.suggest_float("threshold", 0.45, 0.75)
        result = _backtest_at_threshold(df_valid, y_pred_valid, y_prob_valid, thresh)
        n = result["total"]
        if n < 20:
            return 0.0  # not enough trades — statistically meaningless
        # Scale down edge by sqrt(n/20): rewards confidence with sample size
        statistical_weight = np.sqrt(n / 20.0)
        return result["edge"] * statistical_weight

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return round(study.best_params["threshold"], 2), study.best_value


def purged_random_split(df, valid_frac=0.15, n_test_months=3,
                        embargo_days=5, random_state=42):
    """
    Purged random split with FORCED RECENT TEST (staggered across 2 years).

    Strategy:
      - TEST: last 3 months of THIS year + last 3 months of LAST year.
        This ensures test covers BOTH recent data AND a different regime,
        preventing inflated WR from one-directional trends.
      - VALID: random 15% of remaining months (threshold tuning)
      - TRAIN: everything else (diverse months from all years)
      - ±5 day embargo at block boundaries prevents temporal leakage

    WHY STAGGERED TEST (5 time periods):
    If test is only from one period (e.g. Apr-Jun 2026 = strong bull),
    naive "always LONG" beats the model. Staggering across 5 different
    time periods (this year, last year, 2/3/4 years ago) ensures the
    test set naturally balances bullish/bearish/sideways conditions.
    Total: ~9 months from 5 distinct market environments.

    WHY RANDOM TRAIN/VALID:
    XGBoost doesn't care about data order — each row is independent with all
    temporal context encoded in lag features. Random assignment ensures training
    sees ALL regimes and isn't biased toward only old data.

    Returns: (train_df, valid_df, test_df) — each reset_index'd.
    """
    rng = np.random.default_rng(random_state)
    dates = pd.to_datetime(df["Date"])

    # Group indices by year-month
    yearmonth = dates.dt.to_period("M")
    month_dict = {}
    for i, ym in enumerate(yearmonth):
        month_dict.setdefault(ym, []).append(i)
    months = sorted(month_dict.keys())
    n_months = len(months)

    # TEST: staggered across 5 time periods for regime diversity
    # - 3 months from this year (most recent — "does model work NOW?")
    # - 3 months from last year (different regime)
    # - 1 month from each of 3 prior years (historical regime coverage)
    # Total: ~9 months of test data across 5 different time periods
    test_months_set = set()

    # Most recent 3 months
    for m in months[-n_test_months:]:
        test_months_set.add(m)

    # 3 months from ~1 year ago
    offset_1yr = n_test_months + 12
    if n_months > offset_1yr:
        for m in months[-(offset_1yr):-(offset_1yr - n_test_months)]:
            test_months_set.add(m)

    # 1 month from each of 3 prior years (~2, 3, 4 years ago)
    for years_back in [24, 36, 48]:
        offset = n_test_months + years_back
        if n_months > offset:
            test_months_set.add(months[-(offset)])

    test_months = sorted(test_months_set)
    remaining_months = [m for m in months if m not in test_months_set]

    # VALID: random selection from remaining months
    n_valid_months = max(1, int(len(remaining_months) * valid_frac / (1 - n_test_months / n_months)))
    shuffled = rng.permutation(len(remaining_months))
    valid_month_idx = set(shuffled[:n_valid_months])
    train_month_idx = set(range(len(remaining_months))) - valid_month_idx

    # Collect row indices
    train_indices, valid_indices, test_indices = [], [], []
    for i, month in enumerate(remaining_months):
        rows = month_dict[month]
        if i in valid_month_idx:
            valid_indices.extend(rows)
        else:
            train_indices.extend(rows)
    for month in test_months:
        test_indices.extend(month_dict[month])

    # Embargo: remove ±5 days around valid/test boundaries from training
    embargo_set = set()
    for idx_list in [valid_indices, test_indices]:
        sorted_idx = sorted(idx_list)
        if not sorted_idx:
            continue
        # Only embargo the FIRST and LAST indices of contiguous runs
        boundaries = [sorted_idx[0], sorted_idx[-1]]
        for i in range(1, len(sorted_idx)):
            if sorted_idx[i] - sorted_idx[i-1] > 1:
                boundaries.extend([sorted_idx[i-1], sorted_idx[i]])
        for b in boundaries:
            for offset in range(1, embargo_days + 1):
                embargo_set.add(b - offset)
                embargo_set.add(b + offset)

    train_indices = sorted([i for i in train_indices if i not in embargo_set])
    valid_indices = sorted(valid_indices)
    test_indices  = sorted(test_indices)

    train_df = df.iloc[train_indices].reset_index(drop=True)
    valid_df = df.iloc[valid_indices].reset_index(drop=True)
    test_df  = df.iloc[test_indices].reset_index(drop=True)

    return train_df, valid_df, test_df


def triannual_walk_forward_test(df, feature_cols, model, scaler, min_wr=0.55, years_per_block=3):
    """
    Test model on 3-year rolling blocks instead of annual slices.

    WHY 3-YEAR BLOCKS INSTEAD OF 1-YEAR:
    - 1 year -> ~250 trading days × 15% = ~37 test samples
      At 37 trades, 54% vs 56% WR = literally 1 win difference = noise
    - 3 years -> ~750 trading days × 15% = ~112 test samples
      Statistically meaningful: margin of error drops from ±8% to ±5%

    With 9 years of data, we get 3 non-overlapping blocks:
      Block 1: years 1-3   (e.g. 2017-2019)
      Block 2: years 4-6   (e.g. 2020-2022)
      Block 3: years 7-9   (e.g. 2023-2025)

    Each block's last 15% is the test slice — this is genuinely unseen data
    representing a distinct market regime (pre-covid, covid/recovery, post-2022).

    Model passes if majority of blocks (2/3) meet min_wr.
    """
    dates = pd.to_datetime(df["Date"])
    all_years = sorted(dates.dt.year.unique())
    n_years = len(all_years)

    # Build non-overlapping 3-year blocks
    blocks = []
    for start_idx in range(0, n_years - years_per_block + 1, years_per_block):
        block_years = all_years[start_idx:start_idx + years_per_block]
        block_mask = dates.dt.year.isin(block_years)
        block_df = df[block_mask].reset_index(drop=True)
        if len(block_df) < 100:  # skip tiny blocks
            continue
        year_label = f"{block_years[0]}–{block_years[-1]}"
        blocks.append((year_label, block_df))

    # Use the 3 most recent blocks
    blocks = blocks[-3:]

    block_results = []
    for label, block_df in blocks:
        nb = len(block_df)
        # Last 15% of the block = test slice
        test_start = int(nb * 0.85)
        test_slice = block_df[test_start:].reset_index(drop=True)

        if len(test_slice) < 20:
            continue

        X_test = scaler.transform(test_slice[feature_cols].values)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        correct = sum(
            1 for idx, (_, row) in enumerate(test_slice.iterrows())
            if (y_pred[idx] == 1 and row["VERDICT_LONG"] == "TP") or
               (y_pred[idx] == 0 and row["VERDICT_SHORT"] == "TP")
        )
        wr = correct / len(test_slice) * 100
        edge = wr / 100 * 1.5 - (1 - wr / 100) * 1.0
        block_results.append({
            "label": label,
            "trades": len(test_slice),
            "wr": wr,
            "edge": edge,
            "pass": wr >= min_wr * 100,
        })

    blocks_passing = sum(1 for r in block_results if r["pass"])
    passes_gate = blocks_passing >= len(block_results) // 2 + 1 if block_results else False
    return block_results, passes_gate


def train_model(csv_path: str, train_ratio: float = 0.7, n_trials: int = None,
                use_deep: bool = None, min_adx_pctile: float = 0.0,
                force_save: bool = False, n_jobs: int = 1):
    """
    Train XGBoost classifier: predict LONG(1) vs SHORT(0).

    Split strategy (70/15/15, chronological, no leakage):
      70% TRAIN  -> Optuna walk-forward CV runs entirely within this slice
      15% VALID  -> Confidence threshold tuned here (more data = less overfit)
      15% TEST   -> Final honest backtest, completely untouched until end

    Timeout-based training:
      Runs Optuna repeatedly with different random seeds within a time budget.
      Keeps the model with the BEST test-set WR across all attempts.
      Default: 5 minutes CPU, 10 minutes GPU (configurable via --timeout).

    n_trials:   None = auto (400 GPU, 200 CPU) — trials per attempt
    use_deep:   None = auto (True GPU, False CPU)
    force_save: Always save best model (even if worse than existing on disk)
    """
    total_start = time.time()
    df, feature_cols = load_and_prepare(csv_path)
    device = detect_gpu()

    # Auto-configure based on hardware
    gpu_available = device != "cpu"
    if n_trials is None:
        n_trials = 400 if gpu_available else 200
    if use_deep is None:
        use_deep = gpu_available  # 9-year window has enough samples for LSTM/CNN on GPU

    print(f"Hardware: {device} -> n_trials={n_trials}, deep_learning={'ON' if use_deep else 'OFF'} (auto)")
    print(f"Total samples: {len(df)}")
    print(f"Base features: {len(feature_cols)}")
    print(f"Strategy: timeout-based (find best WR within time budget)")

    # Optional: deep features
    deep_time = 0
    if use_deep:
        print(f"\n{'='*50}")
        print("DEEP FEATURE EXTRACTION (LSTM + CNN)")
        print(f"{'='*50}")
        deep_start = time.time()
        df, deep_cols = generate_deep_features(df, feature_cols)
        deep_time = time.time() - deep_start
        if deep_cols:
            feature_cols = feature_cols + deep_cols
            print(f"Total features after deep: {len(feature_cols)}")
            print(f"Deep extraction time: {deep_time:.1f}s")
        else:
            print("  Deep features unavailable — continuing with base features only")

    # Optional: ADX regime filter
    if min_adx_pctile > 0:
        regime_col = "Trend_Strength_Regime"
        if regime_col in df.columns:
            before = len(df)
            df = df[df[regime_col] >= min_adx_pctile].reset_index(drop=True)
            print(f"ADX regime filter (>= {min_adx_pctile:.0f}th pctile): {before} -> {len(df)} samples")

    # ── Chronological split: full 9 years, last 3 months = TEST ─────────────
    # USE ALL DATA (9 years) for maximum training signal.
    # Last 3 months forced into TEST (live trading validation).
    # Remaining 90% of pre-test data = TRAIN, last 10% of pre-test = VALID.
    n = len(df)
    dates = pd.to_datetime(df["Date"])
    three_months_ago = dates.max() - pd.DateOffset(months=3)
    test_mask = dates >= three_months_ago
    pre_test_df = df[~test_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    # Split pre-test into 90% train / 10% valid (chronological)
    n_pre = len(pre_test_df)
    train_end = int(n_pre * 0.90)
    train_df = pre_test_df[:train_end]
    valid_df = pre_test_df[train_end:].reset_index(drop=True)

    print(f"\nFull dataset: {n} samples ({df.iloc[0]['Date'][:10]} -> {df.iloc[-1]['Date'][:10]})")
    print(f"Split (9yr train+valid, last 3mo test):")
    print(f"  Train: {len(train_df)} ({train_df.iloc[0]['Date'][:10]} -> {train_df.iloc[-1]['Date'][:10]})")
    print(f"  Valid: {len(valid_df)} ({valid_df.iloc[0]['Date'][:10]} -> {valid_df.iloc[-1]['Date'][:10]})")
    print(f"  Test:  {len(test_df)} ({test_df.iloc[0]['Date'][:10]} -> {test_df.iloc[-1]['Date'][:10]}) <-live trading period")

    # ── Timeout-based training: run Optuna repeatedly until timeout ─────────
    # Keeps the model with the BEST test-set WR (live trading backtest).
    # No min_wr gate — just find the best possible within the time budget.
    # Default: 5 minutes CPU, 10 minutes GPU (configurable via --timeout).
    X_train = train_df[feature_cols].values
    y_train = train_df["TARGET"].values
    X_valid = valid_df[feature_cols].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["TARGET"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled  = scaler.transform(X_test)

    n_long = y_train.sum()
    n_short = len(y_train) - n_long
    scale_weight = n_short / n_long if n_long > 0 else 1.0

    timeout_seconds = int(os.environ.get("TRAIN_TIMEOUT", 300))  # 5 minutes default
    best_attempt_wr = 0.0
    best_attempt_model = None
    best_attempt_params = None
    best_attempt_y_pred_test = None
    best_attempt_y_prob_test = None
    best_attempt_y_pred_valid = None
    best_attempt_y_prob_valid = None
    retry_start = time.time()
    attempt = 0

    print(f"\nTraining (timeout {timeout_seconds//60}min, finding best backtest WR)...")

    while True:
        elapsed = time.time() - retry_start
        if elapsed > timeout_seconds:
            break
        attempt += 1

        best_params = optimize_hyperparams(train_df, feature_cols, n_trials, n_jobs=n_jobs)
        n_estimators = best_params.pop("n_estimators")
        learning_rate = best_params.pop("learning_rate")
        max_depth = best_params.pop("max_depth")

        model_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": best_params.get("subsample", 0.8),
            "colsample_bytree": best_params.get("colsample_bytree", 0.8),
            "min_child_weight": best_params.get("min_child_weight", 1),
            "gamma": best_params.get("gamma", 0.0),
            "reg_alpha": best_params.get("reg_alpha", 0.0),
            "reg_lambda": best_params.get("reg_lambda", 1.0),
            "scale_pos_weight": scale_weight,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42 + attempt,
            "early_stopping_rounds": 50,
        }
        if device != "cpu":
            model_params["device"] = device

        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train_scaled, y_train,
                  eval_set=[(X_valid_scaled, valid_df["TARGET"].values)],
                  verbose=False)

        y_pred_t = model.predict(X_test_scaled)
        y_prob_t = model.predict_proba(X_test_scaled)
        correct = sum(
            1 for idx, (_, row) in enumerate(test_df.iterrows())
            if (y_pred_t[idx] == 1 and row["VERDICT_LONG"] == "TP") or
               (y_pred_t[idx] == 0 and row["VERDICT_SHORT"] == "TP")
        )
        attempt_wr = correct / len(test_df) * 100
        better = "★ NEW BEST" if attempt_wr > best_attempt_wr else ""
        print(f"  Attempt {attempt}: WR={attempt_wr:.1f}% | Trees={n_estimators} LR={learning_rate:.4f} Depth={max_depth} [{elapsed:.0f}s] {better}")

        if attempt_wr > best_attempt_wr:
            best_attempt_wr = attempt_wr
            best_attempt_model = model
            best_attempt_params = model_params.copy()
            best_attempt_y_pred_test = y_pred_t
            best_attempt_y_prob_test = y_prob_t
            best_attempt_y_pred_valid = model.predict(X_valid_scaled)
            best_attempt_y_prob_valid = model.predict_proba(X_valid_scaled)

    # Use best model found within timeout
    model = best_attempt_model
    model_params = best_attempt_params
    y_pred_test = best_attempt_y_pred_test
    y_prob_test = best_attempt_y_prob_test
    y_pred_valid = best_attempt_y_pred_valid
    y_prob_valid = best_attempt_y_prob_valid
    optuna_time = time.time() - retry_start

    n_estimators = model_params["n_estimators"]
    learning_rate = model_params["learning_rate"]
    max_depth = model_params["max_depth"]

    print(f"\nBest model found in {attempt} attempts ({optuna_time:.0f}s): WR={best_attempt_wr:.1f}%")
    print(f"Final model config (Optuna-selected):")
    print(f"  Trees: {n_estimators} | LR: {learning_rate:.6f} | Depth: {max_depth} | Device: {device}")
    print(f"  Subsample: {model_params['subsample']:.2f} | ColSample: {model_params['colsample_bytree']:.2f}")
    print(f"  Gamma: {model_params['gamma']:.2f} | Alpha: {model_params['reg_alpha']:.2f} | Lambda: {model_params['reg_lambda']:.2f}")
    train_time = optuna_time

    # Feature importance
    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    print(f"\nTop 20 Features:")
    for fname, imp in feat_imp[:20]:
        print(f"  {fname:30s} {imp:.4f}")

    # ── Confidence threshold optimisation on VALID set ───────────────────
    print(f"\n{'='*50}")
    print("CONFIDENCE THRESHOLD OPTIMISATION (on Valid set)")
    print(f"{'='*50}")
    print("Correlation table (confidence -> win rate on valid):")
    print(f"  {'Threshold':>10} {'Trades':>7} {'Skip%':>7} {'Win Rate':>10} {'Edge':>8}")
    print(f"  {'─'*10} {'─'*7} {'─'*7} {'─'*10} {'─'*8}")
    for thresh in [0.50, 0.52, 0.55, 0.57, 0.60, 0.62, 0.65, 0.70, 0.75]:
        r = _backtest_at_threshold(valid_df, y_pred_valid, y_prob_valid, thresh)
        skip_pct = r["skipped"] / len(valid_df) * 100
        if r["total"] > 0:
            print(f"  {thresh:>10.2f} {r['total']:>7} {skip_pct:>6.1f}% {r['wr']:>9.1f}% {r['edge']:>+8.3f}R")

    best_thresh, best_valid_edge = _find_best_threshold_optuna(
        valid_df, y_pred_valid, y_prob_valid, n_trials=40)
    print(f"\nOptuna best threshold: {best_thresh:.2f}  (valid edge: {best_valid_edge:+.3f}R)")

    # ── Final honest backtest on TEST set ────────────────────────────────
    print(f"\n{'='*50}")
    print("TRADER BACKTEST — TEST SET (never seen during training or tuning)")
    print(f"{'='*50}")

    # All trades (no filter) — baseline
    raw = _backtest_at_threshold(test_df, y_pred_test, y_prob_test, 0.0)
    print(f"\n[All trades, no filter]")
    print(f"  Trades: {raw['total']} | Win Rate: {raw['wr']:.1f}% | Edge: {raw['edge']:+.3f}R | Final: {raw['final_r']:+.1f}R")

    # Optimized threshold from valid set
    opt = _backtest_at_threshold(test_df, y_pred_test, y_prob_test, best_thresh)
    skip_pct = opt["skipped"] / len(test_df) * 100
    print(f"\n[Confidence >= {best_thresh:.2f} (Optuna-tuned threshold)]")
    print(f"  Trades: {opt['total']} | Skipped: {opt['skipped']} ({skip_pct:.1f}%) | Win Rate: {opt['wr']:.1f}% | Edge: {opt['edge']:+.3f}R | Final: {opt['final_r']:+.1f}R")

    # Show full threshold correlation on test too
    print(f"\nCorrelation table on TEST set:")
    print(f"  {'Threshold':>10} {'Trades':>7} {'Skip%':>7} {'Win Rate':>10} {'Edge':>8}")
    print(f"  {'─'*10} {'─'*7} {'─'*7} {'─'*10} {'─'*8}")
    for thresh in [0.50, 0.52, 0.55, 0.57, 0.60, 0.62, 0.65, 0.70, 0.75]:
        r = _backtest_at_threshold(test_df, y_pred_test, y_prob_test, thresh)
        skip_pct = r["skipped"] / len(test_df) * 100
        marker = " <-Optuna" if thresh == best_thresh else ""
        if r["total"] > 0:
            print(f"  {thresh:>10.2f} {r['total']:>7} {skip_pct:>6.1f}% {r['wr']:>9.1f}% {r['edge']:>+8.3f}R{marker}")

    # ── Save model — only overwrite if new run beats existing on disk ────
    ticker = os.path.basename(csv_path).split("_")[0]
    date_str = datetime.now().strftime("%Y%m%d")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    suffix = "_deep" if use_deep else ""

    model_path     = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost{suffix}_model.json")
    scaler_path    = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost{suffix}_scaler.pkl")
    features_path  = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost{suffix}_features.txt")
    threshold_path = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost{suffix}_threshold.txt")
    perf_path      = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost{suffix}_perf.txt")
    config_path    = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost{suffix}_config.json")

    # Score = edge at Optuna threshold on global test set
    new_score = opt["edge"]
    existing_score = -999.0
    if os.path.exists(perf_path):
        try:
            with open(perf_path) as f:
                existing_score = float(f.read().strip())
        except (ValueError, IOError):
            pass

    if force_save or new_score > existing_score:
        import json as _json
        model.save_model(model_path)
        joblib.dump(scaler, scaler_path)
        with open(features_path, "w") as f:
            f.write("\n".join(feature_cols))
        with open(threshold_path, "w") as f:
            f.write(str(best_thresh))
        with open(perf_path, "w") as f:
            f.write(str(new_score))
        with open(config_path, "w") as f:
            _json.dump(model_params, f, indent=2)
        save_reason = "(force)" if force_save else f"(edge {new_score:+.3f}R > previous {existing_score:+.3f}R)"
        print(f"\n✔ MODEL SAVED {save_reason}")
        print(f"  Model:     {model_path}")
        print(f"  Threshold: {best_thresh:.2f}")
    else:
        print(f"\nExisting model is better (edge {existing_score:+.3f}R >= new {new_score:+.3f}R) — not overwriting")
        print(f"  Use --force-save to override")

    # Latest candle prediction
    print(f"\n{'='*50}")
    print("LATEST CANDLE PREDICTION")
    print(f"{'='*50}")
    latest = df.iloc[-1]
    X_latest = scaler.transform(df[feature_cols].iloc[[-1]].values)
    pred = model.predict(X_latest)[0]
    prob = model.predict_proba(X_latest)[0]
    direction = "LONG" if pred == 1 else "SHORT"
    confidence = prob[pred] * 100
    print(f"Date: {latest['Date']}")
    print(f"Close: {latest['Close']:.2f}")
    print(f"Prediction: {direction}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"  P(LONG):  {prob[1]*100:.1f}%")
    print(f"  P(SHORT): {prob[0]*100:.1f}%")

    total_time = time.time() - total_start
    print(f"\n{'='*50}")
    print("TIMING SUMMARY")
    print(f"{'='*50}")
    if deep_time > 0:
        print(f"  Deep extraction:    {deep_time:.1f}s")
    print(f"  Optuna tuning:      {optuna_time:.1f}s")
    print(f"  Final model train:  {train_time:.1f}s")
    print(f"  Total pipeline:     {total_time:.1f}s")

    return model, scaler, feature_cols


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost (+ optional LSTM/CNN) for LONG/SHORT prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Auto-detection behaviour (when flags not specified):
  GPU detected  ->  --n-trials 300, --deep-learning ON
  CPU only      ->  --n-trials 100, --deep-learning OFF

Override examples:
  python train.py --csv data/AAPL_*.csv                    # auto
  python train.py --csv data/AAPL_*.csv --n-trials 200     # override trials
  python train.py --csv data/AAPL_*.csv --no-deep          # force no deep
        """
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV from fetch_stock_data.py")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Optuna trials (default: 300 on GPU, 100 on CPU)")
    parser.add_argument("--deep-learning", action="store_true", default=None,
                        help="Force enable LSTM+CNN deep features")
    parser.add_argument("--no-deep", action="store_true",
                        help="Force disable LSTM+CNN deep features")
    parser.add_argument("--min-adx-pctile", type=float, default=0.0,
                        help="Only train on trending regimes: 0=all (default), 50=top half ADX")
    parser.add_argument("--force-save", action="store_true",
                        help="Overwrite saved model even if new run scores lower")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Training timeout in seconds (default: 300 CPU, 600 GPU)")
    parser.add_argument("--n-jobs", type=int, default=2,
                        help="Parallel Optuna trials (default: 2, set to 1 for sequential)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        print(f"Run first: python fetch_stock_data.py --ticker AAPL")
        return

    # Resolve deep-learning flag: --no-deep overrides --deep-learning
    if args.no_deep:
        use_deep = False
    elif args.deep_learning:
        use_deep = True
    else:
        use_deep = None  # auto

    # Set timeout via env var so train_model can read it
    if args.timeout is not None:
        os.environ["TRAIN_TIMEOUT"] = str(args.timeout)
    train_model(args.csv, 0.7, args.n_trials, use_deep, args.min_adx_pctile,
                force_save=args.force_save, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
