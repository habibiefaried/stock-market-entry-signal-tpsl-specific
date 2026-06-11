import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import argparse
import warnings
from datetime import datetime

# Suppress XGBoost device-mismatch warning (numpy arrays on CPU auto-converted
# to CUDA DMatrix — harmless performance hint, not a correctness issue)
warnings.filterwarnings("ignore", message=".*mismatched devices.*")

import time

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from optuna_integration import XGBoostPruningCallback
except ImportError:
    raise ImportError("optuna and optuna-integration are required. Run: pip install optuna 'optuna-integration[xgboost]'")


def load_and_prepare(csv_path: str):
    """Load CSV, engineer scale-invariant lag features, create target."""
    df = pd.read_csv(csv_path)
    df = df.copy()  # defragment (fetch_stock_data.py creates 170+ columns via sequential assignment)

    # === Add scale-invariant lag features (collect all, concat once) ===
    close = df["Close"]
    volume = df["Volume"]

    new_cols = {}
    daily_return = close.pct_change() * 100
    for lag in [1, 2, 3, 5, 10]:
        new_cols[f"Return_Lag_{lag}"] = daily_return.shift(lag)

    vol_change = volume.pct_change() * 100
    for lag in [1, 2, 3, 5]:
        new_cols[f"VolChange_Lag_{lag}"] = vol_change.shift(lag)

    for lag in [1, 2, 3, 5]:
        new_cols[f"MACD_Hist_Lag_{lag}"] = df["MACD_Hist"].shift(lag)

    for lag in [1, 2, 3]:
        new_cols[f"BB_Pct_Lag_{lag}"] = df["BB_Pct"].shift(lag)

    for lag in [1, 2, 3]:
        new_cols[f"Stoch_K_Lag_{lag}"] = df["Stoch_K"].shift(lag)

    for lag in [1, 2, 3]:
        new_cols[f"ADX_Lag_{lag}"] = df["ADX"].shift(lag)

    for lag in [1, 2, 3]:
        new_cols[f"CCI_Lag_{lag}"] = df["CCI"].shift(lag)

    new_cols["Volatility_Change_5d"] = df["Volatility_5d"] - df["Volatility_5d"].shift(5)
    new_cols["RSI14_Accel"] = df["RSI14_Slope_3d"] - df["RSI14_Slope_3d"].shift(3)

    # Concat all new columns at once (avoids PerformanceWarning: DataFrame fragmentation)
    new_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_df], axis=1)

    # Drop rows with NaN from new lags
    df = df.dropna().reset_index(drop=True)

    # === Market regime features (percentile-based, scale-invariant) ===
    close = df["Close"]
    atr_series = df["ATR"]
    regime_cols = {}
    regime_cols["ATR_Pctile_60d"] = atr_series.rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    ret_20d = close.pct_change(20)
    regime_cols["Return20d_Pctile_252d"] = ret_20d.rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    regime_cols["Volatility_Regime"] = df["Volatility_20d"].rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    regime_cols["Trend_Strength_Regime"] = df["ADX"].rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    regime_df = pd.DataFrame(regime_cols, index=df.index)
    df = pd.concat([df, regime_df], axis=1)

    # Drop NaN from regime features
    df = df.dropna().reset_index(drop=True)

    # === Target: only label clear outcomes, drop noisy samples ===
    targets = []
    keep_mask = []
    for _, row in df.iterrows():
        long_tp = row["VERDICT_LONG"] == "TP"
        short_tp = row["VERDICT_SHORT"] == "TP"

        if long_tp and short_tp:
            targets.append(1 if row["DAY_PASS_LONG"] <= row["DAY_PASS_SHORT"] else 0)
            keep_mask.append(True)
        elif long_tp:
            targets.append(1)
            keep_mask.append(True)
        elif short_tp:
            targets.append(0)
            keep_mask.append(True)
        else:
            # Both hit SL — noisy sample, skip
            targets.append(-1)
            keep_mask.append(False)

    df["TARGET"] = targets
    n_before = len(df)
    df = df[keep_mask].reset_index(drop=True)
    n_dropped = n_before - len(df)
    print(f"Dropped {n_dropped} noisy samples (both SL hit) — {len(df)} clean samples remain")

    # === Feature selection: scale-invariant only ===
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
        "ATR", "MACD", "MACD_Signal",
        "AO", "AO_Signal", "DPO", "Bull_Power", "Bear_Power",
        "EMA9_EMA21_Diff", "SMA20_SMA50_Diff", "SMA50_SMA200_Diff",
        "Volume_SMA_20",
        # New absolute-price indicators
        "Wilder_MA_14", "Chaikin_Osc", "DMA_20_5",
        "EMV", "EMV_Signal",
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Defragment DataFrame (prevents segfaults when used with PyTorch)
    df = df.copy()

    return df, feature_cols


def walk_forward_cv(df, feature_cols, params, n_splits=5, test_size=0.1, trial=None):
    """
    Walk-forward cross-validation: train on expanding window, test on next chunk.
    Uses XGBoostPruningCallback to prune bad trials mid-training (per-tree level).
    """
    n = len(df)
    test_len = int(n * test_size)
    min_train = int(n * 0.4)

    scores = []
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
        y_test = test_df["TARGET"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        callbacks = []
        if trial is not None:
            callbacks.append(XGBoostPruningCallback(trial, "validation_0-mlogloss"))

        model = xgb.XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            early_stopping_rounds=30,
            callbacks=callbacks if callbacks else None,
        )
        model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)

        y_pred = model.predict(X_test_s)

        # Score = win rate on trades taken (LONG/SHORT only, SKIP excluded)
        correct = 0
        traded = 0
        for idx, (_, row) in enumerate(test_df.iterrows()):
            if y_pred[idx] == 2:
                continue
            traded += 1
            if y_pred[idx] == 1 and row["VERDICT_LONG"] == "TP":
                correct += 1
            elif y_pred[idx] == 0 and row["VERDICT_SHORT"] == "TP":
                correct += 1
        win_rate = correct / traded if traded > 0 else 0.0
        scores.append(win_rate)

        # Report intermediate value for pruning (after each fold)
        if trial is not None:
            trial.report(np.mean(scores), i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return np.mean(scores) if scores else 0.0


def optimize_hyperparams(df, feature_cols, n_trials=100):
    """
    Optuna Bayesian hyperparameter optimization with XGBoostPruningCallback.

    Uses two levels of pruning:
    1. XGBoostPruningCallback: prunes individual trials mid-training
    2. MedianPruner: prunes trials whose fold scores are in the bottom half
    """
    print(f"Running Optuna optimization ({n_trials} trials, with XGBoostPruningCallback)...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 10000),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        }
        return walk_forward_cv(df, feature_cols, params, n_splits=5, trial=trial)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest walk-forward win rate: {study.best_value*100:.1f}%")
    print(f"Best params: {study.best_params}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}/{n_trials}")
    return study.best_params


def detect_gpu():
    """Detect if CUDA is available for XGBoost."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def generate_deep_features(df, feature_cols, seq_len=15, epochs=30, lr=0.001):
    """Train a deep LSTM+CNN on indicator sequences, extract compressed features.

    LSTM: 2-layer 64-hidden → 32 → 8 → 2 features (regime encoding)
    CNN:  3-layer 64→32→16 conv → 8 → 2 features (pattern encoding)

    Returns (df_trimmed, deep_cols) where deep_cols = [LSTM_0, LSTM_1, CNN_0, CNN_1].
    First seq_len rows are dropped to form complete sequences.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  [SKIP] PyTorch not installed — skipping deep features. pip install torch")
        return df, []

    # Select a representative subset of scale-invariant indicators for the deep model
    deep_input_cols = [c for c in feature_cols if any(k in c for k in [
        "RSI", "Stoch", "MACD_Hist", "BB_Pct", "CCI", "ADX", "Williams",
        "MFI", "CMF", "ROC", "Momentum", "Volatility", "CMO", "PPO",
        "TRIX", "Vortex", "Aroon", "VHF", "Body_Ratio", "Price_Change",
    ])]
    if len(deep_input_cols) < 10:
        deep_input_cols = feature_cols[:30]

    print(f"  Training LSTM+CNN ({len(deep_input_cols)} inputs, seq_len={seq_len}, epochs={epochs})...")

    # Deep models: more layers, bottleneck to 2 features each (4 total)
    class LSTMFeat(nn.Module):
        def __init__(self, n_in, hidden=64):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, num_layers=2, batch_first=True, dropout=0.2)
            self.fc1 = nn.Linear(hidden, 32)
            self.fc2 = nn.Linear(32, 8)
            self.fc3 = nn.Linear(8, 2)
        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            x = torch.relu(self.fc1(h_n[-1]))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    class CNNFeat(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.conv1 = nn.Conv1d(n_in, 64, 3, padding=1)
            self.conv2 = nn.Conv1d(64, 32, 3, padding=1)
            self.conv3 = nn.Conv1d(32, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(16, 8)
            self.fc2 = nn.Linear(8, 2)
        def forward(self, x):
            x = x.transpose(1, 2)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.pool(x).squeeze(-1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    class CombinedExtractor(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.lstm = LSTMFeat(n_in)
            self.cnn = CNNFeat(n_in)
            self.classifier = nn.Linear(4, 2)
        def forward(self, x):
            l_feat = self.lstm(x)
            c_feat = self.cnn(x)
            combined = torch.cat([l_feat, c_feat], dim=1)
            return self.classifier(combined), combined

    # Prepare data
    from sklearn.preprocessing import StandardScaler
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

    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_size = int(n_seq * 0.8)
    X_train_t = torch.from_numpy(X_seq[:train_size].copy()).to(device)
    y_train_t = torch.from_numpy(y_seq[:train_size].copy()).to(device)

    model = CombinedExtractor(len(deep_input_cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    model.train()
    bs = 64
    for epoch in range(epochs):
        perm = torch.randperm(len(X_train_t))
        total_loss = 0.0
        for i in range(0, len(X_train_t), bs):
            idx = perm[i:i + bs]
            logits, _ = model(X_train_t[idx])
            loss = crit(logits, y_train_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}/{epochs}  loss={total_loss:.4f}")

    # Extract features for all sequences
    model.eval()
    all_feats = []
    with torch.no_grad():
        X_all_t = torch.from_numpy(X_seq.copy()).to(device)
        for i in range(0, len(X_all_t), 256):
            _, feats = model(X_all_t[i:i + 256])
            all_feats.append(feats.cpu().numpy())
    all_feats = np.concatenate(all_feats, axis=0)

    deep_cols = [f"LSTM_{i}" for i in range(2)] + [f"CNN_{i}" for i in range(2)]

    # Trim df to match sequences
    df = df.iloc[seq_len:].reset_index(drop=True).copy()
    for i, col in enumerate(deep_cols):
        df[col] = all_feats[:, i]

    print(f"  Extracted {len(deep_cols)} deep features (samples: {len(df)})")

    # Clean up GPU memory
    del model, X_train_t, y_train_t, X_all_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return df, deep_cols


def train_model(csv_path: str, train_ratio: float = 0.9, n_trials: int = 100,
                use_deep: bool = True):
    """Train XGBoost classifier: predict LONG(1) vs SHORT(0)."""
    total_start = time.time()

    df, feature_cols = load_and_prepare(csv_path)
    device = detect_gpu()

    print(f"Total samples: {len(df)}")
    print(f"Base features: {len(feature_cols)}")
    print(f"Device: {device}")
    print(f"Target distribution: LONG={df['TARGET'].sum()}, SHORT={len(df)-df['TARGET'].sum()}")

    # Optional: LSTM+CNN deep feature extraction
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

    # Always run Optuna to find best hyperparameters
    optuna_start = time.time()
    best_params = optimize_hyperparams(df, feature_cols, n_trials)
    optuna_time = time.time() - optuna_start
    print(f"Optuna time: {optuna_time:.1f}s")
    n_estimators = best_params.pop("n_estimators")
    learning_rate = best_params.pop("learning_rate")
    max_depth = best_params.pop("max_depth")

    # Chronological split
    split_idx = int(len(df) * train_ratio)
    train_df = df[:split_idx]
    test_df = df[split_idx:]

    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")

    X_train = train_df[feature_cols].values
    y_train = train_df["TARGET"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["TARGET"].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost with best params (3-class: LONG=1, SHORT=0, SKIP=2)
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
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "early_stopping_rounds": 50,
    }
    if device != "cpu":
        model_params["device"] = device

    print(f"\nModel params: depth={max_depth}, lr={learning_rate:.4f}, trees={n_estimators}, device={device}")
    train_start = time.time()
    model = xgb.XGBClassifier(**model_params)

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False,
    )
    train_time = time.time() - train_start
    print(f"Final model training time: {train_time:.1f}s")

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    print(f"\n{'='*50}")
    print("TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["SHORT", "LONG", "SKIP"]))

    # Feature importance
    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    print(f"\nTop 20 Features:")
    for fname, imp in feat_imp[:20]:
        print(f"  {fname:30s} {imp:.4f}")

    # Backtest: only count trades where model says LONG or SHORT (not SKIP)
    print(f"\n{'='*50}")
    print("BACKTEST ON TEST SET")
    print(f"{'='*50}")
    correct_tp = 0
    total_trades = 0
    skipped = 0
    for idx, (_, row) in enumerate(test_df.iterrows()):
        pred = y_pred[idx]
        if pred == 2:
            skipped += 1
            continue
        total_trades += 1
        if pred == 1 and row["VERDICT_LONG"] == "TP":
            correct_tp += 1
        elif pred == 0 and row["VERDICT_SHORT"] == "TP":
            correct_tp += 1

    if total_trades > 0:
        win_rate = correct_tp / total_trades * 100
        edge = win_rate / 100 * 1.5 - (1 - win_rate / 100) * 1.0
    else:
        win_rate = 0
        edge = 0
    print(f"Total candles: {len(test_df)}")
    print(f"Trades taken: {total_trades} (LONG/SHORT)")
    print(f"Skipped: {skipped} ({skipped/len(test_df)*100:.1f}%)")
    print(f"Wins (TP hit): {correct_tp}")
    print(f"Losses (SL hit): {total_trades - correct_tp}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Expected Edge: {edge:+.3f}R per trade")

    # Confidence analysis (on non-SKIP predictions only)
    print(f"\n{'='*50}")
    print("CONFIDENCE ANALYSIS")
    print(f"{'='*50}")
    for threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        filtered_correct = 0
        filtered_total = 0
        for idx, (_, row) in enumerate(test_df.iterrows()):
            pred = y_pred[idx]
            if pred == 2:
                continue
            conf = y_prob[idx][pred]
            if conf < threshold:
                continue
            filtered_total += 1
            if pred == 1 and row["VERDICT_LONG"] == "TP":
                filtered_correct += 1
            elif pred == 0 and row["VERDICT_SHORT"] == "TP":
                filtered_correct += 1
        if filtered_total > 0:
            filtered_wr = filtered_correct / filtered_total * 100
            f_edge = filtered_wr / 100 * 1.5 - (1 - filtered_wr / 100) * 1.0
            print(f"  Confidence >= {threshold*100:.0f}%: {filtered_total} trades, Win Rate: {filtered_wr:.1f}%, Edge: {f_edge:+.3f}R")

    # Save model and artifacts
    ticker = os.path.basename(csv_path).split("_")[0]
    date_str = datetime.now().strftime("%Y%m%d")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost_deep_model.json")
    scaler_path = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost_deep_scaler.pkl")
    features_path = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost_deep_features.txt")

    model.save_model(model_path)
    joblib.dump(scaler, scaler_path)
    with open(features_path, "w") as f:
        f.write("\n".join(feature_cols))

    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Features saved to: {features_path}")

    # Predict on latest candle
    print(f"\n{'='*50}")
    print("LATEST CANDLE PREDICTION")
    print(f"{'='*50}")
    latest = df.iloc[-1]
    X_latest = scaler.transform(df[feature_cols].iloc[[-1]].values)
    pred = model.predict(X_latest)[0]
    prob = model.predict_proba(X_latest)[0]
    direction_map = {0: "SHORT", 1: "LONG", 2: "SKIP"}
    direction = direction_map[pred]
    confidence = prob[pred] * 100
    print(f"Date: {latest['Date']}")
    print(f"Close: {latest['Close']:.2f}")
    print(f"Prediction: {direction}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"  P(SHORT): {prob[0]*100:.1f}%")
    print(f"  P(LONG):  {prob[1]*100:.1f}%")
    print(f"  P(SKIP):  {prob[2]*100:.1f}%")

    total_time = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"TIMING SUMMARY")
    print(f"{'='*50}")
    if deep_time > 0:
        print(f"  Deep extraction:    {deep_time:.1f}s")
    print(f"  Optuna tuning:      {optuna_time:.1f}s")
    print(f"  Final model train:  {train_time:.1f}s")
    print(f"  Total pipeline:     {total_time:.1f}s")

    return model, scaler, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost for LONG/SHORT prediction")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV from fetch_stock_data.py")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train/test split ratio")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--no-deep", action="store_true", help="Disable LSTM+CNN deep features")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        print(f"Run first: python fetch_stock_data.py --ticker AAPL")
        return

    train_model(args.csv, args.train_ratio, args.n_trials, use_deep=not args.no_deep)


if __name__ == "__main__":
    main()
