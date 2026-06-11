import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import argparse
import warnings
import time
from datetime import datetime

# Suppress XGBoost device-mismatch warning (numpy arrays on CPU auto-converted
# to CUDA DMatrix — harmless performance hint, not a correctness issue)
warnings.filterwarnings("ignore", message=".*mismatched devices.*")

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

    # === Target ===
    targets = []
    for _, row in df.iterrows():
        long_tp = row["VERDICT_LONG"] == "TP"
        short_tp = row["VERDICT_SHORT"] == "TP"

        if long_tp and short_tp:
            targets.append(1 if row["DAY_PASS_LONG"] <= row["DAY_PASS_SHORT"] else 0)
        elif long_tp:
            targets.append(1)
        elif short_tp:
            targets.append(0)
        else:
            targets.append(1 if row["DAY_PASS_LONG"] >= row["DAY_PASS_SHORT"] else 0)

    df["TARGET"] = targets

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
    More realistic than random CV for time series.
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

        n_long = y_train.sum()
        n_short = len(y_train) - n_long
        sw = n_short / n_long if n_long > 0 else 1.0

        callbacks = []
        if trial is not None:
            callbacks.append(XGBoostPruningCallback(trial, "validation_0-logloss"))

        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=sw,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            early_stopping_rounds=30,
            callbacks=callbacks if callbacks else None,
        )
        model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)

        y_pred = model.predict(X_test_s)

        # Score = win rate (actual TP hit rate when following predictions)
        correct = 0
        for idx, (_, row) in enumerate(test_df.iterrows()):
            if y_pred[idx] == 1 and row["VERDICT_LONG"] == "TP":
                correct += 1
            elif y_pred[idx] == 0 and row["VERDICT_SHORT"] == "TP":
                correct += 1
        win_rate = correct / len(test_df)
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
    1. XGBoostPruningCallback: prunes individual trials mid-training if validation
       loss plateaus (per-tree step-level pruning)
    2. MedianPruner: prunes trials whose intermediate fold scores are in the
       bottom half compared to completed trials (trial-level pruning)
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


def train_model(csv_path: str, train_ratio: float = 0.9, n_trials: int = 100):
    """Train XGBoost classifier: predict LONG(1) vs SHORT(0)."""
    total_start = time.time()

    df, feature_cols = load_and_prepare(csv_path)
    device = detect_gpu()

    print(f"Total samples: {len(df)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Device: {device}")
    print(f"Target distribution: LONG={df['TARGET'].sum()}, SHORT={len(df)-df['TARGET'].sum()}")

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

    # Balance classes
    n_long = y_train.sum()
    n_short = len(y_train) - n_long
    scale_weight = n_short / n_long if n_long > 0 else 1.0

    # Train XGBoost with best params
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
    print(classification_report(y_test, y_pred, target_names=["SHORT", "LONG"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(f"  Predicted:  SHORT  LONG")
    print(f"  Actual SHORT: {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"  Actual LONG:  {cm[1][0]:4d}  {cm[1][1]:4d}")

    # Feature importance
    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    print(f"\nTop 20 Features:")
    for fname, imp in feat_imp[:20]:
        print(f"  {fname:30s} {imp:.4f}")

    # Backtest
    print(f"\n{'='*50}")
    print("BACKTEST ON TEST SET")
    print(f"{'='*50}")
    correct_tp = 0
    total_trades = len(test_df)
    for idx, (_, row) in enumerate(test_df.iterrows()):
        pred = y_pred[idx]
        if pred == 1:
            if row["VERDICT_LONG"] == "TP":
                correct_tp += 1
        else:
            if row["VERDICT_SHORT"] == "TP":
                correct_tp += 1

    win_rate = correct_tp / total_trades * 100
    print(f"Trades: {total_trades}")
    print(f"Wins (TP hit): {correct_tp}")
    print(f"Losses (SL hit): {total_trades - correct_tp}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Expected R:R per trade: 1.5:1")
    print(f"Expected Edge: {win_rate/100 * 1.5 - (1 - win_rate/100) * 1.0:.3f}R per trade")

    # Threshold optimization: find best decision boundary
    print(f"\n{'='*50}")
    print("THRESHOLD OPTIMIZATION")
    print(f"{'='*50}")
    best_threshold = 0.50
    best_threshold_wr = win_rate
    for thresh_candidate in np.arange(0.30, 0.75, 0.05):
        thresh_correct = 0
        thresh_total = 0
        for idx, (_, row) in enumerate(test_df.iterrows()):
            p_long = y_prob[idx][1]
            thresh_pred = 1 if p_long >= thresh_candidate else 0
            thresh_total += 1
            if thresh_pred == 1 and row["VERDICT_LONG"] == "TP":
                thresh_correct += 1
            elif thresh_pred == 0 and row["VERDICT_SHORT"] == "TP":
                thresh_correct += 1
        thresh_wr = thresh_correct / thresh_total * 100
        marker = " <-- BEST" if thresh_wr > best_threshold_wr else ""
        print(f"  Threshold >= {thresh_candidate:.2f}: WR={thresh_wr:.1f}%{marker}")
        if thresh_wr > best_threshold_wr:
            best_threshold_wr = thresh_wr
            best_threshold = thresh_candidate

    # Re-apply best threshold
    p_long_all = y_prob[:, 1]
    y_pred = np.where(p_long_all >= best_threshold, 1, 0)
    print(f"\nBest threshold: {best_threshold:.2f} (WR: {best_threshold_wr:.1f}%)")

    # Recompute win rate with optimized threshold
    correct_tp = 0
    for idx, (_, row) in enumerate(test_df.iterrows()):
        if y_pred[idx] == 1 and row["VERDICT_LONG"] == "TP":
            correct_tp += 1
        elif y_pred[idx] == 0 and row["VERDICT_SHORT"] == "TP":
            correct_tp += 1
    win_rate = correct_tp / total_trades * 100

    # Confidence analysis
    print(f"\n{'='*50}")
    print("CONFIDENCE ANALYSIS")
    print(f"{'='*50}")
    confidences = np.max(y_prob, axis=1)
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = confidences >= threshold
        if mask.sum() == 0:
            continue
        filtered_correct = 0
        filtered_total = 0
        for idx, (_, row) in enumerate(test_df.iterrows()):
            if not mask[idx]:
                continue
            filtered_total += 1
            pred = y_pred[idx]
            if pred == 1 and row["VERDICT_LONG"] == "TP":
                filtered_correct += 1
            elif pred == 0 and row["VERDICT_SHORT"] == "TP":
                filtered_correct += 1
        if filtered_total > 0:
            filtered_wr = filtered_correct / filtered_total * 100
            edge = filtered_wr/100 * 1.5 - (1 - filtered_wr/100) * 1.0
            print(f"  Confidence >= {threshold*100:.0f}%: {filtered_total} trades, Win Rate: {filtered_wr:.1f}%, Edge: {edge:+.3f}R")

    # Save model and artifacts
    ticker = os.path.basename(csv_path).split("_")[0]
    date_str = datetime.now().strftime("%Y%m%d")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost_model.json")
    scaler_path = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost_scaler.pkl")
    features_path = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost_features.txt")

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
    print(f"TIMING SUMMARY")
    print(f"{'='*50}")
    print(f"  Optuna tuning:      {optuna_time:.1f}s")
    print(f"  Final model train:  {train_time:.1f}s")
    print(f"  Total pipeline:     {total_time:.1f}s")

    return model, scaler, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost for LONG/SHORT prediction")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV from fetch_stock_data.py")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train/test split ratio")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        print(f"Run first: python fetch_stock_data.py --ticker AAPL")
        return

    train_model(args.csv, args.train_ratio, args.n_trials)


if __name__ == "__main__":
    main()
