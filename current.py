"""
current.py - Make a LONG/SHORT decision based on current price.

When you enter the traderoom, you see the current price but today's OHLCV
candle isn't complete yet. This script:
1. Loads the latest trained model for the ticker
2. Fetches recent completed candles, computes indicators + all engineered features
3. Uses the LAST COMPLETED candle's features to predict direction
4. Calculates TP/SL levels and percentages based on YOUR current price + ATR

Usage:
    python current.py --ticker AAPL --price 301.50
    python current.py --ticker AAPL --price 301.50 --tp-mult 2.0 --sl-mult 1.0
"""
import os
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")

from fetch_stock_data import fetch_ohlcv, compute_atr, compute_technical_indicators
from train_xgboost import load_and_prepare


def find_latest_model(ticker: str, model_dir: str):
    """Find the most recent model files for a ticker. Prefers _deep_ models if flag set."""
    pattern = os.path.join(model_dir, f"{ticker}_*_xgboost_model.json")
    models = sorted(glob.glob(pattern), reverse=True)
    if not models:
        return None, None, None

    model_path = models[0]
    prefix = model_path.replace("_xgboost_model.json", "")
    scaler_path = prefix + "_xgboost_scaler.pkl"
    features_path = prefix + "_xgboost_features.txt"

    if not os.path.exists(scaler_path) or not os.path.exists(features_path):
        return None, None, None

    return model_path, scaler_path, features_path


def get_prepared_data(ticker: str):
    """
    Fetch data and run the full feature engineering pipeline (same as training).
    Returns the fully prepared dataframe with all engineered features and ATR.
    """
    # Fetch enough data: 14 months covers the 252-day rolling percentile window + warmup
    df_raw = fetch_ohlcv(ticker, months=14, warmup_days=300)
    df_raw["ATR"] = compute_atr(df_raw, 14)
    df_raw = compute_technical_indicators(df_raw)
    df_raw = df_raw.dropna().reset_index(drop=True)

    if len(df_raw) == 0:
        return None

    # Save a temp CSV so load_and_prepare can process it (it also builds all lag/regime/interaction features)
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp_path = tmp.name
    tmp.close()

    # Add dummy VERDICT columns so load_and_prepare doesn't crash on missing columns
    df_raw["VERDICT_LONG"] = "TP"
    df_raw["VERDICT_SHORT"] = "TP"
    df_raw["DAY_PASS_LONG"] = 1
    df_raw["DAY_PASS_SHORT"] = 1
    df_raw["LONG_TP_Level"] = 0
    df_raw["LONG_SL_Level"] = 0
    df_raw["SHORT_TP_Level"] = 0
    df_raw["SHORT_SL_Level"] = 0
    df_raw.to_csv(tmp_path, index=False)

    try:
        df_prepared, feature_cols_from_prep = load_and_prepare(tmp_path)
    finally:
        os.unlink(tmp_path)

    return df_prepared


def main():
    parser = argparse.ArgumentParser(description="Get LONG/SHORT decision for current price")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker")
    parser.add_argument("--price", type=float, required=True, help="Current market price")
    parser.add_argument("--tp-mult", type=float, default=1.5, help="TP multiplier of ATR")
    parser.add_argument("--sl-mult", type=float, default=1.0, help="SL multiplier of ATR")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")

    # Find model
    model_path, scaler_path, features_path = find_latest_model(args.ticker, model_dir)
    if model_path is None:
        print(f"ERROR: No trained model found for {args.ticker}")
        print(f"Run first: python train_xgboost.py --csv data/{args.ticker}_tpsl_data_YYYYMMDD.csv")
        return

    print(f"{'='*60}")
    print(f"  {args.ticker} TRADE DECISION — Current Price: ${args.price:.2f}")
    print(f"{'='*60}")
    print(f"\nModel: {os.path.basename(model_path)}")

    # Load model
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)
    with open(features_path) as f:
        feature_cols = [line.strip() for line in f.readlines()]

    # Fetch + run full feature engineering pipeline (same as training)
    print("Fetching latest market data...")
    df = get_prepared_data(args.ticker)
    if df is None:
        print("ERROR: Could not fetch or prepare market data")
        return

    latest_row = df.iloc[-1]
    last_close = latest_row["Close"]
    last_date = latest_row["Date"]
    atr = latest_row["ATR"]
    print(f"Last completed candle: {str(last_date)[:10]} | Close: ${last_close:.2f}")
    print(f"ATR(14): ${atr:.2f}")

    # Verify all required features are present
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"ERROR: Missing features: {missing[:5]}...")
        print("Retrain model with: python train_xgboost.py --csv data/{ticker}_tpsl_data_YYYYMMDD.csv")
        return

    # Predict using the latest completed candle's indicators
    X = df[feature_cols].iloc[[-1]].values
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]

    direction = "LONG" if pred == 1 else "SHORT"
    confidence = prob[pred] * 100

    # Calculate TP/SL based on CURRENT PRICE (not last close)
    if direction == "LONG":
        tp_price = args.price + args.tp_mult * atr
        sl_price = args.price - args.sl_mult * atr
    else:
        tp_price = args.price - args.tp_mult * atr
        sl_price = args.price + args.sl_mult * atr

    tp_pct = (tp_price - args.price) / args.price * 100
    sl_pct = (sl_price - args.price) / args.price * 100

    # Output
    direction_symbol = "▲" if direction == "LONG" else "▼"
    print(f"\n{'─'*60}")
    print(f"  VERDICT: {direction_symbol} {direction}")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"  P(LONG): {prob[1]*100:.1f}%  |  P(SHORT): {prob[0]*100:.1f}%")
    print(f"{'─'*60}")
    print(f"\n  Entry Price:  ${args.price:.2f}")
    print(f"  Take Profit:  ${tp_price:.2f}  ({tp_pct:+.2f}%)")
    print(f"  Stop Loss:    ${sl_price:.2f}  ({sl_pct:+.2f}%)")
    print(f"  R:R Ratio:    {args.tp_mult:.1f} : {args.sl_mult:.1f}")
    print(f"\n  ATR-based levels:")
    print(f"    TP = Entry {'+ ' if direction == 'LONG' else '- '}{args.tp_mult} × ATR({atr:.2f}) = ${tp_price:.2f}")
    print(f"    SL = Entry {'- ' if direction == 'LONG' else '+ '}{args.sl_mult} × ATR({atr:.2f}) = ${sl_price:.2f}")

    # Risk table for common position sizes
    print(f"\n{'─'*60}")
    print(f"  POSITION SIZING (per $10,000 capital)")
    print(f"{'─'*60}")
    risk_per_share = abs(args.price - sl_price)
    for risk_pct in [1, 2, 3, 5]:
        risk_amount = 10000 * risk_pct / 100
        shares = int(risk_amount / risk_per_share)
        position_size = shares * args.price
        potential_gain = shares * abs(tp_price - args.price)
        print(f"  {risk_pct}% risk (${risk_amount:.0f}): {shares} shares (${position_size:.0f}) → TP gain: ${potential_gain:.0f}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
