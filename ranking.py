"""
ranking.py - Train all stocks in target_stocks.txt and rank by confidence + win rate.

Reads tickers from target_stocks.txt, trains XGBoost for each, ranks results.
Models are saved for reuse with current.py.

Usage:
    python ranking.py
    python ranking.py --n-trials 200
    python ranking.py --deep-learning             # use LSTM+CNN experimental model
    python ranking.py --lookahead 5               # tighter label window (opt 2)
    python ranking.py --min-adx-pctile 50         # trending regime only (opt 3)
"""
import os
import sys
import argparse
import subprocess
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Train and rank all stocks")
    parser.add_argument("--n-trials", type=int, default=100, help="Optuna trials per stock")
    parser.add_argument("--deep-learning", action="store_true", help="Use LSTM+CNN experimental model")
    parser.add_argument("--lookahead", type=int, default=10,
                        help="Days to look forward for TP/SL label (default 10, try 5 for tighter signals)")
    parser.add_argument("--min-adx-pctile", type=float, default=0.0,
                        help="Only train on trending regimes: 0=all (default), 50=top half ADX")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_file = os.path.join(base_dir, "target_stocks.txt")

    if not os.path.exists(target_file):
        print("ERROR: target_stocks.txt not found!")
        print("Create it with one ticker per line, e.g.:")
        print("  AAPL")
        print("  MSFT")
        print("  GOOGL")
        sys.exit(1)

    with open(target_file) as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    if not tickers:
        print("ERROR: target_stocks.txt is empty!")
        sys.exit(1)

    train_script = "train.py"
    model_label = "XGBoost-DeepLearning" if args.deep_learning else "XGBoost"

    print(f"{'='*70}")
    print(f"  STOCK RANKING PIPELINE")
    print(f"{'='*70}")
    print(f"  Tickers:        {', '.join(tickers)}")
    print(f"  Model:          {model_label}")
    print(f"  Optuna trials:  {args.n_trials}")
    print(f"  Lookahead:      {args.lookahead} days")
    print(f"  ADX filter:     {args.min_adx_pctile if args.min_adx_pctile > 0 else 'off'}")
    print(f"{'='*70}\n")

    date_str = datetime.now().strftime("%Y%m%d")
    data_dir = os.path.join(base_dir, "data")
    results = []

    for i, ticker in enumerate(tickers, 1):
        print(f"\n{'─'*70}")
        print(f"  [{i}/{len(tickers)}] {ticker}")
        print(f"{'─'*70}")

        # Step 1: Fetch data
        print(f"  Fetching data (lookahead={args.lookahead})...")
        fetch_result = subprocess.run(
            [sys.executable, os.path.join(base_dir, "fetch_stock_data.py"),
             "--ticker", ticker, "--lookahead", str(args.lookahead)],
            capture_output=True, text=True, cwd=base_dir
        )
        if fetch_result.returncode != 0:
            print(f"  ERROR fetching {ticker}: {fetch_result.stderr[:200]}")
            results.append({"ticker": ticker, "error": "fetch_failed"})
            continue

        csv_path = os.path.join(data_dir, f"{ticker}_tpsl_data_{date_str}.csv")
        if not os.path.exists(csv_path):
            print(f"  ERROR: CSV not found at {csv_path}")
            results.append({"ticker": ticker, "error": "csv_not_found"})
            continue

        # Step 2: Train
        adx_label = f", ADX>={args.min_adx_pctile:.0f}%" if args.min_adx_pctile > 0 else ""
        print(f"  Training ({model_label}, {args.n_trials} trials{adx_label})...")
        train_cmd = [sys.executable, os.path.join(base_dir, train_script),
                     "--csv", csv_path, "--n-trials", str(args.n_trials)]
        if args.deep_learning:
            train_cmd.append("--deep-learning")
        if args.min_adx_pctile > 0:
            train_cmd += ["--min-adx-pctile", str(args.min_adx_pctile)]
        train_result = subprocess.run(
            train_cmd, capture_output=True, text=True, cwd=base_dir
        )

        if train_result.returncode != 0:
            print(f"  ERROR training {ticker}: {train_result.stderr[:200]}")
            results.append({"ticker": ticker, "error": "train_failed"})
            continue

        # Parse output
        output = train_result.stdout
        win_rate = None
        edge = None
        confidence = None
        prediction = None
        close_price = None
        optuna_wr = None

        for line in output.split("\n"):
            if "Win Rate:" in line and "Confidence" not in line:
                try:
                    win_rate = float(line.split("Win Rate:")[1].strip().replace("%", ""))
                except (ValueError, IndexError):
                    pass
            if "Expected Edge:" in line:
                try:
                    edge = float(line.split("Expected Edge:")[1].strip().replace("R per trade", "").strip())
                except (ValueError, IndexError):
                    pass
            if "Prediction:" in line:
                prediction = line.split("Prediction:")[1].strip()
            if "Confidence:" in line:
                try:
                    confidence = float(line.split("Confidence:")[1].strip().replace("%", ""))
                except (ValueError, IndexError):
                    pass
            if "Close:" in line and close_price is None:
                try:
                    close_price = float(line.split("Close:")[1].strip())
                except (ValueError, IndexError):
                    pass
            if "Best walk-forward win rate:" in line:
                try:
                    optuna_wr = float(line.split(":")[1].strip().replace("%", ""))
                except (ValueError, IndexError):
                    pass

        result = {
            "ticker": ticker,
            "prediction": prediction or "N/A",
            "confidence": confidence or 0,
            "win_rate": win_rate or 0,
            "edge": edge or 0,
            "optuna_wr": optuna_wr or 0,
            "close": close_price or 0,
            "error": None,
        }
        results.append(result)
        print(f"  Result: {prediction} | Conf: {confidence:.1f}% | WR: {win_rate:.1f}% | Edge: {edge:+.3f}R")

    # Rank by composite score: confidence * win_rate (both matter)
    valid_results = [r for r in results if r.get("error") is None]
    valid_results.sort(key=lambda r: r["confidence"] * r["win_rate"], reverse=True)
    failed_results = [r for r in results if r.get("error") is not None]

    # Generate report
    print(f"\n\n{'='*70}")
    print(f"  FINAL RANKING")
    print(f"{'='*70}")
    print(f"  {'Rank':<5} {'Ticker':<7} {'Direction':<10} {'Confidence':<12} {'Win Rate':<10} {'Edge':<10} {'Close':<10}")
    print(f"  {'─'*5} {'─'*7} {'─'*10} {'─'*12} {'─'*10} {'─'*10} {'─'*10}")

    for rank, r in enumerate(valid_results, 1):
        print(f"  {rank:<5} {r['ticker']:<7} {r['prediction']:<10} {r['confidence']:.1f}%{'':<7} {r['win_rate']:.1f}%{'':<5} {r['edge']:+.3f}R{'':<4} ${r['close']:.2f}")

    if failed_results:
        print(f"\n  FAILED:")
        for r in failed_results:
            print(f"    {r['ticker']}: {r['error']}")

    # Save to file
    output_path = os.path.join(base_dir, "ranking-report.txt")
    with open(output_path, "w") as f:
        f.write(f"STOCK RANKING REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Model: {model_label} | Trials: {args.n_trials} | Lookahead: {args.lookahead}d | ADX filter: {args.min_adx_pctile if args.min_adx_pctile > 0 else 'off'}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"{'Rank':<5} {'Ticker':<7} {'Direction':<10} {'Confidence':<12} {'Win Rate':<10} {'Edge':<10} {'Close':<10}\n")
        f.write(f"{'─'*5} {'─'*7} {'─'*10} {'─'*12} {'─'*10} {'─'*10} {'─'*10}\n")
        for rank, r in enumerate(valid_results, 1):
            f.write(f"{rank:<5} {r['ticker']:<7} {r['prediction']:<10} {r['confidence']:.1f}%{'':<7} {r['win_rate']:.1f}%{'':<5} {r['edge']:+.3f}R{'':<4} ${r['close']:.2f}\n")
        if failed_results:
            f.write(f"\nFAILED:\n")
            for r in failed_results:
                f.write(f"  {r['ticker']}: {r['error']}\n")
        f.write(f"\n{'='*70}\n")
        f.write(f"Ranking sorted by: Confidence × Win Rate (composite score)\n")
        f.write(f"Only trade top-ranked stocks with confidence >= 55%\n")

    print(f"\n  Report saved to: {output_path}")
    print(f"  Models saved to: models/{'{TICKER}'}_{date_str}_xgboost_*")


if __name__ == "__main__":
    main()
