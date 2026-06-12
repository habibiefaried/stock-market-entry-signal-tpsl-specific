import pandas as pd
import numpy as np
import json
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
import joblib
import os
import argparse
import warnings
from datetime import datetime
from train import (load_and_prepare, detect_gpu,
                   generate_deep_features, _backtest_at_threshold)

# Suppress XGBoost device-mismatch warning (numpy arrays on CPU auto-converted
# to CUDA DMatrix — harmless performance hint, not a correctness issue)
warnings.filterwarnings("ignore", message=".*mismatched devices.*")


def backtest(df, y_pred, y_prob, threshold=0.0):
    """Run trader-style backtest: only enter trades where confidence >= threshold."""
    trades = []
    for idx, (_, row) in enumerate(df.iterrows()):
        pred = y_pred[idx]
        prob = y_prob[idx][pred]
        if prob < threshold:
            continue
        direction = "LONG" if pred == 1 else "SHORT"

        if direction == "LONG":
            hit_tp = row["VERDICT_LONG"] == "TP"
            days = row["DAY_PASS_LONG"]
        else:
            hit_tp = row["VERDICT_SHORT"] == "TP"
            days = row["DAY_PASS_SHORT"]

        pnl_r = 1.5 if hit_tp else -1.0
        trades.append({
            "date": row["Date"],
            "close": row["Close"],
            "direction": direction,
            "confidence": prob,
            "hit_tp": hit_tp,
            "pnl_r": pnl_r,
            "days": int(days),
            "cumulative_r": 0,
        })

    # Compute cumulative R
    cum = 0
    for t in trades:
        cum += t["pnl_r"]
        t["cumulative_r"] = cum

    return trades


def monte_carlo(trades, n_simulations=10000, n_trades=252):
    """
    Monte Carlo simulation of future performance based on observed win rate.
    Simulates n_trades (1 year of daily trades) for n_simulations paths.
    """
    wins = sum(1 for t in trades if t["hit_tp"])
    win_rate = wins / len(trades)
    win_pnl = 1.5
    loss_pnl = -1.0

    rng = np.random.default_rng(42)
    results = rng.random((n_simulations, n_trades))
    results = np.where(results < win_rate, win_pnl, loss_pnl)
    cumulative = results.cumsum(axis=1)

    final_pnl = cumulative[:, -1]
    percentiles = np.percentile(final_pnl, [5, 25, 50, 75, 95])
    max_drawdowns = []
    for path in cumulative:
        peak = np.maximum.accumulate(path)
        dd = peak - path
        max_drawdowns.append(dd.max())

    return {
        "win_rate": win_rate,
        "n_simulations": n_simulations,
        "n_trades": n_trades,
        "final_pnl_mean": final_pnl.mean(),
        "final_pnl_std": final_pnl.std(),
        "percentiles": percentiles,
        "p5": percentiles[0],
        "p25": percentiles[1],
        "p50": percentiles[2],
        "p75": percentiles[3],
        "p95": percentiles[4],
        "prob_profitable": (final_pnl > 0).mean() * 100,
        "avg_max_drawdown": np.mean(max_drawdowns),
        "worst_drawdown": np.max(max_drawdowns),
        "paths_sample": cumulative[:100],
    }


def generate_html(ticker, trades, mc_results, model_metrics, latest_prediction, feature_importance,
                  model_label="XGBoost"):
    """Generate HTML report with backtest, Monte Carlo, and verdict."""
    wins = sum(1 for t in trades if t["hit_tp"])
    losses = len(trades) - wins
    win_rate = wins / len(trades) * 100
    total_r = sum(t["pnl_r"] for t in trades)
    avg_r = total_r / len(trades)

    # Equity curve data
    equity_dates = [t["date"] for t in trades]
    equity_values = [t["cumulative_r"] for t in trades]

    # Monte Carlo paths for chart (ensure pure Python floats for JSON)
    mc_paths_js = []
    for path in mc_results["paths_sample"][:50]:
        mc_paths_js.append([round(float(x), 2) for x in path])

    # Feature importance top 20
    top_features = feature_importance[:20]

    # Latest prediction
    direction = latest_prediction["direction"]
    confidence = latest_prediction["confidence"]
    tradeable = latest_prediction.get("tradeable", True)
    best_threshold = latest_prediction.get("threshold", 0.50)
    verdict_color = "#22c55e" if (direction == "LONG" and tradeable) else ("#ef4444" if direction == "SHORT" and tradeable else "#64748b")
    verdict_emoji = "&#9650;" if direction == "LONG" else "&#9660;"
    trade_badge = f'<span style="background:#22c55e22;color:#22c55e;padding:4px 12px;border-radius:4px;font-size:0.85rem">&#10004; TRADE (conf {confidence:.1f}% &ge; {best_threshold*100:.0f}%)</span>' if tradeable else f'<span style="background:#ef444422;color:#ef4444;padding:4px 12px;border-radius:4px;font-size:0.85rem">&#10008; SKIP (conf {confidence:.1f}% &lt; {best_threshold*100:.0f}%)</span>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{ticker} - TP/SL Trading Signal Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ font-size: 2rem; margin-bottom: 8px; color: #f8fafc; }}
        h2 {{ font-size: 1.3rem; margin-bottom: 12px; color: #94a3b8; }}
        h3 {{ font-size: 1.1rem; margin-bottom: 8px; color: #cbd5e1; }}
        .subtitle {{ color: #64748b; margin-bottom: 24px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 24px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }}
        .verdict-card {{ background: linear-gradient(135deg, #1e293b 0%, {verdict_color}22 100%); border-color: {verdict_color}66; text-align: center; padding: 30px; }}
        .verdict-direction {{ font-size: 3rem; font-weight: 800; color: {verdict_color}; }}
        .verdict-confidence {{ font-size: 1.5rem; color: #94a3b8; margin-top: 8px; }}
        .stat {{ margin-bottom: 12px; }}
        .stat-label {{ font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }}
        .stat-value {{ font-size: 1.4rem; font-weight: 600; color: #f1f5f9; }}
        .stat-value.positive {{ color: #22c55e; }}
        .stat-value.negative {{ color: #ef4444; }}
        .chart-container {{ position: relative; height: 300px; margin-top: 12px; }}
        .chart-container-lg {{ height: 400px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ color: #64748b; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; }}
        .bar {{ height: 8px; border-radius: 4px; background: #334155; }}
        .bar-fill {{ height: 100%; border-radius: 4px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
        .badge-green {{ background: #22c55e22; color: #22c55e; }}
        .badge-red {{ background: #ef444422; color: #ef4444; }}
        .trades-scroll {{ max-height: 400px; overflow-y: auto; }}
        .mc-stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #334155; color: #475569; font-size: 0.8rem; text-align: center; }}
    </style>
</head>
<body>
<div class="container">
    <h1>{ticker} Trading Signal Report</h1>
    <p class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | Model: {model_label} | TP: 1.5x ATR | SL: 1x ATR</p>

    <!-- Verdict Section -->
    <div class="grid">
        <div class="card verdict-card">
            <h2>TODAY'S VERDICT</h2>
            <div class="verdict-direction">{verdict_emoji} {direction}</div>
            <div class="verdict-confidence">Confidence: {confidence:.1f}%</div>
            <div style="margin-top: 10px;">{trade_badge}</div>
            <p style="margin-top: 12px; color: #94a3b8; font-size: 0.9rem;">
                Date: {latest_prediction['date']} | Close: ${latest_prediction['close']:.2f}
            </p>
            <p style="margin-top: 4px; color: #94a3b8; font-size: 0.85rem;">
                P(LONG): {latest_prediction['p_long']:.1f}% | P(SHORT): {latest_prediction['p_short']:.1f}%
            </p>
        </div>

        <div class="card">
            <h3>Backtest Performance</h3>
            <div class="stat">
                <div class="stat-label">Win Rate</div>
                <div class="stat-value {'positive' if win_rate > 50 else 'negative'}">{win_rate:.1f}%</div>
            </div>
            <div class="stat">
                <div class="stat-label">Total Trades</div>
                <div class="stat-value">{len(trades)}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Wins / Losses</div>
                <div class="stat-value">{wins} / {losses}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Total P&L</div>
                <div class="stat-value {'positive' if total_r > 0 else 'negative'}">{total_r:+.1f}R</div>
            </div>
            <div class="stat">
                <div class="stat-label">Avg P&L per Trade</div>
                <div class="stat-value {'positive' if avg_r > 0 else 'negative'}">{avg_r:+.3f}R</div>
            </div>
        </div>

        <div class="card">
            <h3>Monte Carlo (252 trades)</h3>
            <div class="stat">
                <div class="stat-label">Prob. Profitable (1yr)</div>
                <div class="stat-value {'positive' if mc_results['prob_profitable'] > 50 else 'negative'}">{mc_results['prob_profitable']:.1f}%</div>
            </div>
            <div class="stat">
                <div class="stat-label">Median Final P&L</div>
                <div class="stat-value {'positive' if mc_results['p50'] > 0 else 'negative'}">{mc_results['p50']:+.1f}R</div>
            </div>
            <div class="stat">
                <div class="stat-label">5th / 95th Percentile</div>
                <div class="stat-value">{mc_results['p5']:+.1f}R / {mc_results['p95']:+.1f}R</div>
            </div>
            <div class="stat">
                <div class="stat-label">Avg Max Drawdown</div>
                <div class="stat-value negative">{mc_results['avg_max_drawdown']:.1f}R</div>
            </div>
            <div class="stat">
                <div class="stat-label">Worst Drawdown (all sims)</div>
                <div class="stat-value negative">{mc_results['worst_drawdown']:.1f}R</div>
            </div>
        </div>
    </div>

    <!-- Model Metrics -->
    <div class="grid">
        <div class="card">
            <h3>Model Metrics (Test Set)</h3>
            <div class="stat">
                <div class="stat-label">Accuracy</div>
                <div class="stat-value">{model_metrics['accuracy']:.1f}%</div>
            </div>
            <div class="stat">
                <div class="stat-label">Precision (LONG)</div>
                <div class="stat-value">{model_metrics['precision_long']:.1f}%</div>
            </div>
            <div class="stat">
                <div class="stat-label">Precision (SHORT)</div>
                <div class="stat-value">{model_metrics['precision_short']:.1f}%</div>
            </div>
            <div class="stat">
                <div class="stat-label">Expected Edge (@ conf≥{model_metrics['threshold']*100:.0f}%)</div>
                <div class="stat-value {'positive' if model_metrics['edge'] > 0 else 'negative'}">{model_metrics['edge']:+.3f}R/trade</div>
            </div>
        </div>

        <div class="card" style="grid-column: span 2;">
            <h3>Equity Curve (Backtest)</h3>
            <div class="chart-container chart-container-lg">
                <canvas id="equityChart"></canvas>
            </div>
        </div>
    </div>

    <!-- Monte Carlo Chart -->
    <div class="card" style="margin-bottom: 24px;">
        <h3>Monte Carlo Simulation ({mc_results['n_simulations']:,} paths, {mc_results['n_trades']} trades each)</h3>
        <div class="chart-container chart-container-lg">
            <canvas id="mcChart"></canvas>
        </div>
    </div>

    <!-- Feature Importance -->
    <div class="grid">
        <div class="card">
            <h3>Top 20 Feature Importance</h3>
            <table>
                <thead><tr><th>Feature</th><th>Importance</th><th></th></tr></thead>
                <tbody>
"""
    max_imp = top_features[0][1] if top_features else 1
    for fname, imp in top_features:
        pct = imp / max_imp * 100
        html += f"""                    <tr><td>{fname}</td><td>{imp:.4f}</td><td><div class="bar"><div class="bar-fill" style="width:{pct:.0f}%"></div></div></td></tr>\n"""

    html += """                </tbody>
            </table>
        </div>

        <div class="card">
            <h3>All Test Trades</h3>
            <div class="trades-scroll">
            <table>
                <thead><tr><th>Date</th><th>Dir</th><th>Result</th><th>Days</th><th>Conf</th><th>Cum R</th></tr></thead>
                <tbody>
"""
    for t in trades:
        badge = 'badge-green' if t['hit_tp'] else 'badge-red'
        result = 'TP' if t['hit_tp'] else 'SL'
        html += f"""                    <tr><td>{t['date'][:10]}</td><td>{t['direction']}</td><td><span class="badge {badge}">{result}</span></td><td>{t['days']}d</td><td>{t['confidence']*100:.0f}%</td><td>{t['cumulative_r']:+.1f}R</td></tr>\n"""

    html += f"""                </tbody>
            </table>
            </div>
        </div>
    </div>

    <div class="footer">
        <p>Model trained on {model_metrics['train_samples']} samples | Tested on {model_metrics['test_samples']} samples | Features: {model_metrics['n_features']}</p>
        <p>Risk Warning: Past performance does not guarantee future results. This is a statistical model, not financial advice.</p>
    </div>
</div>

<script>
// Equity Curve
const equityCtx = document.getElementById('equityChart').getContext('2d');
new Chart(equityCtx, {{
    type: 'line',
    data: {{
        labels: {json.dumps([t['date'][:10] for t in trades])},
        datasets: [{{
            label: 'Cumulative P&L (R)',
            data: {json.dumps([round(float(v), 2) for v in equity_values])},
            borderColor: '{verdict_color}',
            backgroundColor: '{verdict_color}22',
            fill: true,
            tension: 0.1,
            pointRadius: 1,
            borderWidth: 2,
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ display: true, ticks: {{ maxTicksLimit: 10, color: '#64748b' }}, grid: {{ color: '#1e293b' }} }},
            y: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }}
        }}
    }}
}});

// Monte Carlo
const mcCtx = document.getElementById('mcChart').getContext('2d');
const mcPaths = {json.dumps(mc_paths_js[:50])};
const mcDatasets = mcPaths.map((path, i) => ({{
    data: path,
    borderColor: path[path.length-1] > 0 ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)',
    borderWidth: 1,
    pointRadius: 0,
    fill: false,
}}));
// Add percentile lines
const p50Path = {json.dumps([round(float(x), 2) for x in np.percentile(mc_results['paths_sample'], 50, axis=0)])};
const p5Path = {json.dumps([round(float(x), 2) for x in np.percentile(mc_results['paths_sample'], 5, axis=0)])};
const p95Path = {json.dumps([round(float(x), 2) for x in np.percentile(mc_results['paths_sample'], 95, axis=0)])};
mcDatasets.push({{ data: p50Path, borderColor: '#f59e0b', borderWidth: 3, pointRadius: 0, fill: false, label: 'Median' }});
mcDatasets.push({{ data: p5Path, borderColor: '#ef4444', borderWidth: 2, borderDash: [5,5], pointRadius: 0, fill: false, label: '5th pct' }});
mcDatasets.push({{ data: p95Path, borderColor: '#22c55e', borderWidth: 2, borderDash: [5,5], pointRadius: 0, fill: false, label: '95th pct' }});

new Chart(mcCtx, {{
    type: 'line',
    data: {{
        labels: Array.from({{length: {mc_results['n_trades']}}}, (_, i) => i+1),
        datasets: mcDatasets
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: true, labels: {{ filter: (item) => item.text !== undefined, color: '#94a3b8' }} }} }},
        scales: {{
            x: {{ title: {{ display: true, text: 'Trade #', color: '#64748b' }}, ticks: {{ maxTicksLimit: 10, color: '#64748b' }}, grid: {{ color: '#1e293b' }} }},
            y: {{ title: {{ display: true, text: 'Cumulative R', color: '#64748b' }}, grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }}
        }}
    }}
}});
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report with backtest and Monte Carlo")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV from fetch_stock_data.py")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Optuna trials (default: 300 on GPU, 100 on CPU)")
    parser.add_argument("--mc-simulations", type=int, default=10000, help="Monte Carlo simulations")
    parser.add_argument("--mc-trades", type=int, default=252, help="Trades per MC simulation")
    parser.add_argument("--deep-learning", action="store_true", default=False,
                        help="Force enable LSTM+CNN deep features")
    parser.add_argument("--no-deep", action="store_true",
                        help="Force disable LSTM+CNN deep features")
    args = parser.parse_args()

    # --- Validate inputs ---
    if not os.path.exists(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        print(f"Run first: python fetch_stock_data.py --ticker AAPL")
        return

    ticker = os.path.basename(args.csv).split("_")[0]
    date_str = os.path.basename(args.csv).split("_tpsl_data_")[1].replace(".csv", "")
    device = detect_gpu()
    gpu_available = device != "cpu"

    # Resolve deep flag
    if args.no_deep:
        use_deep = False
    elif args.deep_learning:
        use_deep = True
    else:
        use_deep = gpu_available

    suffix = "_deep" if use_deep else ""
    model_label = "XGBoost-DeepLearning" if use_deep else "XGBoost"

    # ── Try to load existing model (avoid retraining inconsistency) ───────
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")
    model_path     = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost{suffix}_model.json")
    scaler_path    = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost{suffix}_scaler.pkl")
    features_path  = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost{suffix}_features.txt")
    threshold_path = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost{suffix}_threshold.txt")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path, threshold_path]):
        print(f"ERROR: No trained model found for {ticker} {date_str} ({suffix or 'standard'})")
        print(f"Run first:")
        print(f"  python train.py --csv {args.csv}")
        return

    # ── Load saved model (single source of truth — matches current.py exactly) ──
    print(f"Loading model: {os.path.basename(model_path)}")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)
    with open(features_path) as f:
        feature_cols = [line.strip() for line in f.readlines()]
    with open(threshold_path) as f:
        best_threshold = float(f.read().strip())
    print(f"Model loaded | Threshold: {best_threshold:.2f} | Features: {len(feature_cols)}")

    # Load + prepare data for test set evaluation and report
    df, _ = load_and_prepare(args.csv)
    n = len(df)
    # Same 3-year chronological 80/10/10 split as train.py
    dates = pd.to_datetime(df["Date"])
    three_years_ago = dates.max() - pd.DateOffset(years=3)
    df = df[dates >= three_years_ago].reset_index(drop=True)
    n = len(df)
    train_df = df[:int(n * 0.80)]
    test_df  = df[int(n * 0.90):].reset_index(drop=True)
    print(f"Test set: {len(test_df)} samples ({test_df.iloc[0]['Date'][:10]} → {test_df.iloc[-1]['Date'][:10]})")

    X_test = test_df[feature_cols].values
    y_test = test_df["TARGET"].values
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred) * 100
    prec_long = precision_score(y_test, y_pred, pos_label=1, zero_division=0) * 100
    prec_short = precision_score(y_test, y_pred, pos_label=0, zero_division=0) * 100

    accuracy = accuracy_score(y_test, y_pred) * 100
    prec_long = precision_score(y_test, y_pred, pos_label=1, zero_division=0) * 100
    prec_short = precision_score(y_test, y_pred, pos_label=0, zero_division=0) * 100

    # Run trader-style backtest at optimised threshold
    print(f"Running backtest (confidence threshold: {best_threshold:.2f})...")
    bt = _backtest_at_threshold(test_df, y_pred, y_prob, best_threshold)
    win_rate = bt["wr"] / 100
    edge = bt["edge"]

    # Build trades list for equity curve + Monte Carlo (at optimised threshold)
    trades = backtest(test_df, y_pred, y_prob, threshold=best_threshold)

    # Monte Carlo
    print(f"Running Monte Carlo ({args.mc_simulations:,} simulations)...")
    mc_results = monte_carlo(trades, args.mc_simulations, args.mc_trades)

    # Feature importance
    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)

    # Latest prediction
    X_latest = scaler.transform(df[feature_cols].iloc[[-1]].values)
    pred_latest = model.predict(X_latest)[0]
    prob_latest = model.predict_proba(X_latest)[0]
    tradeable = prob_latest[pred_latest] >= best_threshold
    latest_prediction = {
        "direction": "LONG" if pred_latest == 1 else "SHORT",
        "confidence": prob_latest[pred_latest] * 100,
        "p_long": prob_latest[1] * 100,
        "p_short": prob_latest[0] * 100,
        "date": df.iloc[-1]["Date"],
        "close": df.iloc[-1]["Close"],
        "threshold": best_threshold,
        "tradeable": tradeable,
    }

    model_metrics = {
        "accuracy": accuracy,
        "precision_long": prec_long,
        "precision_short": prec_short,
        "edge": edge,
        "threshold": best_threshold,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "n_features": len(feature_cols),
    }

    # Generate HTML
    print("Generating HTML report...")
    html = generate_html(ticker, trades, mc_results, model_metrics, latest_prediction, feat_imp,
                         model_label=model_label)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    report_date = datetime.now().strftime("%Y%m%d")
    report_suffix = "_deep-learning" if use_deep else ""
    output_path = os.path.join(output_dir, f"{ticker}_report_{report_date}{report_suffix}.html")
    with open(output_path, "w") as f:
        f.write(html)

    print(f"\nReport saved to: {output_path}")
    print(f"\nVERDICT: {latest_prediction['direction']} (confidence: {latest_prediction['confidence']:.1f}%)")
    print(f"Win Rate: {win_rate*100:.1f}% | Edge: {edge:+.3f}R/trade")
    print(f"Monte Carlo P(profitable 1yr): {mc_results['prob_profitable']:.1f}%")


if __name__ == "__main__":
    main()
