"""
Genetic Algorithm optimization for XGBoost with LSTM+CNN feature extraction.

1. LSTM + CNN extract learned temporal features from raw indicator sequences
2. GA evolves feature selection (traditional + deep) AND hyperparameters
3. XGBoost makes final LONG/SHORT prediction

Usage:
    python train_xgboost_ga.py --csv data/AAPL_tpsl_data_20260610.csv
    python train_xgboost_ga.py --csv data/AAPL_tpsl_data_20260610.csv --generations 50 --population 40
    python train_xgboost_ga.py --csv data/AAPL_tpsl_data_20260610.csv --no-deep  # skip LSTM/CNN
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os
import argparse
from datetime import datetime
from train_xgboost import load_and_prepare

HAS_TORCH = False
torch = None
nn = None


def detect_device():
    """Detect best available device for XGBoost and PyTorch."""
    xgb_device = "cpu"
    torch_device = "cpu"
    try:
        import torch as _t
        if _t.cuda.is_available():
            torch_device = "cuda"
            xgb_device = "cuda"
        elif hasattr(_t.backends, "mps") and _t.backends.mps.is_available():
            torch_device = "mps"
    except ImportError:
        pass
    return xgb_device, torch_device


# === LSTM + CNN Feature Extractor ===
# Model classes are defined inside generate_deep_features() to avoid
# referencing torch.nn at module import time (deferred import for compatibility).


def generate_deep_features(df, feature_cols, target_col="TARGET", seq_len=15, epochs=30, lr=0.001):
    """
    Train LSTM+CNN on sequences, then extract learned features for each row.
    Returns 24 new columns: 16 from LSTM + 8 from CNN.
    Only feeds key oscillator/ratio features to the neural net (not all 146).
    """
    try:
        import torch as _torch
        import torch.nn as _nn
        global torch, nn, HAS_TORCH
        torch = _torch
        nn = _nn
        HAS_TORCH = True
    except ImportError:
        print("  [SKIP] PyTorch not installed. Run: pip install torch")
        return df, []

    print("  Training LSTM+CNN feature extractor...")

    # Define models (deferred so torch isn't needed at import time)
    class LSTMFeatureExtractor(nn.Module):
        def __init__(self, input_size, hidden_size=32):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
            self.fc = nn.Linear(hidden_size, 16)
        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return self.fc(h_n[-1])

    class CNNFeatureExtractor(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(16, 8)
            self.relu = nn.ReLU()
        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)

    class CombinedExtractor(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.lstm = LSTMFeatureExtractor(input_size)
            self.cnn = CNNFeatureExtractor(input_size)
            self.classifier = nn.Linear(24, 2)
        def forward(self, x):
            lstm_feat = self.lstm(x)
            cnn_feat = self.cnn(x)
            combined = torch.cat([lstm_feat, cnn_feat], dim=1)
            logits = self.classifier(combined)
            return logits, combined

    # Use a subset of scale-invariant features for the deep model
    deep_input_cols = [c for c in feature_cols if any(k in c for k in [
        "RSI", "Stoch", "MACD_Hist", "BB_Pct", "CCI", "ADX", "Williams",
        "MFI", "CMF", "ROC", "Momentum", "Volatility", "CMO", "PPO",
        "TRIX", "Vortex", "Aroon", "VHF", "Body_Ratio", "Price_Change",
    ])]
    if len(deep_input_cols) < 10:
        deep_input_cols = feature_cols[:30]

    print(f"  Deep model input: {len(deep_input_cols)} features, seq_len={seq_len}")

    # Extract all data as contiguous numpy arrays BEFORE touching torch
    feature_data_raw = df[deep_input_cols].values.astype(np.float64).copy()
    targets_raw = df[target_col].values.copy()

    scaler = StandardScaler()
    feature_data = scaler.fit_transform(feature_data_raw).astype(np.float32)

    # Build sequences as contiguous array
    n_seq = len(df) - seq_len
    X_seq = np.zeros((n_seq, seq_len, len(deep_input_cols)), dtype=np.float32)
    y_seq = np.zeros(n_seq, dtype=np.int64)
    for i in range(n_seq):
        X_seq[i] = feature_data[i:i + seq_len]
        y_seq[i] = targets_raw[i + seq_len]

    # Now use torch (all numpy work is done)
    _, torch_device = detect_device()
    device = torch.device(torch_device)
    print(f"  Using device: {device}")

    train_size = int(n_seq * 0.8)
    X_train_t = torch.from_numpy(X_seq[:train_size].copy()).to(device)
    y_train_t = torch.from_numpy(y_seq[:train_size].copy()).to(device)

    input_size = len(deep_input_cols)
    model = CombinedExtractor(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    batch_size = 64
    for epoch in range(epochs):
        perm = torch.randperm(len(X_train_t))
        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i + batch_size]
            batch_x = X_train_t[idx]
            batch_y = y_train_t[idx]

            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Extract features for ALL sequences
    model.eval()
    all_features_list = []
    with torch.no_grad():
        X_all_t = torch.from_numpy(X_seq.copy()).to(device)
        for i in range(0, len(X_all_t), 256):
            batch = X_all_t[i:i + 256]
            _, feats = model(batch)
            all_features_list.append(feats.cpu().numpy().copy())
    all_features = np.concatenate(all_features_list, axis=0)

    # Done with torch - build output columns
    deep_cols = [f"LSTM_{i}" for i in range(16)] + [f"CNN_{i}" for i in range(8)]

    # Trim df to only rows that have sequences (drop first seq_len rows)
    df = df.iloc[seq_len:].reset_index(drop=True).copy()

    # Add deep features
    for i, col in enumerate(deep_cols):
        df[col] = all_features[:, i]

    print(f"  LSTM+CNN extracted {len(deep_cols)} features (seq_len={seq_len}, epochs={epochs})")
    print(f"  Samples after sequence trimming: {len(df)}")

    return df, deep_cols


def decode_chromosome(chromosome, feature_cols):
    """Decode a chromosome into feature mask + hyperparameters."""
    n_features = len(feature_cols)

    # First n_features genes = binary feature selection
    feature_mask = chromosome[:n_features] > 0.5
    selected_features = [f for f, m in zip(feature_cols, feature_mask) if m]

    # Ensure at least 10 features selected
    if len(selected_features) < 10:
        indices = np.argsort(chromosome[:n_features])[-20:]
        feature_mask = np.zeros(n_features, dtype=bool)
        feature_mask[indices] = True
        selected_features = [f for f, m in zip(feature_cols, feature_mask) if m]

    # Last 7 genes = hyperparameters (normalized 0-1, decoded to ranges)
    hp_genes = chromosome[n_features:]
    params = {
        "n_estimators": int(hp_genes[0] * 4500 + 500),      # 500 - 5000
        "learning_rate": 10 ** (hp_genes[1] * -2 - 0.7),    # ~0.005 - 0.2
        "max_depth": int(hp_genes[2] * 7 + 3),              # 3 - 10
        "subsample": hp_genes[3] * 0.4 + 0.6,               # 0.6 - 1.0
        "colsample_bytree": hp_genes[4] * 0.5 + 0.4,        # 0.4 - 0.9
        "min_child_weight": int(hp_genes[5] * 9 + 1),       # 1 - 10
        "gamma": hp_genes[6] * 5.0,                          # 0.0 - 5.0
    }

    return selected_features, params


def evaluate_fitness(chromosome, df, feature_cols, n_folds=3, xgb_device="cpu"):
    """Evaluate chromosome fitness using walk-forward validation."""
    selected_features, params = decode_chromosome(chromosome, feature_cols)
    if xgb_device != "cpu":
        params["device"] = xgb_device

    n = len(df)
    test_size = int(n * 0.08)
    min_train = int(n * 0.6)

    scores = []
    for fold in range(n_folds):
        test_end = n - fold * test_size
        test_start = test_end - test_size
        if test_start < min_train:
            break

        train_df = df[:test_start]
        test_df = df[test_start:test_end].reset_index(drop=True)

        X_train = train_df[selected_features].values
        y_train = train_df["TARGET"].values
        X_test = test_df[selected_features].values
        y_test = test_df["TARGET"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        n_long = y_train.sum()
        n_short = len(y_train) - n_long
        sw = n_short / n_long if n_long > 0 else 1.0

        try:
            model = xgb.XGBClassifier(
                **params,
                scale_pos_weight=sw,
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
                early_stopping_rounds=30,
            )
            model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)

            y_pred = model.predict(X_test_s)

            # Win rate = actual TP hit rate
            correct = 0
            for idx, (_, row) in enumerate(test_df.iterrows()):
                if y_pred[idx] == 1 and row["VERDICT_LONG"] == "TP":
                    correct += 1
                elif y_pred[idx] == 0 and row["VERDICT_SHORT"] == "TP":
                    correct += 1
            scores.append(correct / len(test_df))
        except Exception:
            scores.append(0.0)

    return np.mean(scores) if scores else 0.0


def tournament_selection(population, fitness_scores, tournament_size=3):
    """Select parent via tournament selection."""
    indices = np.random.choice(len(population), tournament_size, replace=False)
    best_idx = indices[np.argmax(fitness_scores[indices])]
    return population[best_idx].copy()


def crossover(parent1, parent2):
    """Two-point crossover."""
    n = len(parent1)
    pt1, pt2 = sorted(np.random.choice(n, 2, replace=False))
    child1 = np.concatenate([parent1[:pt1], parent2[pt1:pt2], parent1[pt2:]])
    child2 = np.concatenate([parent2[:pt1], parent1[pt1:pt2], parent2[pt2:]])
    return child1, child2


def mutate(chromosome, mutation_rate=0.05, n_features=146):
    """Mutate chromosome genes."""
    for i in range(len(chromosome)):
        if np.random.random() < mutation_rate:
            if i < n_features:
                # Feature gene: flip selection
                chromosome[i] = 1.0 - chromosome[i] if chromosome[i] > 0.5 else np.random.random()
            else:
                # Hyperparameter gene: small perturbation
                chromosome[i] = np.clip(chromosome[i] + np.random.normal(0, 0.15), 0, 1)
    return chromosome


def run_ga(csv_path, population_size=30, generations=30, mutation_rate=0.05,
           crossover_rate=0.8, elitism=2, use_deep=True):
    """Run genetic algorithm optimization with optional LSTM+CNN features."""
    xgb_device, torch_device = detect_device()

    df, feature_cols = load_and_prepare(csv_path)

    # Generate deep features (LSTM + CNN)
    if use_deep:
        try:
            df, deep_cols = generate_deep_features(df, feature_cols, seq_len=15, epochs=30)
            feature_cols = feature_cols + deep_cols
        except Exception as e:
            print(f"  [WARN] Deep feature extraction failed: {e}")
            print("  Continuing with traditional features only.")

    n_features = len(feature_cols)
    chromosome_length = n_features + 7  # features + hyperparams

    print(f"{'='*60}")
    print(f"GENETIC ALGORITHM OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Samples: {len(df)} | Features: {n_features}")
    print(f"Population: {population_size} | Generations: {generations}")
    print(f"Mutation: {mutation_rate} | Crossover: {crossover_rate} | Elitism: {elitism}")
    print(f"Chromosome length: {chromosome_length} (features: {n_features} + hyperparams: 7)")
    print(f"XGBoost device: {xgb_device} | PyTorch device: {torch_device}")
    print()

    # Initialize population
    population = np.random.random((population_size, chromosome_length))
    # Bias initial feature selection to ~60% features on
    population[:, :n_features] = np.random.random((population_size, n_features)) * 0.7 + 0.3

    best_fitness_history = []
    avg_fitness_history = []
    best_ever_chromosome = None
    best_ever_fitness = 0.0

    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = np.array([
            evaluate_fitness(chrom, df, feature_cols, xgb_device=xgb_device) for chrom in population
        ])

        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        avg_fitness = np.mean(fitness_scores)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)

        if best_fitness > best_ever_fitness:
            best_ever_fitness = best_fitness
            best_ever_chromosome = population[best_idx].copy()

        selected_feats, params = decode_chromosome(population[best_idx], feature_cols)
        print(f"  Gen {gen+1:3d}/{generations}: Best={best_fitness*100:.1f}% | Avg={avg_fitness*100:.1f}% | "
              f"Features={len(selected_feats)} | depth={params['max_depth']} lr={params['learning_rate']:.4f}")

        # Create next generation
        new_population = []

        # Elitism: keep top N
        elite_indices = np.argsort(fitness_scores)[-elitism:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # Fill rest with crossover + mutation
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            if np.random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = mutate(child1, mutation_rate, n_features)
            child2 = mutate(child2, mutation_rate, n_features)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = np.array(new_population[:population_size])

    # Final evaluation with best chromosome
    print(f"\n{'='*60}")
    print(f"BEST CHROMOSOME (fitness: {best_ever_fitness*100:.1f}%)")
    print(f"{'='*60}")

    selected_features, best_params = decode_chromosome(best_ever_chromosome, feature_cols)
    print(f"\nSelected features ({len(selected_features)}/{n_features}):")
    for f in sorted(selected_features):
        print(f"  {f}")
    print(f"\nHyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Train final model with best chromosome on full train split
    print(f"\n{'='*60}")
    print("FINAL MODEL TRAINING")
    print(f"{'='*60}")

    split_idx = int(len(df) * 0.9)
    train_df = df[:split_idx]
    test_df = df[split_idx:].reset_index(drop=True)

    X_train = train_df[selected_features].values
    y_train = train_df["TARGET"].values
    X_test = test_df[selected_features].values
    y_test = test_df["TARGET"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_long = y_train.sum()
    n_short = len(y_train) - n_long
    sw = n_short / n_long if n_long > 0 else 1.0

    if xgb_device != "cpu":
        best_params["device"] = xgb_device

    model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=sw,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        early_stopping_rounds=50,
    )
    model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")

    # Win rate backtest
    correct = 0
    for idx, (_, row) in enumerate(test_df.iterrows()):
        if y_pred[idx] == 1 and row["VERDICT_LONG"] == "TP":
            correct += 1
        elif y_pred[idx] == 0 and row["VERDICT_SHORT"] == "TP":
            correct += 1

    win_rate = correct / len(test_df) * 100
    edge = win_rate / 100 * 1.5 - (1 - win_rate / 100) * 1.0
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Edge: {edge:+.3f}R per trade")

    # Confidence analysis
    print(f"\nConfidence Analysis:")
    confidences = np.max(y_prob, axis=1)
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = confidences >= threshold
        if mask.sum() == 0:
            continue
        hi_correct = 0
        hi_total = mask.sum()
        for idx, (_, row) in enumerate(test_df.iterrows()):
            if not mask[idx]:
                continue
            if y_pred[idx] == 1 and row["VERDICT_LONG"] == "TP":
                hi_correct += 1
            elif y_pred[idx] == 0 and row["VERDICT_SHORT"] == "TP":
                hi_correct += 1
        hi_wr = hi_correct / hi_total * 100
        hi_edge = hi_wr / 100 * 1.5 - (1 - hi_wr / 100) * 1.0
        print(f"  >= {threshold*100:.0f}%: {hi_total} trades, WR={hi_wr:.1f}%, Edge={hi_edge:+.3f}R")

    # Save
    ticker = os.path.basename(csv_path).split("_")[0]
    date_str = datetime.now().strftime("%Y%m%d")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost_ga_model.json")
    scaler_path = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost_ga_scaler.pkl")
    features_path = os.path.join(model_dir, f"{ticker}_{date_str}_xgboost_ga_features.txt")

    model.save_model(model_path)
    joblib.dump(scaler, scaler_path)
    with open(features_path, "w") as f:
        f.write("\n".join(selected_features))

    print(f"\nModel saved to: {model_path}")
    print(f"Features saved to: {features_path}")

    # Latest prediction
    print(f"\n{'='*60}")
    print("LATEST CANDLE PREDICTION")
    print(f"{'='*60}")
    latest = df.iloc[-1]
    X_latest = scaler.transform(df[selected_features].iloc[[-1]].values)
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

    return model, scaler, selected_features


def main():
    parser = argparse.ArgumentParser(description="GA-optimized XGBoost for LONG/SHORT prediction")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV")
    parser.add_argument("--population", type=int, default=30, help="Population size")
    parser.add_argument("--generations", type=int, default=30, help="Number of generations")
    parser.add_argument("--mutation-rate", type=float, default=0.05, help="Mutation rate")
    parser.add_argument("--crossover-rate", type=float, default=0.8, help="Crossover rate")
    parser.add_argument("--no-deep", action="store_true", help="Skip LSTM+CNN feature extraction")
    args = parser.parse_args()

    run_ga(args.csv, args.population, args.generations, args.mutation_rate,
           args.crossover_rate, use_deep=not args.no_deep)


if __name__ == "__main__":
    main()
