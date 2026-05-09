"""
train_revin.py
Train a RandomForest model WITH RevIN normalization on stock_sample.csv.
Tests generalisation on shifted_stock_sample.csv to show RevIN's advantage.
Saves the fitted model to models/revin_model.pkl.
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ml_utils import (
    RevIN, create_sequences, train_test_split_time_series, rmse, mae
)

ROOT    = Path(__file__).parent.parent
DATA    = ROOT / "data" / "stock_sample.csv"
SHIFTED = ROOT / "data" / "shifted_stock_sample.csv"
MODELS  = ROOT / "models"
MODELS.mkdir(exist_ok=True)

LOOKBACK = 20
HORIZON  = 1


def apply_revin_to_sequences(X: np.ndarray, y: np.ndarray):
    """
    Apply per-instance RevIN normalisation.
    Each row of X is normalised independently; y is denormalised using the
    same stats so the model learns normalised→normalised mappings.
    """
    X_norm  = np.zeros_like(X)
    y_norm  = np.zeros_like(y)
    revins  = []

    for i in range(len(X)):
        r = RevIN()
        X_norm[i] = r.normalize(X[i])
        # shift y into the normalised space using window statistics
        y_norm[i] = (y[i] - r.stats["mean"]) / r.stats["std"]
        revins.append(r)

    return X_norm, y_norm, revins


def evaluate_with_revin(model, X: np.ndarray, y_true: np.ndarray, label: str):
    preds = []
    for i in range(len(X)):
        r      = RevIN()
        x_norm = r.normalize(X[i])
        y_norm = model.predict(x_norm.reshape(1, -1))[0]
        y_hat  = r.denormalize(np.array([y_norm]))[0]
        preds.append(y_hat)

    preds = np.array(preds)
    print(f"{label}  RMSE : {rmse(y_true, preds):.4f}")
    print(f"{label}  MAE  : {mae(y_true, preds):.4f}")
    return preds


def main():
    # ── Training data ────────────────────────────────────────────────────────
    df    = pd.read_csv(DATA)
    close = df["close"].values.astype(float)

    X, y  = create_sequences(close, lookback=LOOKBACK, horizon=HORIZON)
    y     = y.ravel()

    X_tr, X_te, y_tr, y_te = train_test_split_time_series(X, y, test_ratio=0.2)

    X_tr_n, y_tr_n, _ = apply_revin_to_sequences(X_tr, y_tr)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_tr_n, y_tr_n)

    # ── Evaluation: same distribution ────────────────────────────────────────
    evaluate_with_revin(model, X_te, y_te, "RevIN (same dist)")

    # ── Evaluation: shifted distribution ────────────────────────────────────
    df_sh   = pd.read_csv(SHIFTED)
    close_s = df_sh["close"].values.astype(float)
    X_s, y_s = create_sequences(close_s, lookback=LOOKBACK, horizon=HORIZON)
    y_s = y_s.ravel()
    evaluate_with_revin(model, X_s, y_s, "RevIN (shifted dist)")

    # ── Save ─────────────────────────────────────────────────────────────────
    with open(MODELS / "revin_model.pkl", "wb") as f:
        pickle.dump({"model": model, "lookback": LOOKBACK, "horizon": HORIZON}, f)
    print("Saved → models/revin_model.pkl")


if __name__ == "__main__":
    main()
