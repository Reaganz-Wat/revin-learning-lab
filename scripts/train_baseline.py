"""
train_baseline.py
Train a RandomForest baseline forecasting model (no RevIN) on stock_sample.csv.
Saves the fitted model to models/baseline_model.pkl.
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ml_utils import create_sequences, train_test_split_time_series, rmse, mae

ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data" / "stock_sample.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

LOOKBACK = 20
HORIZON  = 1


def main():
    df    = pd.read_csv(DATA)
    close = df["close"].values.astype(float)

    X, y = create_sequences(close, lookback=LOOKBACK, horizon=HORIZON)
    y    = y.ravel()

    X_tr, X_te, y_tr, y_te = train_test_split_time_series(X, y, test_ratio=0.2)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    print(f"Baseline  RMSE : {rmse(y_te, y_pred):.4f}")
    print(f"Baseline  MAE  : {mae(y_te, y_pred):.4f}")

    with open(MODELS / "baseline_model.pkl", "wb") as f:
        pickle.dump({"model": model, "lookback": LOOKBACK, "horizon": HORIZON}, f)
    print("Saved → models/baseline_model.pkl")


if __name__ == "__main__":
    main()
