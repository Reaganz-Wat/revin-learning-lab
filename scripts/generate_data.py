"""
generate_data.py
Generates synthetic datasets for the RevIN learning repository.
Run once to populate the /data directory.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 1. fundamentals_data.csv
# ─────────────────────────────────────────────

def generate_fundamentals(n=50):
    df = pd.DataFrame({
        "student_score": np.clip(np.random.normal(72, 12, n), 20, 100).round(1),
        "temperature":   np.clip(np.random.normal(22, 5,  n), 5,  40).round(1),
        "sales":         np.clip(np.random.normal(500, 120, n), 100, 900).round(0).astype(int),
        "stock_price":   np.clip(np.random.normal(150, 30, n), 50, 300).round(2),
    })
    return df


# ─────────────────────────────────────────────
# 2. stock_sample.csv  (realistic synthetic OHLCV)
# ─────────────────────────────────────────────

def _generate_price_series(n, start_price, drift, volatility):
    """Geometric Brownian Motion."""
    dt = 1 / 252
    returns = np.random.normal(drift * dt, volatility * np.sqrt(dt), n)
    prices = start_price * np.exp(np.cumsum(returns))
    return prices


def generate_stock(n=120, start_price=100.0, drift=0.08, volatility=0.2):
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    close = _generate_price_series(n, start_price, drift, volatility)
    daily_range = close * np.random.uniform(0.005, 0.025, n)
    high  = (close + daily_range * np.random.uniform(0.3, 0.7, n)).round(2)
    low   = (close - daily_range * np.random.uniform(0.3, 0.7, n)).round(2)
    open_ = (close + np.random.uniform(-0.5, 0.5, n) * daily_range * 0.4).round(2)
    volume = (np.random.normal(1_500_000, 300_000, n)).clip(500_000).astype(int)

    df = pd.DataFrame({
        "date":   dates.strftime("%Y-%m-%d"),
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close.round(2),
        "volume": volume,
    })
    return df


# ─────────────────────────────────────────────
# 3. shifted_stock_sample.csv  (distribution shift)
# ─────────────────────────────────────────────

def generate_shifted_stock(n=120):
    # Higher starting price (mean shift) and higher volatility (variance shift)
    return generate_stock(n, start_price=180.0, drift=0.05, volatility=0.35)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    fund = generate_fundamentals()
    fund.to_csv(DATA_DIR / "fundamentals_data.csv", index=False)
    print(f"fundamentals_data.csv  → {len(fund)} rows")

    stock = generate_stock()
    stock.to_csv(DATA_DIR / "stock_sample.csv", index=False)
    print(f"stock_sample.csv       → {len(stock)} rows")

    shifted = generate_shifted_stock()
    shifted.to_csv(DATA_DIR / "shifted_stock_sample.csv", index=False)
    print(f"shifted_stock_sample.csv → {len(shifted)} rows")

    print("\nAll datasets written to:", DATA_DIR)
