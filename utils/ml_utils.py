"""
ml_utils.py
Core statistical and machine-learning helpers for the RevIN learning repo.
All functions are pure NumPy/Python so notebooks stay clean.
"""

from __future__ import annotations

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Statistics primitives
# ══════════════════════════════════════════════════════════════════════════════

def calculate_mean(x: np.ndarray) -> float:
    """Arithmetic mean: sum(x) / n."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x) / len(x))


def calculate_variance(x: np.ndarray, ddof: int = 0) -> float:
    """
    Population variance (ddof=0) or sample variance (ddof=1).
    Formula: sum((xi - mean)^2) / (n - ddof)
    """
    x = np.asarray(x, dtype=float)
    mu = calculate_mean(x)
    return float(np.sum((x - mu) ** 2) / (len(x) - ddof))


def calculate_std(x: np.ndarray, ddof: int = 0) -> float:
    """Standard deviation = sqrt(variance)."""
    return float(np.sqrt(calculate_variance(x, ddof=ddof)))


# ══════════════════════════════════════════════════════════════════════════════
# Normalization
# ══════════════════════════════════════════════════════════════════════════════

def z_score_normalize(x: np.ndarray, ddof: int = 0) -> tuple[np.ndarray, dict]:
    """
    Standardise to zero mean and unit variance.
    Returns (z_scores, params) where params holds mean and std for inversion.
    """
    x = np.asarray(x, dtype=float)
    mu  = calculate_mean(x)
    std = calculate_std(x, ddof=ddof)
    if std == 0:
        std = 1.0  # avoid division by zero on constant series
    params = {"mean": mu, "std": std, "method": "zscore"}
    return (x - mu) / std, params


def minmax_normalize(x: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Scale to [0, 1].
    Returns (scaled, params) where params holds min and max for inversion.
    """
    x = np.asarray(x, dtype=float)
    xmin, xmax = x.min(), x.max()
    rng = xmax - xmin if xmax != xmin else 1.0
    params = {"min": xmin, "max": xmax, "range": rng, "method": "minmax"}
    return (x - xmin) / rng, params


def denormalize_data(x_norm: np.ndarray, params: dict) -> np.ndarray:
    """
    Invert z-score or min-max normalisation using stored params.
    """
    x_norm = np.asarray(x_norm, dtype=float)
    method = params.get("method", "zscore")
    if method == "zscore":
        return x_norm * params["std"] + params["mean"]
    elif method == "minmax":
        return x_norm * params["range"] + params["min"]
    else:
        raise ValueError(f"Unknown method: {method}")


# ══════════════════════════════════════════════════════════════════════════════
# Time series utilities
# ══════════════════════════════════════════════════════════════════════════════

def create_sequences(
    data: np.ndarray,
    lookback: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over *data* to build (X, y) pairs.

    X shape: (n_samples, lookback)
    y shape: (n_samples, horizon)
    """
    data = np.asarray(data, dtype=float)
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + horizon])
    return np.array(X), np.array(y)


def train_test_split_time_series(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological split — no shuffling, respects temporal order.
    Returns (X_train, X_test, y_train, y_test).
    """
    split = int(len(X) * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]


# ══════════════════════════════════════════════════════════════════════════════
# RevIN — Reversible Instance Normalization
# ══════════════════════════════════════════════════════════════════════════════

class RevIN:
    """
    Reversible Instance Normalization (Kim et al., 2022).

    Each instance (window) is independently normalised before being
    fed to the forecasting model, then denormalised after prediction.
    This makes the model robust to distribution shift across windows.

    Usage
    -----
    revin = RevIN(eps=1e-5)
    x_norm = revin.normalize(x_window)   # before model
    y_pred = model.predict(x_norm)
    y_actual = revin.denormalize(y_pred)  # after model
    """

    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        self._mean: float | None = None
        self._std:  float | None = None

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Subtract instance mean and divide by instance std."""
        x = np.asarray(x, dtype=float)
        self._mean = float(x.mean())
        self._std  = float(x.std()) + self.eps
        return (x - self._mean) / self._std

    def denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        """Reverse the normalisation stored from the last normalize() call."""
        if self._mean is None or self._std is None:
            raise RuntimeError("Call normalize() before denormalize().")
        x_norm = np.asarray(x_norm, dtype=float)
        return x_norm * self._std + self._mean

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Alias: normalize and store stats."""
        return self.normalize(x)

    def inverse_transform(self, x_norm: np.ndarray) -> np.ndarray:
        """Alias: denormalize using stored stats."""
        return self.denormalize(x_norm)

    @property
    def stats(self) -> dict:
        return {"mean": self._mean, "std": self._std}


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))
