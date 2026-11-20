"""Adaptive Alpha ridge predictor: model loading, scoring and research labels.

Predictions are 21-trading-day EXCESS returns vs SPY, not expected stock returns.
A candidate label is a comparative research indicator, not personalized advice.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_VERSION = "excess-ridge-1.0"
FEATURES = {
    "momentum": ("momentum",),
    "recovery": ("recovery",),
    "hybrid": ("momentum", "recovery", "interaction"),
}

def features(momentum: float, recovery: float, strategy: str):
    values = {
        "momentum": float(momentum),
        "recovery": float(recovery),
        "interaction": float(momentum) * float(recovery) / 100.0,
    }
    return np.array([values[key] for key in FEATURES[strategy]], dtype=float)

def fit_model(X, y, alpha=20.0):
    """Ridge regression, standardized by TRAINING observations only."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or len(X) != len(y) or len(y) < 50:
        raise ValueError("Need at least 50 aligned observations")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite model input")
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Z = (X - mu) / sd
    target_mean = float(y.mean())
    weights = np.linalg.solve(Z.T @ Z + alpha * np.eye(X.shape[1]), Z.T @ (y - target_mean))
    return {"mean": mu.tolist(), "std": sd.tolist(),
            "weights": weights.tolist(), "intercept": target_mean,
            "n_train": len(y), "alpha": alpha}

def predict_model(model, X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    m = np.array(model["mean"])
    sd = np.array(model["std"])
    w = np.array(model["weights"])
    if X.shape[1] != len(w):
        raise ValueError("Model feature count mismatch")
    return float(model["intercept"]) + ((X - m) / sd) @ w

def load_artifact(path="models/predictive_model.json"):
    file = Path(path)
    if not file.exists():
        return None
    data = json.loads(file.read_text(encoding="utf-8"))
    if data.get("model_version") != MODEL_VERSION:
        raise ValueError("Unsupported predictive model version")
    return data

def research_signal(predicted_excess_return: float, threshold: float = 0.02):
    """3-way signal vs SPY. Never present as a certainty or trading instruction."""
    if predicted_excess_return >= threshold:
        return "BUY CANDIDATE"
    if predicted_excess_return <= -threshold:
        return "SELL / AVOID CANDIDATE"
    return "HOLD / NEUTRAL"
