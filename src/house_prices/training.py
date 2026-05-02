"""Training and evaluation utilities for the House Prices pipeline."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score


def evaluate_cv(X: pd.DataFrame, y: pd.Series, models: dict, n_splits: int, seed: int) -> pd.DataFrame:
    """Compute leakage-safe CV scores for each candidate model."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        positive = -scores
        rows.append({"model": name, "rmse_log_mean": float(positive.mean()), "rmse_log_std": float(positive.std())})
    return pd.DataFrame(rows).sort_values("rmse_log_mean")


def get_oof_predictions(X: pd.DataFrame, y: pd.Series, models: dict, n_splits: int, seed: int) -> tuple[np.ndarray, list[str]]:
    """Return out-of-fold predictions for every model in the model zoo."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    names = list(models.keys())
    oof = np.zeros((len(X), len(names)), dtype=float)
    for tr_idx, va_idx in cv.split(X, y):
        x_tr, x_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        for i, name in enumerate(names):
            model = clone(models[name])
            model.fit(x_tr, y_tr)
            oof[va_idx, i] = model.predict(x_va)
    return oof, names


def tune_blend_weights(oof_pred: np.ndarray, y: pd.Series, n_trials: int) -> list[float]:
    """Search for blend weights that minimize OOF RMSE.

    Falls back to uniform weights if Optuna is unavailable.
    """
    try:
        import optuna
    except Exception:
        weights = np.ones(oof_pred.shape[1], dtype=float)
        weights /= weights.sum()
        return weights.tolist()

    y_arr = y.values

    def objective(trial: optuna.Trial) -> float:
        raw = [trial.suggest_float(f"w{i}", 0.01, 1.0) for i in range(oof_pred.shape[1])]
        weights = np.array(raw, dtype=float)
        weights /= weights.sum()
        blend = oof_pred @ weights
        return float(np.sqrt(mean_squared_error(y_arr, blend)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = np.array([study.best_params[f"w{i}"] for i in range(oof_pred.shape[1])], dtype=float)
    best /= best.sum()
    return best.tolist()


def weighted_prediction(pred_map: dict[str, np.ndarray], weights_map: dict[str, float]) -> np.ndarray:
    """Blend model predictions with explicit model-name weights."""
    arr = None
    for name, weight in weights_map.items():
        part = pred_map[name] * float(weight)
        arr = part if arr is None else arr + part
    return arr


def save_submission(pred_log: np.ndarray, passenger_id: pd.Series, out_dir: Path, filename: str) -> Path:
    """Convert log-price predictions back to price space and save a Kaggle submission."""
    pred = np.expm1(pred_log)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{filename}_{ts}.csv"
    pd.DataFrame({"Id": passenger_id, "SalePrice": pred}).to_csv(path, index=False)
    return path


def compute_oof_metrics(oof_pred: np.ndarray, y: pd.Series, names: list[str]) -> pd.DataFrame:
    """Summarize per-model OOF RMSE in a tidy table."""
    rows = []
    y_arr = y.values
    for i, name in enumerate(names):
        rmse = float(np.sqrt(mean_squared_error(y_arr, oof_pred[:, i])))
        rows.append({"model": name, "oof_rmse_log": rmse})
    return pd.DataFrame(rows).sort_values("oof_rmse_log")


def save_oof_predictions(oof_pred: np.ndarray, names: list[str], ids: pd.Series, out_dir: Path, prefix: str = "oof") -> Path:
    """Persist OOF predictions so the stack/blend can be audited later."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(oof_pred, columns=names, index=ids.index)
    df.insert(0, "Id", ids.values)
    path = out_dir / f"{prefix}_predictions_{ts}.csv"
    df.to_csv(path, index=False)
    return path


def save_test_predictions(pred_map: dict[str, np.ndarray], ids: pd.Series, out_dir: Path, prefix: str = "test") -> Path:
    """Persist per-model test predictions for downstream blending and analysis."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame({"Id": ids.values})
    for name, arr in pred_map.items():
        df[name] = arr
    path = out_dir / f"{prefix}_predictions_{ts}.csv"
    df.to_csv(path, index=False)
    return path


def save_metrics(metrics: dict, out_dir: Path, filename: str = "metrics") -> Path:
    """Save a small JSON blob with the run-level metrics and summary values."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{filename}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path