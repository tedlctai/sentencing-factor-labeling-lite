"""Label-wise metrics over (judgment, factor) pairs (paper Sec. V-C).

Accuracy, Precision, Recall and F1 are computed over all pairs (positive class
= factor present). Hamming Loss is the fraction of wrongly labeled pairs; it
equals 1 - Accuracy by definition, and both come from the same confusion
counts. Values are returned in percent.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    yt = np.asarray(y_true).astype(bool)
    yp = np.asarray(y_pred).astype(bool)
    return {
        "tp": int((yt & yp).sum()),
        "fp": int((~yt & yp).sum()),
        "fn": int((yt & ~yp).sum()),
        "tn": int((~yt & ~yp).sum()),
    }


def _safe(a: float, b: float) -> float:
    return a / b if b else 0.0


def from_confusion(c: Dict[str, int]) -> Dict[str, float]:
    n = c["tp"] + c["fp"] + c["fn"] + c["tn"]
    prec = _safe(c["tp"], c["tp"] + c["fp"])
    rec = _safe(c["tp"], c["tp"] + c["fn"])
    f1 = _safe(2 * prec * rec, prec + rec)
    acc = _safe(c["tp"] + c["tn"], n)
    ham = _safe(c["fp"] + c["fn"], n)
    return {
        "accuracy": 100 * acc,
        "precision": 100 * prec,
        "recall": 100 * rec,
        "f1": 100 * f1,
        "hamming_loss": 100 * ham,
    }


def compute(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return from_confusion(confusion(y_true, y_pred))


def per_factor(y_true: np.ndarray, y_pred: np.ndarray, factors: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for j, f in enumerate(factors):
        c = confusion(y_true[:, j], y_pred[:, j])
        m = from_confusion(c)
        m["support"] = c["tp"] + c["fn"]
        out[f] = m
    return out


def paired_bootstrap_diff(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
    n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> Dict[str, tuple]:
    """Metric(a) - Metric(b) with a percentile CI; judgments (rows) are resampled
    together for both predictions, so the comparison is paired.
    Returns {metric: (difference, low, high)}."""
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    keys = ("accuracy", "precision", "recall", "f1", "hamming_loss")
    point = {k: compute(y_true, pred_a)[k] - compute(y_true, pred_b)[k] for k in keys}
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ma, mb = compute(y_true[idx], pred_a[idx]), compute(y_true[idx], pred_b[idx])
        for k in keys:
            draws[k].append(ma[k] - mb[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (point[k], float(np.percentile(draws[k], lo)), float(np.percentile(draws[k], hi))) for k in keys}


def bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> Dict[str, tuple]:
    """Percentile CI, resampling judgments (rows) with replacement."""
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    draws = {k: [] for k in ("accuracy", "precision", "recall", "f1", "hamming_loss")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = compute(y_true[idx], y_pred[idx])
        for k in draws:
            draws[k].append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in draws.items()}
