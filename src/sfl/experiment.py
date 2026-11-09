"""Evaluate the labeling variants of Table II that need no fine-tuning, and
format the results. Fine-tuned (SFT) variants use the same code with a
fine-tuned encoder."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from . import metrics, splits, threshold
from .labeler import FactorLabeler, apply_threshold

VARIANT_UNITS = {"w/o chunking": "whole", "w/ Chunking": "chunk", "w/ CUA": "cua"}


def similarity_by_variant(encoder, texts: Sequence[str], factors: Sequence[str], cua=None,
                          batch_size: int = 16) -> Dict[str, np.ndarray]:
    out = {}
    for name, unit in VARIANT_UNITS.items():
        if unit == "cua" and cua is None:
            continue
        out[name] = FactorLabeler(encoder, factors, unit=unit, cua=cua, batch_size=batch_size).max_similarity(texts)
    return out


def evaluate_variants(S_by_variant: Dict[str, np.ndarray], Y: np.ndarray, tau: float) -> List[dict]:
    rows = []
    for name, S in S_by_variant.items():
        rows.append({"variant": name, "tau": tau, **metrics.compute(Y, apply_threshold(S, tau))})
    return rows


def crossfit_report(S_by_variant: Dict[str, np.ndarray], Y: np.ndarray, ids: Sequence[str],
                    taus=None, k: int = 2, seed: int = 0):
    """k-fold report in which every judgment is labeled with a threshold chosen on
    the OTHER folds only.

    For each fold: choose one tau shared by all variants on the remaining folds
    (threshold.select_shared), label the held-out fold with it, then pool the
    predictions of all folds and compute the metrics once over every judgment.

    Returns (rows, pooled predictions per variant, tau used per fold).
    """
    pos = {i: n for n, i in enumerate(ids)}
    preds = {v: np.zeros_like(Y, dtype=np.int8) for v in S_by_variant}
    taus_used = []
    for train_ids, test_ids in splits.kfold(ids, k=k, seed=seed):
        splits.assert_disjoint(train_ids, test_ids)
        tr = [pos[i] for i in train_ids]
        te = [pos[i] for i in test_ids]
        tau = threshold.select_shared({v: S[tr] for v, S in S_by_variant.items()}, Y[tr], taus)
        taus_used.append(tau)
        for v, S in S_by_variant.items():
            preds[v][te] = apply_threshold(S[te], tau)
    tau_str = "/".join(f"{t:.2f}" for t in taus_used)
    rows = [{"variant": v, "tau": tau_str, **metrics.compute(Y, preds[v])} for v in S_by_variant]
    return rows, preds, taus_used


def markdown_table(rows: List[dict], title_col: str = "variant") -> str:
    cols = [title_col, "tau", "accuracy", "precision", "recall", "f1", "hamming_loss"]
    head = "| " + " | ".join(["Method", "tau", "Acc (%)", "Prec (%)", "Rec (%)", "F1 (%)", "Hamming (%)"]) + " |"
    sep = "|" + "---|" * len(cols)
    lines = [head, sep]
    for r in rows:
        tau = r["tau"] if isinstance(r["tau"], str) else f"{r['tau']:.2f}"
        cells = [str(r[title_col]), tau] + [f"{r[c]:.2f}" for c in cols[2:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
