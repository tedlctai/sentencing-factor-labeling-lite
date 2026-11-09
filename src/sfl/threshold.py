"""Threshold grid, sweep and selection (paper Sec. V-B).

The paper: tau is chosen from a coarse grid {0.50, 0.55, ..., 0.85} and depends
only on the case type and on whether the encoder is pre-trained or fine-tuned
(all variants of one family share one value). The selection data is an explicit
argument of this module; the demo selects on the other folds of the judgments.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from . import metrics
from .config import paper_config
from .labeler import apply_threshold


def grid() -> List[float]:
    t = paper_config()["threshold"]
    n = int(round((t["grid_stop"] - t["grid_start"]) / t["grid_step"])) + 1
    return [round(t["grid_start"] + i * t["grid_step"], 2) for i in range(n)]


def sweep(S: np.ndarray, Y: np.ndarray, taus: Sequence[float] | None = None) -> List[Dict[str, float]]:
    """Metrics at every tau. Use for a threshold-sensitivity table."""
    rows = []
    for tau in (taus if taus is not None else grid()):
        m = metrics.compute(Y, apply_threshold(S, tau))
        rows.append({"tau": tau, **m})
    return rows


def select_shared(
    S_by_variant: Dict[str, np.ndarray],
    Y: np.ndarray,
    taus: Sequence[float] | None = None,
) -> float:
    """One tau shared by several variants (as in Table II), maximising the mean F1.

    Ties go to the smallest tau. ``S_by_variant`` and ``Y`` must come from the
    documents chosen for selection (e.g. a training fold), not from the
    documents used for the final report.
    """
    best_tau, best = None, -1.0
    for tau in (taus if taus is not None else grid()):
        score = float(np.mean([
            metrics.compute(Y, apply_threshold(S, tau))["f1"] for S in S_by_variant.values()
        ]))
        if score > best + 1e-12:
            best_tau, best = tau, score
    return best_tau
