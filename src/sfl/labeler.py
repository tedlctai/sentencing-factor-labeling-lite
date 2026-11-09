"""Thresholded factor labeling (paper Sec. IV-A).

    L(J, f) = 1  if  max over chunks c of sim(c, f) >= tau,  else 0

``unit`` selects what the "chunks" are, which gives the variants of Table II:
    "whole"  the whole fact text is one unit        (BGE-M3 / SFT rows)
    "chunk"  heading-based chunks                   (w/ Chunking rows)
    "cua"    heading-based chunks kept by the CUA   (w/ CUA rows)

The maximum similarity matrix is computed once; thresholds are applied
afterwards, so sweeping tau is cheap.
"""
from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np

from .chunking import chunk_text

UNITS = ("whole", "chunk", "cua")
NEG_INF = -np.inf


class FactorLabeler:
    def __init__(
        self,
        encoder,
        factors: Sequence[str],
        unit: str = "chunk",
        cua=None,
        chunker: Callable[[str], List[str]] = chunk_text,
        batch_size: int = 16,
    ):
        if unit not in UNITS:
            raise ValueError(f"unit must be one of {UNITS}")
        if unit == "cua" and cua is None:
            raise ValueError("unit='cua' needs a trained CUA")
        self.encoder = encoder
        self.factors = list(factors)
        self.unit = unit
        self.cua = cua
        self.chunker = chunker
        self.batch_size = batch_size

    def units_of(self, text: str) -> List[str]:
        if self.unit == "whole":
            return [text]
        chunks = self.chunker(text)
        return self.cua.filter(chunks) if self.unit == "cua" else chunks

    def max_similarity(self, texts: Sequence[str]) -> np.ndarray:
        """(n_docs, n_factors) matrix of max cosine similarity.

        A document with no retained unit gets -inf, i.e. every factor absent
        (strict reading of L; configs/impl_defaults.yaml: empty_after_cua).
        """
        factor_emb = self.encoder.encode(self.factors, batch_size=self.batch_size)
        all_units: List[str] = []
        owner: List[int] = []
        for i, t in enumerate(texts):
            for u in self.units_of(t):
                all_units.append(u)
                owner.append(i)
        S = np.full((len(texts), len(self.factors)), NEG_INF, dtype=np.float64)
        if not all_units:
            return S
        unit_emb = self.encoder.encode(all_units, batch_size=self.batch_size)
        sims = unit_emb @ factor_emb.T          # cosine, embeddings are normalised
        owner_arr = np.asarray(owner)
        for i in range(len(texts)):
            rows = sims[owner_arr == i]
            if rows.size:
                S[i] = rows.max(axis=0)
        return S


def apply_threshold(S: np.ndarray, tau: float) -> np.ndarray:
    """Binary label matrix: 1 where max similarity >= tau."""
    return (S >= tau).astype(np.int8)
