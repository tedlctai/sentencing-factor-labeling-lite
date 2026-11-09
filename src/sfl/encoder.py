"""Text encoders.

``SentenceTransformerEncoder`` is the real encoder (BGE-M3 by default, dense
embeddings with cls pooling, L2-normalised, so the dot product is the cosine).
``HashEncoder`` is a deterministic character n-gram encoder used only for
tests and offline smoke runs. It is NOT a substitute for BGE-M3 and any
numbers produced with it carry no meaning for the paper.
"""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Sequence

import numpy as np


class HashEncoder:
    """Deterministic char 1-3gram hashing encoder (tests / smoke runs only)."""

    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for n in (1, 2, 3):
                for j in range(len(t) - n + 1):
                    h = zlib.crc32(t[j : j + n].encode("utf-8"))
                    out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
                    out[i, (h >> 8) % self.dim] += 0.5
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEncoder:
    """Dense encoder via sentence-transformers (default: BAAI/bge-m3)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        max_seq_length: int | None = 8192,
        device: str | None = None,
        fp16: bool = True,
    ):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.name = model_name
        local = Path(model_name)
        if local.is_dir() and not (local / "modules.json").exists():
            # A plain transformers directory (as written by sfl.train): build the
            # same pipeline as BGE-M3 explicitly, cls pooling.
            from sentence_transformers import models

            tr = models.Transformer(str(local))
            dim_fn = getattr(tr, "get_embedding_dimension", None) or tr.get_word_embedding_dimension
            pool = models.Pooling(dim_fn(), pooling_mode="cls")
            self.model = SentenceTransformer(modules=[tr, pool], device=device)
        else:
            self.model = SentenceTransformer(model_name, device=device)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        if fp16 and str(self.model.device).startswith("cuda"):
            self.model.half()

    def encode(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)


def make_encoder(kind: str = "BAAI/bge-m3", **kwargs):
    """``"hash"`` gives the test encoder, anything else is a model name."""
    if kind == "hash":
        return HashEncoder()
    return SentenceTransformerEncoder(kind, **kwargs)
