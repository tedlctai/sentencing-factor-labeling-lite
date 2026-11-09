"""Fine-tuning data (paper Sec. IV-C), built offline from judgments that are
reserved for fine-tuning:

    step 1  split each judgment into chunks and keep those the CUA retains
    step 2  for every (judgment, factor): top-k retained chunks by pre-trained similarity
      -> a verifier (LLM) decides for each of them whether it supports the factor
      -> verified chunks are positives for that factor
      -> negatives are random chunks of other judgments
      -> one training entry {"query", "pos", "neg": [group_size - 1 chunks]}

Rejected candidates are discarded; negatives are random chunks of other judgments that were not accepted
as positives for the factor, as in standard DPR training.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

from .chunking import chunk_text
from .config import impl_defaults


def candidate_chunks(text: str, cua=None) -> List[str]:
    """Paper Sec. IV-C step 1: chunks of a judgment, with low-value ones removed by the CUA.
    ``cua=None`` keeps every chunk (a deviation from the paper, for ablations only)."""
    chunks = chunk_text(text)
    return cua.filter(chunks) if cua is not None else chunks


def verify_top_k(encoder, texts: Sequence[str], ids: Sequence[str], factors: Sequence[str],
                 verifier: Callable[[str, str], bool], top_k: int = 5,
                 batch_size: int = 16, cua=None) -> List[dict]:
    """Records {"id", "factor", "chunk", "sim", "relevant"} for the top-k retained chunks of every pair."""
    f_emb = encoder.encode(list(factors), batch_size=batch_size)
    records = []
    for jid, text in zip(ids, texts):
        chunks = candidate_chunks(text, cua)
        if not chunks:
            continue
        c_emb = encoder.encode(chunks, batch_size=batch_size)
        sims = c_emb @ f_emb.T
        for j, factor in enumerate(factors):
            for r in np.argsort(-sims[:, j], kind="stable")[:top_k]:
                records.append({"id": jid, "factor": factor, "chunk": chunks[r],
                                "sim": float(sims[r, j]), "relevant": bool(verifier(factor, chunks[r]))})
    return records


def build_entries(records: Sequence[dict], all_chunks: Dict[str, List[str]], group_size: int = 16,
                  entries_per_positive: int | None = None, seed: int = 0) -> List[dict]:
    """Turn verification records into training entries. ``all_chunks`` maps a judgment id
    to all its chunks (used for the random negatives)."""
    d = impl_defaults()["finetune_data"]
    entries_per_positive = entries_per_positive or d["entries_per_positive"]
    rng = random.Random(seed)

    by_factor: Dict[str, List[dict]] = {}
    for r in records:
        by_factor.setdefault(r["factor"], []).append(r)

    entries = []
    for factor, recs in by_factor.items():
        positives = [r for r in recs if r["relevant"]]
        pos_texts = {r["chunk"] for r in positives}
        for pos in positives:
            own = pos["id"]
            random_pool = list(dict.fromkeys(c for jid, cs in all_chunks.items() if jid != own
                                             for c in cs if c not in pos_texts))
            for _ in range(entries_per_positive):
                neg = rng.sample(random_pool, min(group_size - 1, len(random_pool)))
                # pad by repeating when the pool is smaller than a group (tiny data only)
                while len(neg) < group_size - 1 and neg:
                    neg.append(rng.choice(neg))
                if neg:
                    entries.append({"query": factor, "pos": [pos["chunk"]], "neg": neg[: group_size - 1],
                                    "source_id": own})
    rng.shuffle(entries)
    return entries


def split_entries(entries: Sequence[dict], dev_fraction: float | None = None, seed: int = 0):
    """Entry-level train / dev split, as Table I of the paper (Train 90% / Test 10% of the
    entries). Entries of the same judgment can land on both sides, so the dev part is only a
    monitor for the fine-tuning; for unseen judgments, use the judgment-level split of the
    demo data."""
    frac = impl_defaults()["finetune_data"]["dev_fraction"] if dev_fraction is None else dev_fraction
    idx = list(range(len(entries)))
    random.Random(seed).shuffle(idx)
    n_dev = int(round(frac * len(entries)))
    if len(entries) >= 2 and frac > 0:
        n_dev = max(1, n_dev)
    dev_idx = set(idx[:n_dev])
    train = [e for i, e in enumerate(entries) if i not in dev_idx]
    dev = [e for i, e in enumerate(entries) if i in dev_idx]
    return train, dev


def table1_summary(entries: Sequence[dict], train: Sequence[dict], dev: Sequence[dict],
                   n_judgments: int, n_factors: int, group_size: int) -> dict:
    """The quantities Table I reports for the fine-tuning data, for the data built here."""
    pairs = n_judgments * n_factors
    return {
        "judgments_in_pool": n_judgments, "factors": n_factors,
        "entries_total": len(entries), "entries_train": len(train), "entries_dev": len(dev),
        "train_share": round(len(train) / len(entries), 4) if entries else None,
        "pos_neg": f"1:{group_size - 1}",
        "entries_per_judgment_factor_pair": round(len(entries) / pairs, 3) if pairs else None,
        "distinct_judgments_with_entries": len({e["source_id"] for e in entries}),
    }


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
