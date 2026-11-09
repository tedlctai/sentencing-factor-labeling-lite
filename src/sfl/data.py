"""Dataset loading. The expected files are described in data/SCHEMA.md.

    judgments.jsonl         {"id", "case_type", "fact"}
    factors.json            {"<case_type>": ["factor text", ...]}
    reference_labels.jsonl  {"id", "labels": {"factor text": 0 | 1}}
    cua_passages.jsonl      {"text", "label": "important" | "unimportant"}

``fact`` is the Fact of the Case section; the paper's "judgment" refers to it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def load_jsonl(path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class Dataset:
    case_type: str
    ids: List[str]
    texts: List[str]
    factors: List[str]
    Y: np.ndarray                      # (n_judgments, n_factors) reference labels
    passages: List[Tuple[str, str]]    # (text, label) for CRF training

    def subset(self, ids) -> "Dataset":
        pos = {i: k for k, i in enumerate(self.ids)}
        rows = [pos[i] for i in ids]
        return Dataset(
            self.case_type,
            [self.ids[r] for r in rows],
            [self.texts[r] for r in rows],
            self.factors,
            self.Y[rows],
            self.passages,
        )


def load_dataset(root, case_type: Optional[str] = None) -> Dataset:
    root = Path(root)
    judgments = load_jsonl(root / "judgments.jsonl")
    factors_by_type = json.loads((root / "factors.json").read_text(encoding="utf-8"))
    if case_type is None:
        case_type = sorted(factors_by_type)[0]
    factors = factors_by_type[case_type]
    labels = {r["id"]: r["labels"] for r in load_jsonl(root / "reference_labels.jsonl")}
    rows = [j for j in judgments if j["case_type"] == case_type]
    missing = [j["id"] for j in rows if j["id"] not in labels]
    if missing:
        raise ValueError(f"{len(missing)} judgments have no reference labels, e.g. {missing[:3]}")
    Y = np.array([[labels[j["id"]][f] for f in factors] for j in rows], dtype=np.int8)
    passages = []
    p = root / "cua_passages.jsonl"
    if p.exists():
        passages = [(r["text"], r["label"]) for r in load_jsonl(p)]
    return Dataset(case_type, [j["id"] for j in rows], [j["fact"] for j in rows], factors, Y, passages)


def load_split(root, name: str = "sft") -> Optional[dict]:
    """``splits/<name>.json`` -> {"train": [ids], "test": [ids]}, or None if absent."""
    path = Path(root) / "splits" / f"{name}.json"
    if str(name).endswith(".json") and Path(name).exists():     # a split file given by path
        path = Path(name)
    if not path.exists():
        return None
    split = json.loads(path.read_text(encoding="utf-8"))
    overlap = set(split["train"]) & set(split["test"])
    if overlap:
        raise ValueError(f"split '{name}' is not disjoint: {len(overlap)} shared ids")
    return split
