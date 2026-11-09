"""Chunk Utility Assessment (paper Sec. IV-B).

Pipeline: sentence embeddings -> k-means (k = 8) -> cluster-id sequence ->
linear-chain CRF with features {current, prev, next, is_first, is_last} ->
chunk type by argmax (majority vote) over per-sentence predictions.

Settings in configs/impl_defaults.yaml:
sentence separators, minimum sentence length, CRF regularisation, how the
passage-level labels are turned into sentence labels (here: every sentence of a
passage gets the passage label), and the tie-break of the vote (here: keep the
chunk). Cluster ids are used as-is; naming clusters is optional metadata and
plays no role in the CRF.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .chunking import split_sentences
from .config import impl_defaults

IMPORTANT = "important"
UNIMPORTANT = "unimportant"


def sentence_features(cluster_ids: Sequence[int], i: int) -> dict:
    """The five features of the paper: current, prev, next, is_first, is_last."""
    n = len(cluster_ids)
    feats = {"current": str(cluster_ids[i]), "is_first": i == 0, "is_last": i == n - 1}
    if i > 0:
        feats["prev"] = str(cluster_ids[i - 1])
    if i < n - 1:
        feats["next"] = str(cluster_ids[i + 1])
    return feats


def sequence_features(cluster_ids: Sequence[int]) -> List[dict]:
    return [sentence_features(cluster_ids, i) for i in range(len(cluster_ids))]


def vote(labels: Sequence[str], tie_break: str = IMPORTANT) -> str:
    """argmax over label counts; ties go to ``tie_break``."""
    if not labels:
        return UNIMPORTANT
    counts = Counter(labels)
    best = max(counts.values())
    winners = [lab for lab, c in counts.items() if c == best]
    return tie_break if tie_break in winners else sorted(winners)[0]


class CUA:
    def __init__(self, encoder, n_clusters: int = 8, **overrides):
        d = {**impl_defaults()["cua"], **overrides}
        self.encoder = encoder
        self.n_clusters = n_clusters
        self.separators = d["sentence_separators"]
        self.min_len = d["min_sentence_len"]
        self.seed = d["kmeans_seed"]
        self.n_init = d["kmeans_n_init"]
        self.c1, self.c2 = d["crf_c1"], d["crf_c2"]
        self.max_iter = d["crf_max_iterations"]
        self.all_transitions = d["crf_all_possible_transitions"]
        self.tie_break = d["vote_tie_break"]
        self.centers: np.ndarray | None = None
        self.cluster_names: dict = {}      # optional, informational only
        self._model_path: str | None = None
        self._tagger = None

    # ------------------------------------------------------------ sentences
    def _sentences(self, text: str) -> List[str]:
        return split_sentences(text, self.separators, self.min_len)

    # ------------------------------------------------------------- k-means
    def fit_clusters(self, texts: Sequence[str]) -> "CUA":
        """Unsupervised: cluster the sentences of a corpus of one case type."""
        from sklearn.cluster import KMeans

        sents = [s for t in texts for s in self._sentences(t)]
        if len(sents) < self.n_clusters:
            raise ValueError(f"need at least {self.n_clusters} sentences, got {len(sents)}")
        emb = self.encoder.encode(sents)
        km = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=self.n_init)
        km.fit(emb)
        self.centers = km.cluster_centers_.astype(np.float32)
        return self

    def assign(self, emb: np.ndarray) -> np.ndarray:
        d = ((emb[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        return d.argmin(axis=1)

    def sentence_clusters(self, text: str) -> List[int]:
        sents = self._sentences(text)
        if not sents:
            return []
        return self.assign(self.encoder.encode(sents)).tolist()

    # ----------------------------------------------------------------- CRF
    def fit_crf(self, passages: Sequence[Tuple[str, str]], model_path: str) -> "CUA":
        """Supervised: ``passages`` are (text, "important" | "unimportant")."""
        import pycrfsuite

        if self.centers is None:
            raise RuntimeError("call fit_clusters first")
        trainer = pycrfsuite.Trainer(verbose=False)
        trainer.set_params({
            "c1": self.c1,
            "c2": self.c2,
            "max_iterations": self.max_iter,
            "feature.possible_transitions": self.all_transitions,
        })
        n_seq = 0
        for text, label in passages:
            ids = self.sentence_clusters(text)
            if not ids:
                continue
            trainer.append(pycrfsuite.ItemSequence(sequence_features(ids)), [label] * len(ids))
            n_seq += 1
        if n_seq == 0:
            raise ValueError("no usable passages for CRF training")
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        trainer.train(str(model_path))
        self._load_tagger(str(model_path))
        return self

    def fit(self, corpus_texts: Sequence[str], passages: Sequence[Tuple[str, str]], model_path: str) -> "CUA":
        return self.fit_clusters(corpus_texts).fit_crf(passages, model_path)

    def _load_tagger(self, path: str) -> None:
        import pycrfsuite

        tagger = pycrfsuite.Tagger()
        tagger.open(path)
        self._tagger = tagger
        self._model_path = path

    # ------------------------------------------------------------ inference
    def classify(self, chunk: str) -> str:
        import pycrfsuite

        if self._tagger is None:
            raise RuntimeError("CRF not trained or loaded")
        ids = self.sentence_clusters(chunk)
        if not ids:
            return UNIMPORTANT
        tags = self._tagger.tag(pycrfsuite.ItemSequence(sequence_features(ids)))
        return vote(tags, self.tie_break)

    def keep(self, chunk: str) -> bool:
        return self.classify(chunk) == IMPORTANT

    def filter(self, chunks: Sequence[str]) -> List[str]:
        return [c for c in chunks if self.keep(c)]

    # ----------------------------------------------------------- persistence
    def save(self, directory: str) -> None:
        """Plain files only (no pickle): centers.npy, crf.crfsuite, meta.json."""
        import shutil

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "centers.npy", self.centers)
        if self._model_path and Path(self._model_path).resolve() != (out / "crf.crfsuite").resolve():
            shutil.copy(self._model_path, out / "crf.crfsuite")
        meta = {
            "n_clusters": self.n_clusters,
            "separators": self.separators,
            "min_sentence_len": self.min_len,
            "tie_break": self.tie_break,
            "encoder": getattr(self.encoder, "name", "unknown"),
            "cluster_names": self.cluster_names,
        }
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str, encoder) -> "CUA":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        obj = cls(
            encoder,
            n_clusters=meta["n_clusters"],
            sentence_separators=meta["separators"],
            min_sentence_len=meta["min_sentence_len"],
            vote_tie_break=meta["tie_break"],
        )
        obj.centers = np.load(d / "centers.npy")
        obj.cluster_names = meta.get("cluster_names", {})
        obj._load_tagger(str(d / "crf.crfsuite"))
        return obj
