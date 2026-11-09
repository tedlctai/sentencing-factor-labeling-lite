"""Build the fine-tuning data (paper Sec. IV-C) from the judgments reserved for
fine-tuning (split "train"):

    CUA-retained chunks -> top-k by pre-trained BGE-M3  ->  verifier  ->  positives (+ random negatives)

    python scripts/build_sft_data.py --verifier llm --llm-model <model>      # needs OPENAI_API_KEY
    python scripts/build_sft_data.py --verifier precomputed --verification path/to/verification.jsonl
    python scripts/build_sft_data.py --verifier mock                          # demo only, no API

The mock verifier lets the synthetic demo run without an API key (demo data only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sfl.config import impl_defaults  # noqa: E402
from sfl.cua import CUA  # noqa: E402
from sfl.data import load_dataset, load_split  # noqa: E402
from sfl.encoder import make_encoder  # noqa: E402
from sfl.finetune_data import (build_entries, candidate_chunks, split_entries, table1_summary,  # noqa: E402
                               verify_top_k, write_jsonl)
from sfl.verifier import PROMPT, LLMVerifier, PrecomputedVerifier  # noqa: E402
from sfl.verifier import _key  # noqa: E402


def _cache_key(verifier, factor, chunk):
    return _key(verifier.model, factor, chunk)


def mock_verifier():
    import make_demo_data as m

    def verify(factor, chunk):
        return any(sent in chunk for sent in m.FACTOR_SENTENCES[m.FACTORS.index(factor)])

    return verify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/demo")
    ap.add_argument("--case-type", default=None, help="key in factors.json (default: the first one)")
    ap.add_argument("--split", default="sft", help="splits/<name>.json; only its 'train' ids are used")
    ap.add_argument("--encoder", default="BAAI/bge-m3")
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    ap.add_argument("--verifier", choices=["llm", "precomputed", "mock"], default="llm")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    ap.add_argument("--llm-cache", default=None,
                    help="jsonl file that stores every LLM answer (default: <out>/llm_cache.jsonl). "
                         "crossfit_sft.py shares one file across folds, so a (factor, chunk) pair is sent only once.")
    ap.add_argument("--verification", default=None, help="verification.jsonl for --verifier precomputed")
    ap.add_argument("--top-k", type=int, default=impl_defaults()["finetune_data"]["top_k"])
    ap.add_argument("--group-size", type=int, default=16, help="paper: 16; the Colab demo trains with 8")
    ap.add_argument("--n-clusters", type=int, default=8)
    ap.add_argument("--cua-dir", default=None,
                    help="use this saved CUA instead of fitting one (crossfit_sft.py fits it once and passes it here)")
    ap.add_argument("--dev-fraction", type=float, default=impl_defaults()["finetune_data"]["dev_fraction"],
                    help="share of entries held out as a monitoring split (Table I: 10%%)")
    ap.add_argument("--dry-run", action="store_true",
                    help="retrieve the top-k chunks and report how many verification requests would be sent "
                         "(minus those already cached) and their size in characters; nothing is sent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/sft")
    args = ap.parse_args()

    ds = load_dataset(args.data, args.case_type)
    split = load_split(args.data, args.split)
    if split is None:
        raise SystemExit(f"no splits/{args.split}.json in {args.data}: fine-tuning data needs a pool "
                         "that is disjoint from the evaluation judgments")
    pool = ds.subset(split["train"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.verifier == "llm":
        if not args.llm_model:
            raise SystemExit("--llm-model is required for --verifier llm")
        verifier = LLMVerifier(args.llm_model, base_url=args.llm_base_url, cache_path=args.llm_cache or str(out / "llm_cache.jsonl"))
    elif args.verifier == "precomputed":
        verifier = PrecomputedVerifier(args.verification)
    else:
        verifier = mock_verifier()
        print("NOTE: mock LLM verifier (demo only, no LLM call).")

    enc = make_encoder(args.encoder, device=args.device)
    # Step 1 of Sec. IV-C: the CUA (clusters from the fine-tuning pool, CRF from the labeled
    # passages) filters the chunks before retrieval. It is saved so the evaluation can use the same module.
    if args.cua_dir:
        cua = CUA.load(args.cua_dir, enc)
    else:
        cua = CUA(enc, n_clusters=args.n_clusters).fit(pool.texts, ds.passages, str(out / "crf.crfsuite"))
        cua.save(str(out / "cua"))
    if args.dry_run:
        asked = {}
        verify_top_k(enc, pool.texts, pool.ids, pool.factors, lambda f, c: asked.setdefault((f, c), False),
                     top_k=args.top_k, cua=cua)
        cached = sum(1 for (f, c) in asked if getattr(verifier, "cache", None) is not None
                     and _cache_key(verifier, f, c) in verifier.cache)
        chars = sum(len(PROMPT.format(factor=f, chunk=c)) for (f, c) in asked)
        print(json.dumps({"judgments": len(pool.ids), "factors": len(pool.factors), "unique_requests": len(asked),
                          "already_cached": cached, "to_send": len(asked) - cached,
                          "prompt_characters_total": chars}, indent=2))
        return
    records = verify_top_k(enc, pool.texts, pool.ids, pool.factors, verifier, top_k=args.top_k, cua=cua)
    write_jsonl(out / "verification.jsonl", records)
    all_chunks = {i: candidate_chunks(t, cua) for i, t in zip(pool.ids, pool.texts)}
    d = impl_defaults()["finetune_data"]
    entries = build_entries(records, all_chunks, group_size=args.group_size,
                            entries_per_positive=d["entries_per_positive"], seed=args.seed)
    train, dev = split_entries(entries, args.dev_fraction, seed=args.seed)
    write_jsonl(out / "training.jsonl", train)
    write_jsonl(out / "dev.jsonl", dev)
    summary = {"table1_style": table1_summary(entries, train, dev, len(pool.ids), len(pool.factors), args.group_size),
               "verified_pairs": len(records), "accepted": sum(r["relevant"] for r in records),
               "verifier": args.verifier, "llm_model": args.llm_model, "top_k": args.top_k,
               "group_size": args.group_size, "dev_fraction": args.dev_fraction}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
