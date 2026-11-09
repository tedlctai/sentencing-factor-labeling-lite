"""Cross-fitted fine-tuning and evaluation on a single set of judgments.

When all judgments are to be evaluated (for example one corpus with reference
labels and no separate test set), every judgment can still be scored by an
encoder that never saw it:

    1. split the judgments into K folds (judgment level)
    2. fit the CUA once (the first run; it is saved in <out>/cua and reused afterwards): clusters from
       a fixed set of judgment texts (unsupervised; the "train" list of splits/sft.json if the data has
       one, as in run_demo.py, otherwise all judgments), CRF from the labeled passages
    3. for each fold: build the fine-tuning data from the OTHER folds
       (build_sft_data.py), fine-tune an encoder (train_sft.py), and use it to
       score the held-out fold with the shared CUA
    4. pool the scores of all folds and label them with one threshold per family,
       chosen on the other folds of the judgments (as in run_demo.py)
    5. compare with the reference labels

The result is the Table II layout over every judgment, where the SFT rows come
from K encoders (each fold has its own) and the pre-trained rows from one
encoder. Training happens K times.

Early results: after every fold the script prints and writes <out>/foldN/result.md (that fold
alone, then all folds finished so far pooled) and saves <out>/foldN/scores.npz (a restart reuses
it instead of scoring again). These are preliminary: tau is chosen inside the data shown. The
final results.md (all folds pooled) is the result; results_per_fold.md adds mean +- sd over folds.

    python scripts/crossfit_sft.py --folds 5 --verifier mock --config configs/demo_train.yaml
    python scripts/crossfit_sft.py --folds 5 --verifier llm --llm-model <model> --dry-run   # request counts only
    python scripts/crossfit_sft.py --encoder hash --skip-train --verifier mock --taus 0.1,0.15,0.2,0.25,0.3  # plumbing check
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sfl import splits  # noqa: E402
from sfl.config import impl_defaults  # noqa: E402
from sfl.cua import CUA  # noqa: E402
from sfl.data import load_dataset, load_split  # noqa: E402
from sfl.encoder import make_encoder  # noqa: E402
from sfl.experiment import markdown_table, similarity_by_variant  # noqa: E402
from run_demo import PRE_NAMES, SFT_NAMES, family, finetune_defaults, print_cua_effect, print_diff  # noqa: E402


def run(cmd):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def save_scores(path, sp, ss):
    """Similarity scores of one fold's held-out judgments (rows in the fold's order)."""
    arrays = {"variants": np.array(list(sp))}
    for tag, sims in (("pre", sp), ("sft", ss)):
        for i, v in enumerate(sp):
            arrays[f"{tag}_{i}"] = sims[v]
    np.savez(path, **arrays)


def load_scores(path):
    z = np.load(path)
    names = [str(v) for v in z["variants"]]
    return ({v: z[f"pre_{i}"] for i, v in enumerate(names)},
            {v: z[f"sft_{i}"] for i, v in enumerate(names)})


def early_rows(sub, sp, ss, taus, seed):
    rows, _ = family(sp, sub, taus, None, seed, PRE_NAMES)
    rows2, _ = family(ss, sub, taus, None, seed, SFT_NAMES)
    return rows + rows2


def write_per_fold(out, per_fold):
    """results_per_fold.md: every fold's table and the spread across folds."""
    if not per_fold:
        return
    parts = [f"Per-fold results (each fold alone; tau chosen inside the fold, so descriptive)\n"]
    for f in sorted(per_fold):
        parts.append(f"### fold {f + 1}\n\n" + markdown_table(per_fold[f]) + "\n")
    names = [r["variant"] for r in per_fold[min(per_fold)]]
    lines = ["| Method | Acc (%) mean ± sd | F1 (%) mean ± sd |", "|---|---|---|"]
    for v in names:
        cells = []
        for key in ("accuracy", "f1"):
            x = np.array([r[key] for f in per_fold for r in per_fold[f] if r["variant"] == v])
            cells.append(f"{x.mean():.2f} ± {x.std(ddof=1) if len(x) > 1 else 0.0:.2f}")
        lines.append(f"| {v} | {cells[0]} | {cells[1]} |")
    parts.append(f"### across {len(per_fold)} folds\n\n" + "\n".join(lines) + "\n")
    (out / f"results_per_fold.md").write_text("\n".join(parts), encoding="utf-8")


def main():
    d = impl_defaults()["finetune_data"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/demo")
    ap.add_argument("--case-type", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--encoder", default="BAAI/bge-m3")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="outputs/crossfit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--taus", default=None)
    # fine-tuning data (passed to build_sft_data.py)
    ap.add_argument("--verifier", choices=["llm", "precomputed", "mock"], default=None,
                    help="default: llm if --llm-model is given, precomputed if --verification is given, "
                         "otherwise mock (demo data only, no API)")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--llm-base-url", default=None)
    ap.add_argument("--verification", default=None)
    ap.add_argument("--top-k", type=int, default=d["top_k"])
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--dev-fraction", type=float, default=d["dev_fraction"])
    ap.add_argument("--n-clusters", type=int, default=8)
    ap.add_argument("--cua-dir", default=None,
                    help="use this saved CUA instead of fitting one (default: fit once, save to <out>/cua, reuse)")
    # training (passed to train_sft.py)
    ap.add_argument("--config", default=None,
                    help="yaml with a 'finetune' section (default for the demo data: configs/demo_train.yaml; "
                         "configs/paper.yaml for the paper's settings)")
    ap.add_argument("--base-model", default="BAAI/bge-m3")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--skip-train", action="store_true",
                    help="plumbing check only: use the pre-trained encoder in place of each fold's fine-tuned one")
    ap.add_argument("--dry-run", action="store_true",
                    help="only count the verification requests of every fold (nothing is sent or trained)")
    ap.add_argument("--force", action="store_true", help="rebuild folds that already have outputs")
    args = ap.parse_args()
    taus = [float(x) for x in args.taus.split(",")] if args.taus else None
    finetune_defaults(args)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(args.data, args.case_type)
    folds = splits.kfold(ds.ids, k=args.folds, seed=args.seed)
    print(f"{len(ds.ids)} judgments, {len(ds.factors)} factors, {args.folds} folds: "
          + ", ".join(f"{len(te)}" for _, te in folds) + " held-out judgments per fold")

    py = sys.executable
    pos = {i: n for n, i in enumerate(ds.ids)}
    n, m = len(ds.ids), len(ds.factors)
    S_pre = {}
    S_sft = {}
    done_ids, per_fold = [], {}
    enc = make_encoder(args.encoder, device=args.device)

    # The CUA does not depend on the folds: it is fitted once (clusters from a fixed set of judgment texts,
    # unsupervised; CRF from the labeled passages), saved, and reused by every fold and every later run.
    # The fixed set is the same one run_demo.py uses (splits/sft.json "train"), else all judgments.
    cua_dir = Path(args.cua_dir) if args.cua_dir else out / "cua"
    if args.cua_dir or (cua_dir / "meta.json").exists() and not args.force:
        print(f"using the saved CUA in {cua_dir}")
    else:
        print(f"fitting the CUA once (saved in {cua_dir}, reused by all folds)")
        reserved = load_split(args.data, "sft")
        corpus = ds.subset(reserved["train"]).texts if reserved else ds.texts
        CUA(enc, n_clusters=args.n_clusters).fit(corpus, ds.passages, str(out / "crf.crfsuite")).save(str(cua_dir))
    cua = CUA.load(str(cua_dir), enc)

    for f, (train_ids, test_ids) in enumerate(folds):
        splits.assert_disjoint(train_ids, test_ids)
        fdir = out / f"fold{f}"
        fdir.mkdir(parents=True, exist_ok=True)
        split_path = fdir / "split.json"
        split_path.write_text(json.dumps({"train": train_ids, "test": test_ids}), encoding="utf-8")
        sft_dir, model_dir = fdir / "sft", fdir / "model"
        print(f"\n=== fold {f + 1}/{args.folds}: pool {len(train_ids)} judgments, held-out {len(test_ids)} ===")

        build = [py, ROOT / "scripts/build_sft_data.py", "--data", args.data, "--split", split_path, "--out", sft_dir,
                 "--encoder", args.encoder, "--verifier", args.verifier, "--top-k", args.top_k,
                 "--group-size", args.group_size, "--dev-fraction", args.dev_fraction,
                 "--n-clusters", args.n_clusters, "--cua-dir", cua_dir,
                 "--seed", args.seed]
        if args.case_type:
            build += ["--case-type", args.case_type]
        if args.device:
            build += ["--device", args.device]
        if args.llm_model:
            build += ["--llm-model", args.llm_model]
        if args.llm_base_url:
            build += ["--llm-base-url", args.llm_base_url]
        if args.verifier == "llm":
            # One cache for all folds: the same (factor, chunk) is asked only once, however many folds see it.
            build += ["--llm-cache", out / "llm_cache.jsonl"]
        if args.verification:
            build += ["--verification", args.verification]
        if args.dry_run:
            run(build + ["--dry-run"])
            continue
        if args.force or not (sft_dir / "training.jsonl").exists():
            run(build)
        else:
            print(f"fine-tuning data exists in {sft_dir}, skipping (use --force to rebuild)")

        if args.skip_train:
            sft_enc = enc
        else:
            if args.force or not (model_dir / "config.json").exists():
                train = [py, ROOT / "scripts/train_sft.py", "--entries", sft_dir / "training.jsonl", "--out", model_dir,
                         "--base-model", args.base_model, "--seed", args.seed]
                if args.config:
                    train += ["--config", args.config]
                if args.max_steps:
                    train += ["--max-steps", args.max_steps]
                if args.device:
                    train += ["--device", args.device]
                run(train)
            else:
                print(f"model exists in {model_dir}, skipping (use --force to retrain)")
            sft_enc = make_encoder(str(model_dir), device=args.device)

        # Score the held-out fold with the shared CUA and this fold's encoder.
        sub = ds.subset(test_ids)
        scores_path = fdir / f"scores.npz"
        if scores_path.exists() and not args.force:
            sp, ss = load_scores(scores_path)
            print(f"scores exist in {scores_path}, skipping scoring (use --force to redo)")
        else:
            sp = similarity_by_variant(enc, sub.texts, sub.factors, cua=cua, batch_size=args.batch_size)
            ss = similarity_by_variant(sft_enc, sub.texts, sub.factors, cua=cua, batch_size=args.batch_size)
            save_scores(scores_path, sp, ss)
        rows_idx = [pos[i] for i in sub.ids]
        for store, sims in ((S_pre, sp), (S_sft, ss)):
            for v, S in sims.items():
                store.setdefault(v, np.zeros((n, m), dtype=np.float32))[rows_idx] = S
        done_ids.extend(sub.ids)

        # Early look: this fold alone, then every fold finished so far. Preliminary only; the final
        # table below (all folds pooled) is the result.
        try:
            fold_rows = early_rows(sub, sp, ss, taus, args.seed)
            per_fold[f] = fold_rows
            text = (f"fold {f + 1}/{args.folds} alone ({len(sub.ids)} judgments; tau chosen inside this fold, "
                    f"so preliminary)\n\n" + markdown_table(fold_rows))
            if len(done_ids) > len(sub.ids):
                sofar = ds.subset(done_ids)
                ix = [pos[i] for i in sofar.ids]
                text += (f"\n\nfolds 1-{f + 1} pooled so far ({len(sofar.ids)} judgments)\n\n"
                         + markdown_table(early_rows(sofar, {v: S[ix] for v, S in S_pre.items()},
                                                     {v: S[ix] for v, S in S_sft.items()}, taus, args.seed)))
            (fdir / f"result.md").write_text(text + "\n", encoding="utf-8")
            print("\n--- preliminary result after fold " + str(f + 1) + " ---\n" + text + "\n", flush=True)
        except Exception as e:  # an early report must never stop the run
            print(f"(early report skipped: {e})", flush=True)

    if args.dry_run:
        return

    rows, preds = family(S_pre, ds, taus, None, args.seed, PRE_NAMES)
    rows2, preds2 = family(S_sft, ds, taus, None, args.seed, SFT_NAMES)
    rows += rows2
    note = (f"cross-fitted: {args.folds} folds, every judgment scored by an encoder that did not see it; one CUA shared by all folds; "
            f"SFT rows pool {args.folds} fine-tuned encoders; tau chosen on the other folds of the judgments"
            + ("; PLUMBING CHECK: the pre-trained encoder stands in for the fine-tuned ones" if args.skip_train else ""))
    table = markdown_table(rows)
    print("\n" + note + "\n\n" + table)
    print_diff("BGE-M3 w/ CUA minus w/ Chunking", ds.Y, preds["w/ CUA"], preds["w/ Chunking"], args.seed)
    print_diff("SFT w/ CUA minus SFT w/ Chunking", ds.Y, preds2["w/ CUA"], preds2["w/ Chunking"], args.seed)
    print_diff("SFT minus BGE-M3 (whole fact, no chunking)", ds.Y, preds2["w/o chunking"], preds["w/o chunking"], args.seed)
    print("\nWhat CUA changed relative to chunking (pooled predictions):")
    print_cua_effect("BGE-M3", S_pre, preds, ds.Y)
    print_cua_effect("SFT   ", S_sft, preds2, ds.Y)
    (out / f"results.md").write_text(note + "\n\n" + table + "\n", encoding="utf-8")
    (out / f"results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_per_fold(out, per_fold)
    print(f"\nwrote {out}/results.md" + (f" and {out}/results_per_fold.md" if per_fold else ""))


if __name__ == "__main__":
    main()
