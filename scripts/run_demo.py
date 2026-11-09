"""End-to-end demo: chunking -> CUA (k-means + CRF) -> thresholded labeling -> metrics,
optionally with a fine-tuned encoder (Table II layout).

Data: if data/demo/splits/sft.json exists, the judgments in its "train" list are
reserved for fine-tuning (and for fitting the CUA clusters); every method is
evaluated on its "test" list only. Without a split, all judgments are evaluated.

Threshold protocol: 5-fold. Every evaluated judgment is labeled with a
threshold chosen on the other folds, and the predictions of all folds are
pooled. One threshold is shared by the three variants of a family (pre-trained
or fine-tuned), as in the paper. --tau gives a fixed threshold instead.
The demo data is synthetic, so the scores show the pipeline running end to end.

    python scripts/run_demo.py                                   # BGE-M3, no fine-tuning
    python scripts/run_demo.py --sft-model outputs/sft/model     # adds the SFT rows
    python scripts/run_demo.py --encoder hash --taus 0.1,0.2     # plumbing check only
    python scripts/run_demo.py --finetune                        # builds the fine-tuning data, fine-tunes once, adds the SFT rows (needs a GPU)
    python scripts/run_demo.py --crossval                        # optional: cross-validated evaluation of the fine-tuning (needs a GPU)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # Apple GPU: fall back to CPU for unsupported ops

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfl import metrics, threshold  # noqa: E402
from sfl.cua import CUA  # noqa: E402
from sfl.data import load_dataset, load_split  # noqa: E402
from sfl.encoder import make_encoder  # noqa: E402
from sfl.experiment import crossfit_report, evaluate_variants, markdown_table, similarity_by_variant  # noqa: E402
from sfl.labeler import apply_threshold  # noqa: E402

SFT_NAMES = {"w/o chunking": "SFT", "w/ Chunking": "SFT w/ Chunking", "w/ CUA": "SFT w/ CUA"}
ROOT = Path(__file__).resolve().parents[1]
TAU_FOLDS = 5  # threshold selection: no training, it only reads the similarity scores
PRE_NAMES = {"w/o chunking": "BGE-M3", "w/ Chunking": "BGE-M3 w/ Chunking", "w/ CUA": "BGE-M3 w/ CUA"}


def finetune_defaults(args):
    """Fill in the verifier and the training config when they were not given: the LLM if --llm-model is
    given, a precomputed file if --verification is given, otherwise the demo-only mock LLM verifier
    (synthetic demo data only); the demo training settings for the demo data."""
    is_demo = Path(args.data).resolve() == (ROOT / "data/demo").resolve()
    if args.verifier is None:
        if args.llm_model:
            args.verifier = "llm"
        elif args.verification:
            args.verifier = "precomputed"
        elif is_demo:
            args.verifier = "mock"
        else:
            raise SystemExit("this data needs a verifier: give --llm-model (with OPENAI_API_KEY) or --verification "
                             "<file>; the mock LLM verifier only works on the synthetic demo data")
    if args.config is None and is_demo:
        args.config = str(ROOT / "configs/demo_train.yaml")


def run_finetune(args):
    """Fine-tuning as one step: build the data from the judgments reserved in the split, train once.
    Sets args.cua_dir and args.sft_model so that the report below adds the SFT rows."""
    if args.split == "none":
        raise SystemExit("--finetune needs the reserved judgments of a split (splits/sft.json); --split none has none")
    finetune_defaults(args)
    sft = Path(args.out) / "sft"
    here = Path(__file__).resolve().parent
    py = str(sys.executable)

    def run(cmd):
        print("$ " + " ".join(str(c) for c in cmd), flush=True)
        subprocess.run([str(c) for c in cmd], check=True)

    if (sft / "training.jsonl").exists():
        print(f"fine-tuning data exists in {sft}, skipping (delete the folder to rebuild)")
    else:
        build = [py, here / "build_sft_data.py", "--data", args.data, "--split", args.split, "--out", sft,
                 "--encoder", args.encoder, "--verifier", args.verifier, "--n-clusters", args.n_clusters,
                 "--seed", args.seed]
        if args.case_type:
            build += ["--case-type", args.case_type]
        if args.cua_dir:
            build += ["--cua-dir", args.cua_dir]
        if args.llm_model:
            build += ["--llm-model", args.llm_model]
        if args.verification:
            build += ["--verification", args.verification]
        if args.device:
            build += ["--device", args.device]
        run(build)
    if (sft / "model" / "config.json").exists():
        print(f"fine-tuned model exists in {sft / 'model'}, skipping (delete the folder to retrain)")
    else:
        train = [py, here / "train_sft.py", "--entries", sft / "training.jsonl", "--out", sft / "model"]
        if args.config:
            train += ["--config", args.config]
        if args.device:
            train += ["--device", args.device]
        run(train)
    args.cua_dir = args.cua_dir or str(sft / "cua")
    args.sft_model = str(sft / "model")


def family(S, ds, taus, fixed_tau, seed, names):
    if fixed_tau is not None:
        rows = evaluate_variants(S, ds.Y, fixed_tau)
        preds = {v: apply_threshold(m, fixed_tau) for v, m in S.items()}
    else:
        rows, preds, _ = crossfit_report(S, ds.Y, ds.ids, taus, k=TAU_FOLDS, seed=seed)
    for r in rows:
        r["variant"] = names[r["variant"]]
    return rows, preds


def print_diff(title, Y, pa, pb, seed):
    diff = metrics.paired_bootstrap_diff(Y, pa, pb, seed=seed)
    print(f"\n{title}, same judgments, 95% bootstrap interval:")
    for key, label in (("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
        d, lo, hi = diff[key]
        print(f"  {label:<9} {d:+6.2f}  [{lo:+.2f}, {hi:+.2f}]")


def print_cua_effect(label, S, preds, Y):
    """What CUA actually changed relative to chunking alone."""
    n = Y.size
    changed = int((S["w/ CUA"] < S["w/ Chunking"] - 1e-6).sum())
    removed = (preds["w/ Chunking"] == 1) & (preds["w/ CUA"] == 0)
    print(f"  {label}: CUA lowers the max similarity in {changed}/{n} pairs; removes "
          f"{int(removed.sum())} predicted positives ({int((removed & (Y == 0)).sum())} wrong, "
          f"{int((removed & (Y == 1)).sum())} correct)")


def main():
    if "--crossval" in sys.argv:
        # Evaluation option: hand over to crossfit_sft.py with the remaining arguments.
        sys.argv.remove("--crossval")
        sys.argv[0] = str(Path(__file__).resolve().with_name("crossfit_sft.py"))
        import crossfit_sft
        return crossfit_sft.main()
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossval", action="store_true",
                    help="cross-validated evaluation of the fine-tuned encoder over all judgments "
                         "(runs scripts/crossfit_sft.py; its options such as --folds apply)")
    ap.add_argument("--finetune", action="store_true",
                    help="first build the fine-tuning data from the reserved judgments and fine-tune once, "
                         "then add the SFT rows to the report (needs a GPU)")
    ap.add_argument("--llm-model", default=None, help="with --finetune: LLM that verifies the top-k chunks (needs OPENAI_API_KEY)")
    ap.add_argument("--verification", default=None, help="with --finetune: precomputed verification.jsonl instead of an LLM")
    ap.add_argument("--verifier", choices=["llm", "precomputed", "mock"], default=None, help=argparse.SUPPRESS)
    ap.add_argument("--config", default=None,
                    help="with --finetune: training settings yaml (default for the demo data: configs/demo_train.yaml)")
    ap.add_argument("--data", default="data/demo")
    ap.add_argument("--split", default="sft", help="splits/<name>.json; use 'none' to evaluate on all judgments")
    ap.add_argument("--encoder", default="BAAI/bge-m3",
                    help='model name (default: BAAI/bge-m3). "hash" is an offline test encoder for checking '
                         'that the scripts run (smoke tests only).')
    ap.add_argument("--cua-dir", default=None,
                    help="use a saved CUA (e.g. outputs/sft/cua from build_sft_data.py) so that data building and "
                         "evaluation use the same module, as in the paper")
    ap.add_argument("--sft-model", default=None, help="directory written by scripts/train_sft.py")
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    ap.add_argument("--n-clusters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--taus", default=None,
                    help="comma-separated thresholds; default is the paper grid 0.50..0.85. "
                         "The hash test encoder produces low similarities, so use e.g. 0.10,0.15,...,0.35 with it.")
    ap.add_argument("--case-type", default=None, help="key in factors.json (default: the first one)")
    ap.add_argument("--tau", type=float, default=None,
                    help="use this fixed threshold for every judgment instead of the 5-fold selection")
    ap.add_argument("--tau-pre", type=float, default=None, help="fixed threshold for the pre-trained family only")
    ap.add_argument("--tau-sft", type=float, default=None, help="fixed threshold for the fine-tuned family only")
    args = ap.parse_args()
    taus = [float(x) for x in args.taus.split(",")] if args.taus else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.finetune:
        if args.sft_model:
            raise SystemExit("use either --finetune or --sft-model, not both")
        run_finetune(args)
    full = load_dataset(args.data, args.case_type)
    split = None if args.split == "none" else load_split(args.data, args.split)
    if split:
        fit_ds, ds = full.subset(split["train"]), full.subset(split["test"])
        scope = f"evaluated on {len(ds.ids)} judgments disjoint from the {len(fit_ds.ids)} reserved for fine-tuning"
    else:
        fit_ds, ds = full, full
        scope = f"evaluated on all {len(ds.ids)} judgments (no split)"
    if args.sft_model and not split:
        print("WARNING: --split none evaluates every judgment, including any the fine-tuned encoder was trained on.")
    enc = make_encoder(args.encoder, device=args.device)
    dev = getattr(getattr(enc, "model", None), "device", None)
    print(f"encoder={getattr(enc, 'name', args.encoder)}" + (f" (device: {dev})" if dev else "")
          + f"  factors={len(ds.factors)}  {scope}")

    # CUA: clusters from the reserved pool's texts (unsupervised), CRF from the labeled passages.
    if args.cua_dir:
        cua = CUA.load(args.cua_dir, enc)
    else:
        cua = CUA(enc, n_clusters=args.n_clusters).fit(fit_ds.texts, full.passages, str(out / "crf.crfsuite"))
        cua.save(str(out / "cua"))

    S = similarity_by_variant(enc, ds.texts, ds.factors, cua=cua, batch_size=args.batch_size)
    tau_pre, tau_sft = args.tau_pre, args.tau_sft
    if args.tau is not None:
        tau_pre = tau_pre if tau_pre is not None else args.tau
        tau_sft = tau_sft if tau_sft is not None else args.tau
    rows, preds = family(S, ds, taus, tau_pre, args.seed, PRE_NAMES)
    all_preds = {"pre": preds}

    if args.sft_model:
        sft = make_encoder(args.sft_model, device=args.device)          # the CUA above still uses the original encoder
        S2 = similarity_by_variant(sft, ds.texts, ds.factors, cua=cua, batch_size=args.batch_size)
        rows2, preds2 = family(S2, ds, taus, tau_sft, args.seed, SFT_NAMES)
        rows += rows2
        all_preds["sft"] = preds2

    if tau_pre is None and tau_sft is None:
        how = f"{TAU_FOLDS}-fold: each judgment is labeled with a tau chosen on the other folds"
    else:
        how = f"fixed tau: pre-trained {tau_pre}, fine-tuned {tau_sft}" + ""
    note = f"{how}; {scope}"
    table = markdown_table(rows)
    print("\n" + note + "\n\n" + table)

    print_diff("BGE-M3 w/ CUA minus w/ Chunking", ds.Y, preds["w/ CUA"], preds["w/ Chunking"], args.seed)
    if "sft" in all_preds:
        p2 = all_preds["sft"]
        print_diff("SFT w/ CUA minus SFT w/ Chunking", ds.Y, p2["w/ CUA"], p2["w/ Chunking"], args.seed)
        print_diff("SFT minus BGE-M3 (whole fact, no chunking)", ds.Y, p2["w/o chunking"], preds["w/o chunking"], args.seed)

    print("\nWhat CUA changed relative to chunking (pooled predictions):")
    print_cua_effect("BGE-M3", S, preds, ds.Y)
    if "sft" in all_preds:
        print_cua_effect("SFT   ", S2, all_preds["sft"], ds.Y)

    print("\nThreshold sensitivity (w/ CUA, pre-trained encoder):")
    for r in threshold.sweep(S["w/ CUA"], ds.Y, taus):
        print(f"  tau={r['tau']:.2f}  acc={r['accuracy']:.1f}  prec={r['precision']:.1f}  "
              f"rec={r['recall']:.1f}  f1={r['f1']:.1f}")

    pf = metrics.per_factor(ds.Y, preds["w/ CUA"], ds.factors)
    print("\nPer-factor (BGE-M3 w/ CUA):")
    for f, m in pf.items():
        print(f"  {f}: P={m['precision']:.1f} R={m['recall']:.1f} F1={m['f1']:.1f} (support {m['support']})")

    (out / "results.md").write_text(note + "\n\n" + table + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}/results.md")


if __name__ == "__main__":
    main()
