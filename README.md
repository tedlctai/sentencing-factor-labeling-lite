# Sentencing Factor Labeling (SFL) - Lite

A lightweight implementation of the embedding-based sentencing factor labeling pipeline
described in *A Scalable Embedding-Based Framework for Sentencing Factor Labeling for Legal
Judgments* (KSE 2026). It is meant to make the method easy to follow and to run on a small
example; the original data and trained models of the paper's experiments are not included.

```
judgment (Fact of the Case)
  -> heading-based chunking
  -> chunk utility assessment (CUA): sentence clusters + CRF, drops low-value chunks
  -> factor / chunk similarity (BGE-M3 by default)
  -> threshold  ->  factor labels

optional, offline: LLM-verified top-k chunks -> DPR-style fine-tuning of the encoder
```

## Quick start

Colab: open `notebooks/colab_demo.ipynb` (GPU runtime), or [open it in Colab](https://colab.research.google.com/github/tedlctai/sentencing-factor-labeling-lite/blob/main/notebooks/colab_demo.ipynb).

Local:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e . sentence-transformers
python -m unittest discover -s tests
```

Then pick one:

```bash
python scripts/run_demo.py               # the labeling pipeline: chunking, CUA, similarity, threshold, metrics
python scripts/run_demo.py --finetune    # the same, plus fine-tuning the encoder once (needs a GPU)
```

`--finetune` builds the fine-tuning data from the judgments reserved for it
(`data/demo/splits/sft.json`), fine-tunes once, and adds three SFT rows to the table. On the demo data an
LLM is not needed: a mock LLM verifier stands in for it. To verify with a real LLM add
`--llm-model <model>` (needs `OPENAI_API_KEY`).

The demo runs on synthetic judgments in `data/demo/`, so it works out of the box. `run_demo.py` prints a table in the layout of the
paper's Table II (accuracy, precision, recall, F1, Hamming loss), the paired difference between
*w/ CUA* and *w/ Chunking*, a threshold sweep and per-factor results.

On a Mac with Apple silicon the encoder uses the Apple GPU (MPS) automatically; `--device cpu|mps|cuda`
overrides the choice. Use a smaller `--batch-size` if memory is tight.

### How the labeling pipeline and the fine-tuning fit together

```
 labeling (run_demo.py)                         fine-tuning (offline, optional)
 ----------------------                         -------------------------------
 judgment                                       judgments reserved for fine-tuning
   -> chunking                                    -> chunking
   -> CUA filter  <---------- same CUA ---------->  -> CUA filter
   -> factor / chunk similarity                     -> top-k chunks per factor (pre-trained encoder)
        (BGE-M3 by default, or a      <--------     -> LLM verification (rejected: dropped)
         fine-tuned one: --sft-model) encoder       -> DPR fine-tuning  ->  fine-tuned encoder
   -> max similarity >= tau -> factor labels
```

The CUA is built once (k-means on a fixed set of judgments, CRF on the labeled passages) and is the
same module in both columns (`--cua-dir`). Fine-tuning changes only the encoder used for the
similarity step; chunking, CUA and the rule "max similarity >= tau" stay the same.

### More options

- Other verifier: `--llm-model <model>` (with `OPENAI_API_KEY`) or `--verification <file>`.
  By default the demo uses a mock LLM verifier, which only works on the synthetic demo data.
- Count the verification requests before sending any: `build_sft_data.py --verifier llm --llm-model <model> --dry-run` (also in `crossfit_sft.py`); nothing is sent.
- Other data: `--data <folder>` (see `data/SCHEMA.md`).
- The paper's training settings instead of the reduced demo ones: `--config configs/paper.yaml`.
- One step at a time:

```bash
python scripts/build_sft_data.py --verifier llm --llm-model <model>   # needs OPENAI_API_KEY
python scripts/train_sft.py --config configs/demo_train.yaml           # CUDA GPU or Apple GPU (MPS)
python scripts/run_demo.py --sft-model outputs/sft/model               # adds the three SFT rows
```

`train_sft.py` is a small single-GPU loop. The entries in `training.jsonl` follow the `{"query", "pos", "neg"}`
format that FlagEmbedding's fine-tuning scripts read; a model folder trained elsewhere can be used with `--sft-model`.

### Optional: cross-validated evaluation

For the case where there is no separate test set and every judgment should still be evaluated by an
encoder that did not see it: `python scripts/run_demo.py --crossval`. The CUA is fitted once and
reused; for each of 5 folds it builds the fine-tuning data from the other folds, fine-tunes an encoder
(5 trainings, needs a GPU) and scores the held-out fold; the threshold is chosen on the other folds.
`--folds N` changes the number of folds; the verifier and data options above apply. The same script can
be run directly: `scripts/crossfit_sft.py`.

## What is where

| Part | Module |
|---|---|
| Chunking at enumerated headings | `src/sfl/chunking.py` |
| CUA (k-means with k = 8, CRF with five features, majority vote) | `src/sfl/cua.py` |
| Thresholded labeling, three variants (whole / chunking / CUA) | `src/sfl/labeler.py`, `experiment.py` |
| Threshold grid 0.50 to 0.85 | `src/sfl/threshold.py` |
| Metrics over (judgment, factor) pairs | `src/sfl/metrics.py` |
| Fine-tuning data (top-k, verification, pairs) | `src/sfl/verifier.py`, `finetune_data.py` |
| Contrastive fine-tuning | `src/sfl/train.py` |
| Evaluating every judgment with held-out fine-tuning (optional) | `scripts/crossfit_sft.py` |

Values stated in the paper are in `configs/paper.yaml`; other settings are in
`configs/impl_defaults.yaml`. Details are in `docs/implementation_notes.md`.

## Data

`data/SCHEMA.md` describes the files. To use your own judgments, see "Using your own data" in it.

The demo case type `demo_theft` is invented: 5 synthetic factors with a theft-like theme, separate from
the paper's case types.

In the paper, the threshold tau depends on the case type and on whether the encoder is pre-trained or
fine-tuned (a fine-tuned encoder needs a larger tau). The demo sweeps tau over the paper's grid and
picks the value on its own data.

## Layout

```
configs/    paper.yaml, impl_defaults.yaml, demo_train.yaml
src/sfl/    library
scripts/    run_demo.py, build_sft_data.py, train_sft.py, crossfit_sft.py, make_demo_data.py
tests/      unit tests (offline)
docs/       implementation_notes.md
data/       SCHEMA.md, synthetic demo data
notebooks/  colab_demo.ipynb
```

## License

MIT, see `LICENSE`.
