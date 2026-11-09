# Implementation notes

Settings that the paper states are in `configs/paper.yaml` (`tests/test_paper_consistency.py`
checks them). The choices below fill in the remaining details; they are in
`configs/impl_defaults.yaml`.

## Chunking and CUA

- Headings: Chinese-numeral headings (`一、`, `二、`, ...) at the start of the text or after a
  sentence break. A text without headings is one chunk.
- CUA: sentences are split at `。；：，` (minimum length 5), clustered with k-means (k = 8,
  seed 42); a linear-chain CRF (c1 = c2 = 0.1, 100 iterations) tags each sentence from the five
  features current / prev / next / is_first / is_last; the chunk type is the majority tag, ties keep
  the chunk. Passage labels are copied to every sentence of the passage when training the CRF.
- A judgment whose chunks are all removed is labeled with no factor.
- Models are stored as `centers.npy`, `crf.crfsuite`, `meta.json` (no pickle).

## Labeling and thresholds

- One similarity matrix per variant is computed once; thresholds are applied afterwards.
- The demo selects the threshold on the grid by 5-fold: each judgment is labeled with the value
  that is best on the other folds (one value shared by the three variants of a family, best mean
  F1); predictions of all folds are pooled. `--tau`, `--tau-pre`, `--tau-sft` give fixed values.
  Thresholds depend on the encoder, the labels and the data, so each setup needs its own.

## Fine-tuning data

1. Chunks are filtered by the CUA.
2. For every (judgment, factor), the top-k = 5 chunks by similarity are sent to the verifier.
3. Verified chunks are positives; rejected candidates are discarded.
4. Each positive becomes one entry with 15 negatives: random chunks of other judgments (the paper does not say where
   its negatives come from). Entries are split 90% / 10% into `training.jsonl` and `dev.jsonl`.

The verifier prompt is in `src/sfl/verifier.py`; the model is chosen by the user. Answers are cached.

## Training

`src/sfl/train.py`: one shared encoder, cls pooling, cosine similarity divided by 0.02, cross-entropy
over the group plus the other entries' passages in the batch (passages of another entry with the same
query are masked out). `configs/demo_train.yaml` uses a smaller batch, fewer epochs and shorter
sequences, and keeps the word embeddings fixed, so that BGE-M3 fits one Colab GPU.
The loop is single-GPU. The training entries use the `{"query", "pos", "neg"}` format of FlagEmbedding's
fine-tuning scripts.

## Splits

`data/demo/splits/sft.json` separates the judgments used for building fine-tuning data from those
used for evaluation; `load_split` checks that they do not overlap, and the CUA clusters are fitted on
the former only.

## Devices

`sfl.device.pick_device` chooses CUDA, then Apple GPU (MPS), then CPU. fp16 autocast is used on CUDA only; on MPS
and CPU training runs in fp32. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by the scripts so that operations MPS lacks
run on the CPU.

## Verification requests

`build_sft_data.py --dry-run` reports how many (factor, chunk) requests the pool needs, how many are already
cached, and their total length in characters, without sending anything. `crossfit_sft.py` writes the LLM answers of
all folds to one file, `<out>/llm_cache.jsonl`, so a (factor, chunk) pair that several folds need is sent only once.

## Evaluating every judgment: cross-fitting

A fixed split (`splits/sft.json`) leaves part of the corpus out of the evaluation. To evaluate all
judgments while no judgment is scored by an encoder trained on it, use `crossfit_sft.py`:

```
python scripts/crossfit_sft.py --folds 5 --verifier mock --config configs/demo_train.yaml
python scripts/crossfit_sft.py --folds 5 --verifier llm --llm-model <model> --dry-run   # request counts per fold, nothing is sent or trained
```

The CUA is fitted once on the first run (saved in `<out>/cua`, reused by every fold and later runs;
`--cua-dir` points to another saved CUA). For each fold it builds the fine-tuning data from the other
folds, fine-tunes an encoder, and scores the held-out fold with it and with the shared CUA. The scores of all folds are
pooled and labeled with one threshold per family, chosen on the other folds of the judgments. The
three SFT variants of a fold share that fold's encoder. The SFT rows therefore come from `--folds`
encoders; the pre-trained rows from one encoder. Training runs once per fold; finished folds are
reused unless `--force` is given.

