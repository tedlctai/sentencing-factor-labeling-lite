# Data schema

All files are UTF-8. `.jsonl` files have one JSON object per line.

| File | Content |
|---|---|
| `judgments.jsonl` | `{"id": str, "case_type": str, "fact": str}`; `fact` is the Fact of the Case section |
| `factors.json` | `{"<case_type>": ["factor text", ...]}` |
| `reference_labels.jsonl` | `{"id": str, "labels": {"<factor text>": 0 or 1}}`; labels used for evaluation |
| `cua_passages.jsonl` | `{"text": str, "label": "important" or "unimportant"}`; passages for the CRF (the paper uses 40). Labels are human judgments of importance, not factor labels |

Files for the fine-tuning part:

| File | Content |
|---|---|
| `verification.jsonl` | `{"id": str, "factor": str, "chunk": str, "relevant": true or false}`; verifier answers for top-k chunks (written by `build_sft_data.py`, readable with `--verifier precomputed`) |
| `splits/<name>.json` | `{"train": [ids], "test": [ids]}` judgment ids; must be disjoint. "train": the only pool fine-tuning data may come from; "test": the evaluation pool |

`data/demo/` is synthetic: invented text, no real people, labels constructed
from the inserted template sentences.


## Using your own data

Put the four files above in a folder and pass it with `--data`, for example
`python scripts/run_demo.py --data my_data --case-type <key in factors.json>`. Every judgment needs a
reference label for every factor of its case type, and the label keys must be the factor texts as they
are written in `factors.json`. Without `splits/<name>.json` all judgments are evaluated.
Real judgments name real people: keep them out of the repository.
