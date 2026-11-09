"""Fine-tune BGE-M3 on the entries written by build_sft_data.py.

    python scripts/train_sft.py --entries outputs/sft/training.jsonl --out outputs/sft/model \\
        --config configs/demo_train.yaml           # Colab-size settings
    python scripts/train_sft.py ... --max-steps 2  # quick check that training starts

Without --config the paper's values from configs/paper.yaml are used (they were
run on two RTX 4090 GPUs; a single small GPU will run out of memory).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # Apple GPU: fall back to CPU for unsupported ops

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfl.config import load_yaml  # noqa: E402
from sfl.train import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", default="outputs/sft/training.jsonl")
    ap.add_argument("--out", default="outputs/sft/model")
    ap.add_argument("--base-model", default="BAAI/bge-m3")
    ap.add_argument("--config", default=None, help="yaml with a 'finetune' section that overrides paper values")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    args = ap.parse_args()
    cfg = None
    if args.config:
        p = Path(args.config)
        cfg = load_yaml(p.name, p.parent)["finetune"]
        print("overriding paper values with:", cfg)
    else:
        print("using the paper's settings (two 24 GB GPUs). For a single small GPU, add --config configs/demo_train.yaml")
    print("saved to", train(args.base_model, args.entries, args.out, cfg, args.max_steps, args.seed, args.device))


if __name__ == "__main__":
    main()
