"""DPR-style contrastive fine-tuning of the encoder (paper Sec. IV-C, V-B).

One shared encoder (BGE-M3), cls pooling, L2-normalised embeddings, cosine
similarity divided by the temperature (0.02), cross-entropy over the group
{1 positive + (group_size - 1) negatives} plus the other entries' passages in the
batch (in-batch negatives). A passage that belongs to another entry with the
SAME query text is masked out, because it would be a false negative.

Paper values (configs/paper.yaml): batch 32, group 16, 10 epochs, AdamW,
lr 2e-5, warmup 0.1, temperature 0.02, fp16, two RTX 4090 GPUs. The demo
config (configs/demo_train.yaml) only shrinks what a single Colab GPU needs.
torch / transformers are imported lazily. Runs on CUDA (fp16), Apple GPU (MPS, fp32) or CPU.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from .config import paper_config
from .device import pick_device


def load_entries(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def info_nce_mask(queries):
    """(B, B) bool matrix; True where entry j's passages must be excluded for entry i
    (different entry, identical query text)."""
    n = len(queries)
    return [[i != j and queries[i] == queries[j] for j in range(n)] for i in range(n)]


def train(model_name, entries_path, out_dir, cfg=None, max_steps=None, seed=42, device=None):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    ft = {**paper_config()["finetune"], **(cfg or {})}
    bs, group = int(ft["batch_size"]), int(ft["group_size"])
    epochs, lr = int(ft["epochs"]), float(ft["learning_rate"])
    temp, warm = float(ft["temperature"]), float(ft["warmup_ratio"])
    q_len, p_len = int(ft.get("query_max_len", 64)), int(ft.get("passage_max_len", 512))
    freeze_emb = bool(ft.get("freeze_embeddings", False))
    ckpt = bool(ft.get("gradient_checkpointing", False))

    random.seed(seed)
    torch.manual_seed(seed)
    dev = pick_device(device)
    print(f"device: {dev}", flush=True)
    t0 = time.time()
    entries = load_entries(entries_path)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(dev)
    if freeze_emb:
        for p in model.embeddings.parameters():
            p.requires_grad = False
    if ckpt:
        model.gradient_checkpointing_enable()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    steps_per_epoch = max(1, math.ceil(len(entries) / bs))
    total = steps_per_epoch * epochs if max_steps is None else min(max_steps, steps_per_epoch * epochs)
    sched = get_linear_schedule_with_warmup(opt, int(warm * total), total)
    use_amp = dev == "cuda" and str(ft.get("precision", "fp16")) == "fp16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)      # newer torch
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def embed(texts, max_len):
        b = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        with torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=torch.float16, enabled=use_amp):
            h = model(**b).last_hidden_state[:, 0]          # cls pooling
        return F.normalize(h.float(), dim=-1)

    step = 0
    for epoch in range(epochs):
        random.shuffle(entries)
        for s in range(0, len(entries), bs):
            batch = entries[s : s + bs]
            if len(batch) < 2:
                continue
            queries = [e["query"] for e in batch]
            passages, owner = [], []
            for i, e in enumerate(batch):
                group_p = (e["pos"][:1] + e["neg"])[:group]
                passages += group_p
                owner += [i] * len(group_p)
            q = embed(queries, q_len)
            p = embed(passages, p_len)
            scores = q @ p.T / temp
            mask = torch.zeros_like(scores, dtype=torch.bool)
            same = info_nce_mask(queries)
            for i in range(len(batch)):
                for k, o in enumerate(owner):
                    if same[i][o]:
                        mask[i, k] = True
            scores = scores.masked_fill(mask, float("-inf"))
            first = [owner.index(i) for i in range(len(batch))]          # index of each entry's positive
            loss = F.cross_entropy(scores, torch.tensor(first, device=dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            print(f"epoch {epoch + 1}/{epochs} step {step}/{total} loss {loss.item():.4f}", flush=True)
            if step >= total:
                break
        if step >= total:
            break

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    seconds = time.time() - t0
    print(f"{step} steps on {dev} in {seconds / 60:.1f} min", flush=True)
    (out / "sfl_train_config.json").write_text(
        json.dumps({"base_model": model_name, "entries": str(entries_path), "steps": step,
                    "n_entries": len(entries), "device": str(dev), "seconds": round(seconds, 1),
                    **{k: ft[k] for k in ft}}, indent=2, default=str),
        encoding="utf-8")
    return str(out)
