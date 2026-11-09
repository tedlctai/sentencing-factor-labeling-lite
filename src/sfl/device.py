"""Device choice: CUDA GPU, Apple GPU (MPS) or CPU."""
from __future__ import annotations


def pick_device(requested: str | None = None) -> str:
    """``requested`` ("cuda", "mps", "cpu") is returned unchanged; otherwise the best available
    device is chosen in the order cuda, mps, cpu."""
    if requested:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
