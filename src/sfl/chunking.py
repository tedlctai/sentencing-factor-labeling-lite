"""Heading-based chunking (paper Sec. IV-A).

The paper describes splitting a judgment at its enumerated headings with a
regular expression. The paper writes the headings symbolically as (1), (2),
(3); the implementation matches Chinese-numeral headings (e.g. ``一、``).
"""
from __future__ import annotations

import re
from typing import List

from .config import impl_defaults

_DEFAULT_REGEX = impl_defaults()["chunking"]["heading_regex"]


def chunk_text(text: str, heading_regex: str | None = None) -> List[str]:
    """Split ``text`` at headings. Headings stay with their chunk.

    If no heading is found the whole text is returned as a single chunk.
    Empty pieces are dropped.
    """
    pattern = re.compile(heading_regex or _DEFAULT_REGEX)
    parts = [p.strip() for p in pattern.split(text)]
    parts = [p for p in parts if p]
    return parts if parts else ([text.strip()] if text.strip() else [])


def split_sentences(text: str, separators: str = "。；：", min_len: int = 1) -> List[str]:
    """Split on any character in ``separators``; drop pieces shorter than ``min_len``."""
    pieces = re.split("[" + re.escape(separators) + "]", text)
    return [p.strip() for p in pieces if len(p.strip()) >= min_len]
