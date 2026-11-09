"""Verification of retrieved chunks (paper Sec. IV-C: LLM-in-the-loop).

A verifier answers one question: does this chunk support this sentencing
factor? The paper uses an LLM for this offline step; the model is given by the
caller.

``LLMVerifier``         any OpenAI-compatible chat-completions endpoint
``PrecomputedVerifier`` answers read from a ``verification.jsonl`` file
Every answer an LLMVerifier obtains is cached in a jsonl file, so a re-run
costs nothing and the file can later be used as a PrecomputedVerifier.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

PROMPT = (
    "你是量刑因子標註助手。以下是判決書事實欄中的一段文字與一個量刑因子。\n"
    "請判斷這段文字是否明確支持該量刑因子（即從這段文字可以得知該因子成立）。\n"
    "只回答一個字母：Y（支持）或 N（不支持）。\n\n"
    "量刑因子：{factor}\n\n文字：{chunk}\n\n回答："
)


def _key(model: str, factor: str, chunk: str) -> str:
    return hashlib.sha256(json.dumps([model, factor, chunk], ensure_ascii=False).encode("utf-8")).hexdigest()


def parse_answer(reply: str) -> bool:
    r = reply.strip().upper()
    if r.startswith("Y") or r.startswith("是"):
        return True
    if r.startswith("N") or r.startswith("否"):
        return False
    raise ValueError(f"cannot parse verifier reply: {reply!r}")


class PrecomputedVerifier:
    """Reads ``{"factor", "chunk", "relevant"}`` lines (extra fields are ignored)."""

    def __init__(self, path):
        self.table = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    self.table[(r["factor"], r["chunk"])] = bool(r["relevant"])

    def __call__(self, factor: str, chunk: str) -> bool:
        try:
            return self.table[(factor, chunk)]
        except KeyError:
            raise KeyError("no precomputed answer for this (factor, chunk) pair") from None


def _http_post(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    import requests  # lazy

    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


class LLMVerifier:
    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        cache_path: Optional[str] = None,
        post: Callable[[str, dict, dict, float], dict] = _http_post,
        retries: int = 3,
        timeout: float = 60.0,
        prompt: str = PROMPT,
    ):
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.post = post
        self.retries = retries
        self.timeout = timeout
        self.prompt = prompt
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        self.calls = 0                      # number of API requests actually sent
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.cache[r["key"]] = bool(r["relevant"])

    def _ask(self, factor: str, chunk: str) -> bool:
        if not self.api_key:
            raise RuntimeError("no API key: set the environment variable named by api_key_env")
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 4,
            "messages": [{"role": "user", "content": self.prompt.format(factor=factor, chunk=chunk)}],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last = None
        for attempt in range(self.retries):
            try:
                self.calls += 1
                data = self.post(self.url, headers, payload, self.timeout)
                return parse_answer(data["choices"][0]["message"]["content"])
            except Exception as e:  # network error, rate limit, unparsable reply
                last = e
                time.sleep(min(2 ** attempt, 8) if self.retries > 1 else 0)
        raise RuntimeError(f"verification failed after {self.retries} attempts: {last}")

    def __call__(self, factor: str, chunk: str) -> bool:
        k = _key(self.model, factor, chunk)
        if k not in self.cache:
            self.cache[k] = self._ask(factor, chunk)
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": k, "model": self.model, "factor": factor,
                                        "chunk": chunk, "relevant": self.cache[k]}, ensure_ascii=False) + "\n")
        return self.cache[k]
