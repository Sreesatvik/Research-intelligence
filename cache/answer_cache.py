import re
from collections import OrderedDict
from typing import Any, Callable

import numpy as np

# numbers, ALLCAPS acronyms and CamelCase names: tokens that change the meaning of a question
KEY_TOKEN_RE = re.compile(r"\b(?:[A-Z][A-Za-z]*[A-Z][A-Za-z0-9]*|[A-Z]{2,}|\w*\d\w*)\b")


def normalize(question: str) -> str:
    """Lowercase, drop punctuation, collapse spaces: so trivial rewrites match exactly."""
    text = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    return " ".join(text.split())


def key_tokens(question: str) -> frozenset[str]:
    return frozenset(KEY_TOKEN_RE.findall(question))


class AnswerCache:
    """Two-level cache: exact (always safe) and semantic (strict threshold + guard)."""

    def __init__(self, embed: Callable[[str], np.ndarray] | None = None,
                 semantic_threshold: float = 0.96, max_size: int = 1000,
                 backend=None, ttl_seconds: int = 86_400,
                 dumps: Callable[[Any], str] | None = None, loads: Callable[[Any], Any] | None = None):
        """backend: an optional Redis client. When given, EXACT hits are shared between servers (stored with
        a time limit). The semantic layer needs a vector index, so it stays in this process's memory."""
        self.embed, self.threshold, self.max_size = embed, semantic_threshold, max_size
        self.backend, self.ttl, self.dumps, self.loads = backend, ttl_seconds, dumps, loads
        self.entries: OrderedDict[tuple[str, str], dict] = OrderedDict()  # (scope, normalized) -> entry
        self.stats = {"exact": 0, "semantic": 0, "miss": 0}

    def get(self, question: str, scope: str = "") -> tuple[Any, str] | None:
        """Return (value, 'exact' | 'semantic') or None."""
        key = (scope, normalize(question))
        if key in self.entries:
            self.entries.move_to_end(key)  # mark as recently used
            self.stats["exact"] += 1
            return self.entries[key]["value"], "exact"
        if self.backend is not None:
            raw = self.backend.get(self._remote_key(scope, key[1]))
            if raw is not None:  # another server answered this before
                value = self.loads(raw)
                self._remember(question, value, scope)  # keep it locally for the semantic layer
                self.stats["exact"] += 1
                return value, "exact"
        if self.embed is not None:
            candidates = [e for (s, _), e in self.entries.items() if s == scope]
            if candidates:
                vec = self.embed(question)
                sims = np.stack([e["vec"] for e in candidates]) @ vec
                best = int(np.argmax(sims))
                if sims[best] >= self.threshold and candidates[best]["key_tokens"] == key_tokens(question):
                    self.stats["semantic"] += 1
                    return candidates[best]["value"], "semantic"
        self.stats["miss"] += 1
        return None

    def _remote_key(self, scope: str, normalized: str) -> str:
        return f"answer:{scope}:{normalized}"

    def put(self, question: str, value: Any, scope: str = "") -> None:
        self._remember(question, value, scope)
        if self.backend is not None:
            self.backend.set(self._remote_key(scope, normalize(question)), self.dumps(value), ex=self.ttl)

    def _remember(self, question: str, value: Any, scope: str) -> None:
        vec = self.embed(question) if self.embed is not None else None
        self.entries[(scope, normalize(question))] = {
            "value": value, "vec": vec, "key_tokens": key_tokens(question)}
        self.entries.move_to_end((scope, normalize(question)))
        while len(self.entries) > self.max_size:  # evict the least recently used
            self.entries.popitem(last=False)

    def clear(self) -> None:
        """Call this when the document collection changes, so stale answers are dropped."""
        self.entries.clear()
        if self.backend is not None:
            for key in self.backend.scan_iter("answer:*"):
                self.backend.delete(key)
