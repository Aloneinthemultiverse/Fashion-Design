"""Disk-backed embedding cache.

CLIP costs roughly 0.3s per image on CPU. That is fine for one query and ruinous for a
corpus: re-embedding 2,000 outfits on every process start is about eleven minutes before
the app serves anything, every time. The vectors never change for a given image and
model, so they only need computing once.

Wraps any `Embedder` and is itself one, so it drops in wherever the real encoder goes
and nothing downstream knows the difference.

The cache key includes the model identity, not just the content. Two models produce
incomparable vectors for the same image, and silently mixing them would corrupt the
index in a way that looks like poor retrieval rather than a bug.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

from fashion.ports.embedder import Embedder

log = logging.getLogger(__name__)


class CachingEmbedder:
    """Memoises another embedder's output to disk."""

    def __init__(
        self,
        inner: Embedder,
        cache_dir: Path,
        *,
        model_id: str | None = None,
    ) -> None:
        self._inner = inner
        self._dir = cache_dir
        # Falls back to the class name, which is enough to keep the fake and the real
        # encoder from sharing entries.
        self._model = model_id or type(inner).__name__
        self._lock = threading.Lock()
        self._memory: dict[str, list[float]] = {}
        self.hits = 0
        self.misses = 0
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            log.warning("embedding cache directory unavailable; running uncached")

    @property
    def dim(self) -> int:
        return self._inner.dim

    def _key(self, kind: str, payload: bytes) -> str:
        digest = hashlib.sha256()
        digest.update(self._model.encode())
        digest.update(str(self._inner.dim).encode())
        digest.update(kind.encode())
        digest.update(payload)
        return digest.hexdigest()

    def _path(self, key: str) -> Path:
        # Shard by the first two characters: a flat directory of thousands of files is
        # slow to enumerate on Windows.
        return self._dir / key[:2] / f"{key}.json"

    def _load(self, key: str) -> list[float] | None:
        cached = self._memory.get(key)
        if cached is not None:
            return cached
        path = self._path(key)
        if not path.exists():
            return None
        try:
            vector: list[float] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if len(vector) != self._inner.dim:
            # A stale entry from a different model. Ignoring it is safe; trusting it
            # would put a wrong-width vector into the store.
            return None
        self._memory[key] = vector
        return vector

    def _store(self, key: str, vector: list[float]) -> None:
        self._memory[key] = vector
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(vector), encoding="utf-8")
        except OSError:
            # A cache that cannot be written is a performance loss, never a failure.
            log.debug("could not cache embedding %s", key, exc_info=True)

    def _cached(self, kind: str, payload: bytes, compute: Any) -> list[float]:
        key = self._key(kind, payload)
        with self._lock:
            hit = self._load(key)
        if hit is not None:
            self.hits += 1
            return hit

        self.misses += 1
        vector: list[float] = compute()
        with self._lock:
            self._store(key, vector)
        return vector

    def embed_text(self, text: str) -> list[float]:
        normalised = text.strip().lower()
        return self._cached("text", normalised.encode(), lambda: self._inner.embed_text(text))

    def embed_image(self, image: bytes) -> list[float]:
        return self._cached("image", image, lambda: self._inner.embed_image(image))

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 3) if total else 0.0
