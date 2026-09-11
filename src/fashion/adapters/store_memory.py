"""In-memory vector store.

Brute-force cosine over a dict. Correct and dependency-free, which makes it the right
store for tests and small seed sets; it is O(n) per query, so Qdrant takes over once the
corpus grows. Both implement the same port, so swapping is a config change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from fashion.ports.vectorstore import IMAGE_VECTOR, Filter, OutfitVectors, SearchHit


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@dataclass
class InMemoryVectorStore:
    _vectors: dict[str, OutfitVectors] = field(default_factory=dict)
    _payloads: dict[str, dict[str, object]] = field(default_factory=dict)
    _dim: int | None = None

    def ensure_collection(self, dim: int) -> None:
        self._dim = dim

    def upsert(self, item_id: str, vectors: OutfitVectors, payload: dict[str, object]) -> None:
        if self._dim is not None:
            for name, vec in (("img_vec", vectors.img_vec), ("cap_vec", vectors.cap_vec)):
                if len(vec) != self._dim:
                    raise ValueError(f"{name} has dim {len(vec)}, collection expects {self._dim}")
        self._vectors[item_id] = vectors
        self._payloads[item_id] = dict(payload)

    def search(
        self,
        vector: list[float],
        *,
        using: str = IMAGE_VECTOR,
        limit: int = 10,
        where: Filter | None = None,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for item_id, vectors in self._vectors.items():
            payload = self._payloads[item_id]
            if where is not None and not where.matches(payload):
                continue
            target = vectors.img_vec if using == IMAGE_VECTOR else vectors.cap_vec
            hits.append(SearchHit(item_id, cosine(vector, target), payload))
        # Tie-break on id so results are deterministic for golden-set tests.
        hits.sort(key=lambda h: (-h.score, h.id))
        return hits[:limit]

    def get(self, item_id: str) -> dict[str, object] | None:
        return self._payloads.get(item_id)

    def count(self) -> int:
        return len(self._vectors)
