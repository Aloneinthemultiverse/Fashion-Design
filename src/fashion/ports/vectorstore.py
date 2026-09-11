"""Vector store port."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

IMAGE_VECTOR = "img_vec"
CAPTION_VECTOR = "cap_vec"


@dataclass(frozen=True, slots=True)
class OutfitVectors:
    """The two named vectors stored per outfit.

    `img_vec` (CLIP image tower) answers image-to-image queries and, because CLIP shares
    one space across towers, cross-modal text-to-image queries too. `cap_vec` (CLIP text
    tower over the VLM caption) answers text-to-text queries, which resolve fine garment
    attributes — neckline, silhouette, fabric — more sharply than cross-modal matching.
    """

    img_vec: list[float]
    cap_vec: list[float]


@dataclass(frozen=True, slots=True)
class SearchHit:
    id: str
    score: float
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Filter:
    """Structured pre-filter applied before vector ranking.

    Body match is a hard constraint, not a soft preference: an outfit cut for a pear
    shape does not become right for an apple shape because the colours happen to match.
    Applying it as a filter rather than a score term makes that non-negotiable.
    """

    must_equal: dict[str, str] = field(default_factory=dict)
    must_be_in: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def matches(self, payload: dict[str, object]) -> bool:
        """Evaluate this filter in Python — used by the in-memory store and by tests."""
        for key, value in self.must_equal.items():
            if str(payload.get(key, "")) != value:
                return False
        return all(str(payload.get(key, "")) in allowed for key, allowed in self.must_be_in.items())


@runtime_checkable
class VectorStore(Protocol):
    def ensure_collection(self, dim: int) -> None: ...

    def upsert(self, item_id: str, vectors: OutfitVectors, payload: dict[str, object]) -> None: ...

    def search(
        self,
        vector: list[float],
        *,
        using: str = IMAGE_VECTOR,
        limit: int = 10,
        where: Filter | None = None,
    ) -> list[SearchHit]: ...

    def get(self, item_id: str) -> dict[str, object] | None: ...

    def count(self) -> int: ...
