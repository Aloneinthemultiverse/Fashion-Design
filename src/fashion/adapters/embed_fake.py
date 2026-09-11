"""Deterministic fake embedder.

Hash-derived unit vectors. Not semantically meaningful, but stable and well-distributed,
which is all the store and ranking tests need. Text and image share one derivation, so
identical content embeds identically across the two modalities -- that lets tests assert
the cross-modal retrieval wiring without downloading a real CLIP model.
"""

from __future__ import annotations

import hashlib
import math

DIM = 64


class FakeEmbedder:
    @property
    def dim(self) -> int:
        return DIM

    def _vector(self, payload: bytes) -> list[float]:
        out: list[float] = []
        counter = 0
        while len(out) < DIM:
            digest = hashlib.sha256(payload + counter.to_bytes(4, "big")).digest()
            out.extend((b - 127.5) / 127.5 for b in digest)
            counter += 1
        vec = out[:DIM]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_text(self, text: str) -> list[float]:
        return self._vector(text.strip().lower().encode())

    def embed_image(self, image: bytes) -> list[float]:
        return self._vector(image)
