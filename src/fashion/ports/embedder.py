"""Embedding port."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """A CLIP-style dual encoder.

    Both methods must return vectors of `dim` length in a *shared* space — text-to-image
    retrieval depends on that, and the vector store's `img_vec` channel is queried with
    text embeddings.
    """

    @property
    def dim(self) -> int: ...

    def embed_text(self, text: str) -> list[float]: ...

    def embed_image(self, image: bytes) -> list[float]: ...
