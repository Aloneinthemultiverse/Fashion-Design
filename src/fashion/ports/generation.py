"""Image generation port."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GenerationProvider(Protocol):
    """Text-to-image with optional image conditioning.

    `references` carries the ImageRAG retrieved examples, injected via IP-Adapter by the
    real backend. A provider that cannot condition on images must report
    `supports_references = False` rather than silently ignoring them — dropping the
    references turns ImageRAG back into a plain text prompt, and the caller needs to know.
    """

    @property
    def available(self) -> bool: ...

    @property
    def supports_references(self) -> bool: ...

    def generate(
        self,
        prompt: str,
        *,
        references: tuple[bytes, ...] = (),
        seed: int | None = None,
    ) -> bytes | None:
        """Returns PNG bytes, or None when the provider is unavailable."""
        ...
