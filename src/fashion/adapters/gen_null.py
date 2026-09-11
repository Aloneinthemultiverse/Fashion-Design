"""Generation and try-on providers that do nothing, successfully.

These are the defaults. The architecture requires graceful degradation, and the cleanest
way to guarantee it is to make "no GPU available" the normal path rather than an error
path: callers already handle None, so nothing special happens when the Colab worker is
down.
"""

from __future__ import annotations


class NullGenerationProvider:
    @property
    def available(self) -> bool:
        return False

    @property
    def supports_references(self) -> bool:
        return False

    def generate(
        self,
        prompt: str,
        *,
        references: tuple[bytes, ...] = (),
        seed: int | None = None,
    ) -> bytes | None:
        return None


class NullTryOnProvider:
    @property
    def available(self) -> bool:
        return False

    def try_on(self, person: bytes, garment: bytes) -> bytes | None:
        """None tells the UI to show a side-by-side comparison instead."""
        return None
