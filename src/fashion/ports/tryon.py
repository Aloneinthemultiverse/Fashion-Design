"""Virtual try-on port."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TryOnProvider(Protocol):
    @property
    def available(self) -> bool: ...

    def try_on(self, person: bytes, garment: bytes) -> bytes | None:
        """Render `person` wearing `garment`; None when unavailable.

        A None result is a normal outcome, not an error — the UI falls back to a
        side-by-side comparison, per the architecture doc's graceful-degradation rule.
        """
        ...
