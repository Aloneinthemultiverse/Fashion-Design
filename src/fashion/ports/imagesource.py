"""Image source port.

Exists so the corpus can come from a curated directory now and, if the licensing and
terms-of-service questions are ever resolved, from somewhere else later without the
pipeline changing. No scraping adapter is implemented.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SourcedImage:
    """An image plus the provenance needed to decide whether it may be used."""

    local_path: str
    source: str
    license: str
    celebrity_hint: str | None = None


@runtime_checkable
class ImageSource(Protocol):
    def fetch(self, celebrity: str, *, limit: int = 10) -> Iterator[SourcedImage]: ...
