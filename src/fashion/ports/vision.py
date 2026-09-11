"""Vision-language model port."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from fashion.core.models import BodyMetrics, MissingConcept, OutfitItem


@runtime_checkable
class VisionModel(Protocol):
    """Everything the system asks a VLM to do.

    Deliberately narrow and task-shaped rather than a generic `complete(prompt)`: each
    method has a fixed output schema, which is what lets the Gemini adapter validate and
    retry, and lets the fake adapter be trivially deterministic.
    """

    def analyze_body(self, image: bytes) -> BodyMetrics:
        """Infer body geometry from a full-body photo."""
        ...

    def tag_outfit(self, image: bytes) -> dict[str, object]:
        """Extract structured outfit attributes, for the labelling tool to pre-fill."""
        ...

    def caption_outfit(self, image: bytes) -> str:
        """Write a dense caption for cap_vec embedding and ImageRAG retrieval."""
        ...

    def find_gaps(self, generated: bytes, prompt: str) -> tuple[MissingConcept, ...]:
        """List concepts present in the prompt but absent from the generated image.

        Returns an empty tuple when the image already satisfies the prompt, which
        short-circuits the ImageRAG loop.
        """
        ...

    def write_rationale(self, outfit: OutfitItem, metrics: BodyMetrics) -> str:
        """Explain why this outfit suits this body."""
        ...

    def expand_query(self, text: str, n: int = 3) -> tuple[str, ...]:
        """Rewrite one request as several differently-phrased searches.

        This is the generation half of RAG-Fusion (Raudaschl, 2023): one query is
        expanded into several, each is retrieved for, and the ranked lists are fused.
        Without it, fusing retrieval channels alone is hybrid search -- the expansion
        is what makes it RAG-Fusion, and it is what surfaces items the user's original
        wording would have missed.
        """
        ...
