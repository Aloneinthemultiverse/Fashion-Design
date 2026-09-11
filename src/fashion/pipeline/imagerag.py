"""ImageRAG: retrieval-augmented outfit generation.

Implements arXiv 2502.09411. The loop:

1. Generate an initial image from the prompt.
2. Ask the VLM which concepts the prompt requires but the image does not show.
3. For each missing concept, the VLM writes a **dense retrieval caption**.
4. Retrieve reference images using those captions.
5. Re-generate with the references as image conditioning (IP-Adapter).
6. Repeat until no gaps remain, or the iteration budget is spent.

Step 3 is the load-bearing one and the easiest to get wrong. The paper's measured result
is that retrieving on a rich per-concept caption beats retrieving on the bare concept
name or on the original prompt, so the caption is what reaches the retriever -- never the
concept string. `MissingConcept` enforces that the caption is at least as detailed as the
concept it describes.

The method is training-free and uses only off-the-shelf parts, which is what makes it
viable here: the retrieval index already exists, and the generator is swappable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from fashion.core.models import BodyMetrics, MissingConcept, UserQuery
from fashion.ports.embedder import Embedder
from fashion.ports.generation import GenerationProvider
from fashion.ports.vectorstore import IMAGE_VECTOR, Filter, VectorStore
from fashion.ports.vision import VisionModel

log = logging.getLogger(__name__)

MAX_ITERATIONS = 2
MAX_REFERENCES = 5
REFERENCES_PER_CONCEPT = 2


@dataclass(frozen=True, slots=True)
class Round:
    """What happened in one pass of the loop, for display and debugging."""

    index: int
    gaps: tuple[MissingConcept, ...]
    reference_ids: tuple[str, ...]
    produced_image: bool


@dataclass(frozen=True, slots=True)
class ImageRagResult:
    image: bytes | None
    rounds: tuple[Round, ...] = ()
    references: tuple[str, ...] = ()
    unavailable_reason: str | None = None

    @property
    def converged(self) -> bool:
        """True when the final round found no remaining gaps."""
        return bool(self.rounds) and not self.rounds[-1].gaps


@dataclass
class ImageRagGenerator:
    vision: VisionModel
    embedder: Embedder
    store: VectorStore
    generator: GenerationProvider
    max_iterations: int = MAX_ITERATIONS
    max_references: int = MAX_REFERENCES
    _seen: set[str] = field(default_factory=set, init=False)

    def _retrieve_references(
        self, gaps: tuple[MissingConcept, ...], where: Filter
    ) -> tuple[list[str], list[bytes]]:
        """Find reference images for each gap, using its dense caption."""
        ids: list[str] = []
        images: list[bytes] = []

        for gap in gaps:
            if len(ids) >= self.max_references:
                break
            # The caption, not gap.concept -- this is the paper's central finding.
            vector = self.embedder.embed_text(gap.retrieval_caption)
            hits = self.store.search(
                vector, using=IMAGE_VECTOR, limit=REFERENCES_PER_CONCEPT, where=where
            )
            for hit in hits:
                if len(ids) >= self.max_references or hit.id in self._seen:
                    continue
                path = Path(str(hit.payload.get("image_path", "")))
                if not path.exists():
                    continue
                try:
                    images.append(path.read_bytes())
                except OSError:
                    continue
                ids.append(hit.id)
                # Track across rounds so a second pass brings genuinely new references
                # rather than re-conditioning on what already failed to close the gap.
                self._seen.add(hit.id)

        return ids, images

    def generate(
        self,
        query: UserQuery,
        metrics: BodyMetrics,
        *,
        prompt: str | None = None,
    ) -> ImageRagResult:
        if not self.generator.available:
            # Not an error: the null provider is the default, and callers fall back to
            # showing retrieved reference outfits instead of a generated one.
            return ImageRagResult(
                image=None,
                unavailable_reason=(
                    "No generation backend is configured. Set FASHION_GENERATION_PROVIDER "
                    "and FASHION_COLAB_WORKER_URL to enable generated previews."
                ),
            )

        self._seen.clear()
        text = prompt or self._build_prompt(query, metrics)
        where = Filter(
            must_equal={
                key: value
                for key, value in (
                    ("body_shape", metrics.shape.value),
                    ("culture", query.culture.value if query.culture else ""),
                )
                if value
            }
        )

        image = self.generator.generate(text)
        if image is None:
            return ImageRagResult(image=None, unavailable_reason="Initial generation failed.")

        rounds: list[Round] = []
        all_references: list[str] = []

        for index in range(self.max_iterations):
            gaps = self.vision.find_gaps(image, text)
            if not gaps:
                rounds.append(Round(index, (), (), produced_image=True))
                break

            ids, reference_images = self._retrieve_references(gaps, where)
            if not reference_images:
                # Nothing in the corpus depicts the missing concept. Iterating again
                # would re-run the same failing retrieval, so stop and report the gap.
                log.info("no references found for gaps: %s", [g.concept for g in gaps])
                rounds.append(Round(index, gaps, (), produced_image=False))
                break

            if not self.generator.supports_references:
                # Conditioning is what makes this ImageRAG rather than a text prompt;
                # silently dropping the references would misrepresent the result.
                rounds.append(Round(index, gaps, tuple(ids), produced_image=False))
                break

            regenerated = self.generator.generate(text, references=tuple(reference_images))
            all_references.extend(ids)
            rounds.append(Round(index, gaps, tuple(ids), produced_image=regenerated is not None))
            if regenerated is None:
                break
            image = regenerated

        return ImageRagResult(image=image, rounds=tuple(rounds), references=tuple(all_references))

    @staticmethod
    def _build_prompt(query: UserQuery, metrics: BodyMetrics) -> str:
        """Compose a prompt from the user's request and their body geometry.

        Shape is stated as the styling *goal* rather than as a label, because a
        generator responds to a description of a garment's cut far better than to a
        taxonomy term like "pear".
        """
        from fashion.core.crosscultural import GUIDANCE

        goal = GUIDANCE[metrics.shape].goal
        parts = [query.text.strip() or "an outfit"]
        if query.culture:
            parts.append(f"{query.culture.value} style")
        if query.occasion:
            parts.append(f"for a {query.occasion.value} occasion")
        parts.append(f"cut to {goal}")
        return ", ".join(parts)
