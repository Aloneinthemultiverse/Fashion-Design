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
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from fashion.core.crosscultural import adjustments, explain
from fashion.core.models import (
    BodyMetrics,
    MissingConcept,
    OutfitItem,
    Recommendation,
    UserQuery,
    Wardrobe,
)
from fashion.pipeline.retrieve import _outfit_from_payload
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

        # Name the wearer first. Without it the generator picks one, and it picked a
        # woman for a male user -- the single most visibly wrong output this system has
        # produced.
        wardrobe = query.wardrobe or metrics.wardrobe
        subject = {
            Wardrobe.MENSWEAR: "a man wearing",
            Wardrobe.WOMENSWEAR: "a woman wearing",
        }.get(wardrobe, "a person wearing")

        # Name concrete garments for the wardrobe. An abstract brief leaves the
        # generator to invent one, and what it invents is costume rather than clothing.
        anchor = {
            Wardrobe.MENSWEAR: (
                "such as a bandhgala jacket over straight trousers, a nehru-collar "
                "kurta with tailored trousers, or an embroidered sherwani"
            ),
            Wardrobe.WOMENSWEAR: (
                "such as a cape lehenga, a draped saree gown, or an anarkali with "
                "a contemporary cut"
            ),
        }.get(wardrobe, "")

        parts = [f"{subject} {query.text.strip() or 'an outfit'}"]
        if anchor:
            parts.append(anchor)
        if query.culture:
            parts.append(f"{query.culture.value} style")
        if query.occasion:
            parts.append(f"for a {query.occasion.value} occasion")
        parts.append(f"cut to {goal}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------------
# Retrieval-only ImageRAG
# ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Gap:
    """One unmet part of the request, and what was found to cover it."""

    concept: str
    retrieval_caption: str
    filled_by: tuple[str, ...] = ()

    @property
    def filled(self) -> bool:
        return bool(self.filled_by)


@dataclass(frozen=True, slots=True)
class RefinementResult:
    added: tuple[Recommendation, ...] = ()
    gaps: tuple[Gap, ...] = ()
    rounds: int = 0
    skipped_reason: str | None = None

    @property
    def unmet(self) -> tuple[str, ...]:
        """Concepts the corpus could not supply -- worth telling the user about."""
        return tuple(g.concept for g in self.gaps if not g.filled)


@dataclass
class ImageRagRefiner:
    """ImageRAG applied to retrieval rather than generation.

    The paper defines its loop against a *generated* image: generate, find what the
    prompt asked for and the image lacks, write a dense caption per gap, retrieve
    references, regenerate. The contribution is the middle of that -- dense per-concept
    captions retrieve better than concept names or the raw prompt -- and it does not
    depend on there being a generator at all.

    So the same loop runs here over a result set: retrieve, ask what the request wanted
    that the results do not show, write dense captions, retrieve again for those. It
    needs only an LLM, CLIP and the vector store, which means it runs on a CPU-only
    deployment where the generation loop cannot.

    Gaps the corpus genuinely cannot fill are reported rather than hidden. "No outfit
    here has a dupatta" is useful; silently returning the nearest thing is not.
    """

    vision: VisionModel
    embedder: Embedder
    store: VectorStore
    max_rounds: int = 2
    per_gap: int = 2

    def refine(
        self,
        request: str,
        metrics: BodyMetrics,
        existing: Sequence[Recommendation],
        *,
        where: Filter | None = None,
    ) -> RefinementResult:
        request = request.strip()
        if not request:
            return RefinementResult(skipped_reason="No text request to analyse.")

        seen = {rec.outfit.id for rec in existing}
        described = [_describe(rec.outfit) for rec in existing]
        added: list[Recommendation] = []
        gaps: list[Gap] = []
        rounds = 0

        for _ in range(self.max_rounds):
            try:
                missing = self.vision.find_missing_concepts(request, tuple(described))
            except Exception:
                log.warning("gap analysis failed", exc_info=True)
                return RefinementResult(
                    added=tuple(added),
                    gaps=tuple(gaps),
                    rounds=rounds,
                    skipped_reason="Gap analysis was unavailable.",
                )

            rounds += 1
            # Re-reporting a gap already examined would loop on the same failed
            # retrieval, so only genuinely new concepts continue the loop.
            fresh = [m for m in missing if m.concept not in {g.concept for g in gaps}]
            if not fresh:
                break

            for concept in fresh:
                # The dense caption, never the bare concept name. This is the paper's
                # measured result and the whole reason the step exists.
                vector = self.embedder.embed_text(concept.retrieval_caption)
                hits = self.store.search(
                    vector, using=IMAGE_VECTOR, limit=self.per_gap + len(seen), where=where
                )
                filled: list[str] = []
                for hit in hits:
                    if hit.id in seen or len(filled) >= self.per_gap:
                        continue
                    item = _outfit_from_payload(hit.id, hit.payload)
                    if item is None:
                        continue
                    seen.add(hit.id)
                    filled.append(hit.id)
                    added.append(
                        Recommendation(
                            outfit=item,
                            score=round(hit.score * 0.5, 6),
                            rationale=explain(item, metrics.shape),
                            adjustments=adjustments(item, metrics.shape),
                            # Named so the UI can say *why* this one appeared: it was
                            # retrieved to cover a specific unmet part of the request.
                            matched_on=(f"gap:{concept.concept}",),
                        )
                    )
                    described.append(_describe(item))

                gaps.append(
                    Gap(
                        concept=concept.concept,
                        retrieval_caption=concept.retrieval_caption,
                        filled_by=tuple(filled),
                    )
                )

        return RefinementResult(added=tuple(added), gaps=tuple(gaps), rounds=rounds)


def _describe(outfit: OutfitItem) -> str:
    """A compact textual view of an outfit, for the gap-analysis prompt.

    Built from structured attributes rather than the stored caption so the analysis
    does not inherit whatever a captioning pass got wrong.
    """
    colours = ", ".join(outfit.colors) or "unspecified colour"
    return (
        f"{outfit.garment_type} ({outfit.culture.value}), "
        f"{outfit.silhouette.value.replace('_', '-')} silhouette, "
        f"{outfit.neckline.value.replace('_', '-')} neckline, "
        f"{outfit.waist_emphasis.value} waist, {colours}"
    )
