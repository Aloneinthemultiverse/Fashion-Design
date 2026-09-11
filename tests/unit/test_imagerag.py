"""ImageRAG loop.

The behaviour that decides whether this is really ImageRAG or just a prompt is *what
gets sent to the retriever*. The paper's measured gain comes from retrieving on a dense
per-concept caption, so several tests assert exactly that, and one asserts the loop
refuses to pretend when the backend cannot consume references at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fashion.adapters.embed_fake import FakeEmbedder
from fashion.adapters.gen_null import NullGenerationProvider
from fashion.adapters.store_memory import InMemoryVectorStore
from fashion.adapters.vision_fake import FakeVisionModel
from fashion.core.models import (
    BodyMetrics,
    BodyShape,
    Build,
    HeightBand,
    MissingConcept,
    UserQuery,
)
from fashion.pipeline.imagerag import ImageRagGenerator
from fashion.ports.vectorstore import OutfitVectors


class RecordingGenerator:
    """A generator that always succeeds and records what it was given."""

    def __init__(self, *, supports_references: bool = True) -> None:
        self._supports = supports_references
        self.calls: list[tuple[str, int]] = []

    @property
    def available(self) -> bool:
        return True

    @property
    def supports_references(self) -> bool:
        return self._supports

    def generate(
        self, prompt: str, *, references: tuple[bytes, ...] = (), seed: int | None = None
    ) -> bytes | None:
        self.calls.append((prompt, len(references)))
        # Odd length keeps FakeVisionModel.find_gaps reporting a gap; see below.
        return b"generated-image-odd" if len(self.calls) == 1 else b"generated-final"


class GapThenClean:
    """Reports one gap on the first look and none afterwards."""

    def __init__(self) -> None:
        self.captions_requested = 0

    def find_gaps(self, generated: bytes, prompt: str) -> tuple[MissingConcept, ...]:
        self.captions_requested += 1
        if self.captions_requested > 1:
            return ()
        return (
            MissingConcept(
                concept="dupatta",
                retrieval_caption=(
                    "A sheer embroidered dupatta draped over one shoulder, falling to "
                    "knee length, in a contrasting colour."
                ),
            ),
        )


class RecordingEmbedder(FakeEmbedder):
    """Captures every text sent to the retriever."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.texts.append(text)
        return super().embed_text(text)


@pytest.fixture
def store(tmp_path: Path) -> InMemoryVectorStore:
    embedder = FakeEmbedder()
    store = InMemoryVectorStore()
    store.ensure_collection(embedder.dim)
    for item_id in ("ref1", "ref2", "ref3"):
        image = tmp_path / f"{item_id}.jpg"
        image.write_bytes(f"bytes-{item_id}".encode())
        store.upsert(
            item_id,
            OutfitVectors(
                embedder.embed_image(image.read_bytes()),
                embedder.embed_text(f"caption for {item_id}"),
            ),
            {"body_shape": "pear", "image_path": str(image), "culture": "ethnic"},
        )
    return store


def metrics() -> BodyMetrics:
    return BodyMetrics(
        shape=BodyShape.PEAR,
        build=Build.SLIM,
        height_band=HeightBand.AVERAGE,
        user_confirmed=True,
    )


def test_retrieval_uses_the_dense_caption_not_the_concept_name(
    store: InMemoryVectorStore,
) -> None:
    """This is the paper's central finding and the easiest thing to get wrong."""
    embedder = RecordingEmbedder()
    vision = FakeVisionModel()
    vision.find_gaps = GapThenClean().find_gaps  # type: ignore[method-assign]

    ImageRagGenerator(vision, embedder, store, RecordingGenerator()).generate(
        UserQuery(text="a lehenga"), metrics()
    )

    assert any("sheer embroidered dupatta" in t for t in embedder.texts)
    assert "dupatta" not in embedder.texts  # never the bare concept alone


def test_references_are_passed_to_the_regeneration(store: InMemoryVectorStore) -> None:
    vision = FakeVisionModel()
    vision.find_gaps = GapThenClean().find_gaps  # type: ignore[method-assign]
    generator = RecordingGenerator()

    result = ImageRagGenerator(vision, FakeEmbedder(), store, generator).generate(
        UserQuery(text="a lehenga"), metrics()
    )

    assert generator.calls[0][1] == 0  # initial pass has no references
    assert generator.calls[1][1] > 0  # regeneration is conditioned on them
    assert result.references


def test_loop_stops_once_no_gaps_remain(store: InMemoryVectorStore) -> None:
    vision = FakeVisionModel()
    vision.find_gaps = GapThenClean().find_gaps  # type: ignore[method-assign]

    result = ImageRagGenerator(vision, FakeEmbedder(), store, RecordingGenerator()).generate(
        UserQuery(text="a lehenga"), metrics()
    )
    assert result.converged
    assert len(result.rounds) <= 2


def test_a_clean_first_generation_skips_retrieval_entirely(
    store: InMemoryVectorStore,
) -> None:
    """No gaps means no reason to spend retrieval or a second generation."""
    vision = FakeVisionModel()
    vision.find_gaps = lambda image, prompt: ()  # type: ignore[method-assign]
    generator = RecordingGenerator()

    result = ImageRagGenerator(vision, FakeEmbedder(), store, generator).generate(
        UserQuery(text="a lehenga"), metrics()
    )
    assert len(generator.calls) == 1
    assert result.converged
    assert result.references == ()


def test_generator_without_reference_support_stops_rather_than_pretending(
    store: InMemoryVectorStore,
) -> None:
    """Dropping references turns ImageRAG back into a plain prompt; say so instead."""
    vision = FakeVisionModel()
    vision.find_gaps = GapThenClean().find_gaps  # type: ignore[method-assign]
    generator = RecordingGenerator(supports_references=False)

    result = ImageRagGenerator(vision, FakeEmbedder(), store, generator).generate(
        UserQuery(text="a lehenga"), metrics()
    )
    assert len(generator.calls) == 1
    assert result.rounds[-1].gaps
    assert not result.converged


def test_null_provider_degrades_gracefully(store: InMemoryVectorStore) -> None:
    """No GPU is the default state, not an error path."""
    result = ImageRagGenerator(
        FakeVisionModel(), FakeEmbedder(), store, NullGenerationProvider()
    ).generate(UserQuery(text="a lehenga"), metrics())

    assert result.image is None
    assert result.unavailable_reason is not None
    assert "FASHION_GENERATION_PROVIDER" in result.unavailable_reason


def test_empty_corpus_stops_the_loop_instead_of_spinning() -> None:
    """Re-running a retrieval that found nothing would just fail again."""
    empty = InMemoryVectorStore()
    empty.ensure_collection(FakeEmbedder().dim)
    vision = FakeVisionModel()
    vision.find_gaps = GapThenClean().find_gaps  # type: ignore[method-assign]
    generator = RecordingGenerator()

    result = ImageRagGenerator(vision, FakeEmbedder(), empty, generator).generate(
        UserQuery(text="a lehenga"), metrics()
    )
    assert len(generator.calls) == 1
    assert result.rounds[-1].reference_ids == ()


def test_prompt_states_the_styling_goal_rather_than_the_shape_label() -> None:
    """Generators respond to a description of a cut, not to the word "pear"."""
    prompt = ImageRagGenerator._build_prompt(UserQuery(text="a lehenga"), metrics())
    assert "lehenga" in prompt
    assert "skim the hip" in prompt
    assert "pear" not in prompt


def test_references_are_not_reused_across_rounds(store: InMemoryVectorStore) -> None:
    """A second pass should bring new evidence, not re-condition on what already failed."""
    always_gap = FakeVisionModel()
    always_gap.find_gaps = lambda image, prompt: (  # type: ignore[method-assign]
        MissingConcept(
            concept="dupatta",
            retrieval_caption="A long sheer embroidered dupatta in a contrasting colour.",
        ),
    )
    result = ImageRagGenerator(
        always_gap, FakeEmbedder(), store, RecordingGenerator(), max_iterations=2
    ).generate(UserQuery(text="a lehenga"), metrics())

    assert len(result.references) == len(set(result.references))
