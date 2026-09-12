"""Protocol conformance.

The ports design only pays off if adapters genuinely satisfy their protocols. Python
structural typing is checked statically, not at import time, so without these tests an
adapter could drift out of conformance and only fail at runtime in production -- which,
for the Colab and Gemini adapters, means failing on the machine that has the API key
rather than in CI.
"""

from __future__ import annotations

from fashion.adapters.embed_fake import FakeEmbedder
from fashion.adapters.gen_null import NullGenerationProvider, NullTryOnProvider
from fashion.adapters.store_memory import InMemoryVectorStore
from fashion.adapters.vision_fake import FakeVisionModel
from fashion.ports.embedder import Embedder
from fashion.ports.generation import GenerationProvider
from fashion.ports.tryon import TryOnProvider
from fashion.ports.vectorstore import VectorStore
from fashion.ports.vision import VisionModel


def test_adapters_satisfy_their_ports() -> None:
    assert isinstance(FakeVisionModel(), VisionModel)
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(InMemoryVectorStore(), VectorStore)
    assert isinstance(NullGenerationProvider(), GenerationProvider)
    assert isinstance(NullTryOnProvider(), TryOnProvider)


def test_fake_vision_is_deterministic() -> None:
    """Golden-set tests and cached VLM responses both depend on this."""
    vlm = FakeVisionModel()
    assert vlm.analyze_body(b"photo") == vlm.analyze_body(b"photo")
    assert vlm.caption_outfit(b"x") == vlm.caption_outfit(b"x")


def test_fake_vision_discriminates_between_inputs() -> None:
    """A fake that returned one constant would let broken wiring pass every test."""
    vlm = FakeVisionModel()
    shapes = {vlm.analyze_body(f"person{i}".encode()).shape for i in range(25)}
    assert len(shapes) > 1


def test_fake_embedder_returns_unit_vectors_of_declared_dim() -> None:
    embedder = FakeEmbedder()
    for vec in (embedder.embed_text("a red saree"), embedder.embed_image(b"bytes")):
        assert len(vec) == embedder.dim
        assert abs(sum(v * v for v in vec) ** 0.5 - 1.0) < 1e-9


def test_fake_embedder_is_stable_across_whitespace_and_case() -> None:
    """Caption text arrives inconsistently from the VLM; embeddings must not wobble."""
    embedder = FakeEmbedder()
    assert embedder.embed_text("Red Saree") == embedder.embed_text("  red saree ")


def test_null_providers_report_unavailable_rather_than_raising() -> None:
    """Graceful degradation: absence of a GPU is a normal outcome, not an error."""
    gen = NullGenerationProvider()
    assert gen.available is False
    assert gen.supports_references is False
    assert gen.generate("a red lehenga", references=(b"ref",)) is None
    assert NullTryOnProvider().try_on(b"person", b"garment") is None


def test_missing_concepts_carry_a_denser_caption_than_the_concept() -> None:
    """The ImageRAG result hinges on retrieving by dense caption, not by concept name."""
    gaps = FakeVisionModel().find_gaps(b"odd-hash-input-1", "a lehenga with a dupatta")
    assert gaps, "fixture must actually produce a gap, or this test asserts nothing"
    for gap in gaps:
        assert len(gap.retrieval_caption) > len(gap.concept)


def test_media_type_is_detected_from_content() -> None:
    """Declaring the wrong media type is rejected outright by the API.

    Hardcoding image/jpeg worked only because the seed corpus happens to be JPEG, and
    failed on the first PNG a user uploaded -- the input that matters most.
    """
    from fashion.adapters.vision_proxy import media_type

    assert media_type(b"\x89PNG\r\n\x1a\n" + b"x" * 32) == "image/png"
    assert media_type(b"\xff\xd8\xff\xe0" + b"x" * 32) == "image/jpeg"
    assert media_type(b"RIFF" + b"\x00" * 4 + b"WEBP" + b"x" * 32) == "image/webp"
    assert media_type(b"GIF89a" + b"x" * 32) == "image/gif"
    # Unknown content falls back rather than raising: a wrong guess is recoverable,
    # a crash in the upload path is not.
    assert media_type(b"not an image") == "image/jpeg"
