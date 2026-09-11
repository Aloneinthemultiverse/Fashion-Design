"""Body reference by celebrity name, and RAG-Fusion query expansion.

These cover the two things the problem statement asks for that the retrieval engine did
not previously do: taking a *Western celebrity body profile* as the input, and fusing
several generated phrasings of a query rather than a single one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fashion.adapters.embed_fake import FakeEmbedder
from fashion.adapters.store_memory import InMemoryVectorStore
from fashion.adapters.vision_fake import FakeVisionModel
from fashion.core.models import (
    BodyMetrics,
    BodyShape,
    Build,
    CelebrityProfile,
    HeightBand,
    UserQuery,
)
from fashion.core.reference import BodyReferenceResolver, normalise
from fashion.pipeline.index import IndexBuilder
from fashion.pipeline.retrieve import Retriever
from tests.conftest import make_outfit


def profile(cid: str, name: str, shape: BodyShape, region: str = "american") -> CelebrityProfile:
    return CelebrityProfile(
        id=cid,
        name=name,
        region=region,
        shape=shape,
        build=Build.SLIM,
        height_band=HeightBand.TALL,
    )


@pytest.fixture
def resolver() -> BodyReferenceResolver:
    return BodyReferenceResolver(
        {
            "w1": profile("w1", "Zendaya", BodyShape.RECTANGLE),
            "w2": profile("w2", "Scarlett Johansson", BodyShape.HOURGLASS),
            "i1": profile("i1", "Aishwarya Rai", BodyShape.HOURGLASS, region="indian"),
        }
    )


# -- resolution --------------------------------------------------------------------


def test_resolves_a_western_celebrity_to_their_proportions(
    resolver: BodyReferenceResolver,
) -> None:
    """The problem statement's primary input: a Western body profile, not a photo."""
    match = resolver.resolve("Zendaya")
    assert match is not None
    assert match.exact
    assert match.to_metrics().shape is BodyShape.RECTANGLE


def test_a_named_reference_counts_as_confirmed_not_inferred(
    resolver: BodyReferenceResolver,
) -> None:
    """The user stated it, so nothing downstream should ask them to confirm it."""
    match = resolver.resolve("Zendaya")
    assert match is not None
    metrics = match.to_metrics()
    assert metrics.user_confirmed is True
    assert metrics.confidence == 1.0
    assert metrics.is_trustworthy


@pytest.mark.parametrize("typed", ["zendaya", "  ZENDAYA  ", "Zendaya"])
def test_matching_is_case_and_whitespace_insensitive(
    resolver: BodyReferenceResolver, typed: str
) -> None:
    assert resolver.resolve(typed) is not None


def test_accents_are_ignored() -> None:
    """Users rarely type diacritics, and a silent miss is worse than a loose match."""
    r = BodyReferenceResolver({"a": profile("a", "Aishwarya Rāi", BodyShape.PEAR)})
    assert r.resolve("aishwarya rai") is not None


def test_surname_only_resolves_when_unambiguous(resolver: BodyReferenceResolver) -> None:
    match = resolver.resolve("Johansson")
    assert match is not None
    assert not match.exact
    assert match.profile.name == "Scarlett Johansson"


def test_ambiguous_partials_refuse_rather_than_guess() -> None:
    """Handing back one of several candidates gives the user someone else's body."""
    r = BodyReferenceResolver(
        {
            "a": profile("a", "Kareena Kapoor", BodyShape.PEAR),
            "b": profile("b", "Karisma Kapoor", BodyShape.RECTANGLE),
        }
    )
    assert r.resolve("Kapoor") is None


def test_unknown_names_suggest_alternatives(resolver: BodyReferenceResolver) -> None:
    """An empty result with no explanation is a dead end for the user."""
    assert resolver.resolve("Scarlett Smith") is None
    assert "Scarlett Johansson" in resolver.suggest("Scarlett Smith")


def test_blank_input_resolves_to_nothing(resolver: BodyReferenceResolver) -> None:
    assert resolver.resolve("") is None
    assert resolver.resolve("   ") is None
    assert resolver.suggest("") == ()


def test_normalise_is_stable() -> None:
    assert normalise("  Déepika   PADUKONE ") == "deepika padukone"


def test_duplicate_names_keep_the_first_profile() -> None:
    """A later duplicate must not silently change what a stable query returns."""
    r = BodyReferenceResolver(
        {
            "a": profile("a", "Same Name", BodyShape.PEAR),
            "b": profile("b", "Same Name", BodyShape.APPLE),
        }
    )
    match = r.resolve("Same Name")
    assert match is not None
    assert match.profile.id == "a"


# -- RAG-Fusion query expansion ----------------------------------------------------


@pytest.fixture
def indexed(tmp_path: Path) -> tuple[FakeEmbedder, InMemoryVectorStore]:
    embedder = FakeEmbedder()
    store = InMemoryVectorStore()
    items, profiles = [], {}
    for n in range(6):
        image = tmp_path / f"o{n}.jpg"
        image.write_bytes(f"outfit-{n}".encode())
        items.append(
            make_outfit(f"o{n}", celebrity_id=f"c{n}", caption=f"outfit number {n}").model_copy(
                update={"image_path": str(image)}
            )
        )
        profiles[f"c{n}"] = profile(f"c{n}", f"Celeb {n}", BodyShape.PEAR, region="indian")
    IndexBuilder(embedder, store).build(items, profiles)
    return embedder, store


def metrics() -> BodyMetrics:
    return BodyMetrics(
        shape=BodyShape.PEAR,
        build=Build.SLIM,
        height_band=HeightBand.AVERAGE,
        user_confirmed=True,
    )


def test_expansion_adds_channels(
    indexed: tuple[FakeEmbedder, InMemoryVectorStore],
) -> None:
    """RAG-Fusion retrieves for several phrasings, not just the one typed."""
    embedder, store = indexed
    result = Retriever(embedder, store, vision=FakeVisionModel()).retrieve(
        metrics(), UserQuery(text="a festive saree")
    )
    variant_channels = [c for c in result.channels_used if "#" in c]
    assert variant_channels, "expansion produced no additional channels"


def test_without_an_llm_only_the_original_query_is_used(
    indexed: tuple[FakeEmbedder, InMemoryVectorStore],
) -> None:
    """Losing recall is acceptable degradation; failing the search is not."""
    embedder, store = indexed
    result = Retriever(embedder, store).retrieve(metrics(), UserQuery(text="a saree"))
    assert set(result.channels_used) == {"caption", "cross"}


def test_a_failing_expansion_falls_back_to_the_original(
    indexed: tuple[FakeEmbedder, InMemoryVectorStore],
) -> None:
    embedder, store = indexed

    class Broken(FakeVisionModel):
        def expand_query(self, text: str, n: int = 3) -> tuple[str, ...]:
            raise RuntimeError("quota exhausted")

    result = Retriever(embedder, store, vision=Broken()).retrieve(
        metrics(), UserQuery(text="a saree")
    )
    assert set(result.channels_used) == {"caption", "cross"}
    assert result.recommendations


def test_the_users_own_wording_is_always_the_first_variant() -> None:
    """Expansions widen recall; they must not replace what was actually asked."""
    variants = FakeVisionModel().expand_query("a red lehenga", 3)
    assert variants[0] == "a red lehenga"


def test_expansion_variants_are_distinct() -> None:
    variants = FakeVisionModel().expand_query("a red lehenga", 4)
    assert len(set(variants)) == len(variants)


# -- reference image ---------------------------------------------------------------


def test_a_reference_outfit_drives_the_image_channel(
    indexed: tuple[FakeEmbedder, InMemoryVectorStore],
) -> None:
    """Image-to-image: 'find outfits like this one'."""
    embedder, store = indexed
    result = Retriever(embedder, store).retrieve(
        metrics(), UserQuery(text=""), reference_image=b"outfit-3"
    )
    assert result.channels_used == ("image",)
    assert result.recommendations[0].outfit.id == "o3"


def test_a_reference_outfit_outranks_the_body_photo(
    indexed: tuple[FakeEmbedder, InMemoryVectorStore],
) -> None:
    """A selfie shares background and framing with nothing; a reference is a statement."""
    embedder, store = indexed
    result = Retriever(embedder, store).retrieve(
        metrics(), UserQuery(text=""), photo=b"outfit-0", reference_image=b"outfit-4"
    )
    assert result.recommendations[0].outfit.id == "o4"
