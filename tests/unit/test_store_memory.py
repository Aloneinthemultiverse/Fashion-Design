"""In-memory vector store, including the two-vector and hard-filter behaviour."""

from __future__ import annotations

import pytest

from fashion.adapters.embed_fake import FakeEmbedder
from fashion.adapters.store_memory import InMemoryVectorStore
from fashion.ports.vectorstore import (
    CAPTION_VECTOR,
    IMAGE_VECTOR,
    Filter,
    OutfitVectors,
)


@pytest.fixture
def populated() -> tuple[InMemoryVectorStore, FakeEmbedder]:
    embedder = FakeEmbedder()
    store = InMemoryVectorStore()
    store.ensure_collection(embedder.dim)
    for item_id, shape, caption in [
        ("o1", "pear", "a flowing a-line anarkali in crimson"),
        ("o2", "pear", "a structured bodycon gown in black"),
        ("o3", "apple", "an empire-line kurta in ivory"),
    ]:
        store.upsert(
            item_id,
            OutfitVectors(
                img_vec=embedder.embed_image(item_id.encode()),
                cap_vec=embedder.embed_text(caption),
            ),
            {"body_shape": shape, "caption": caption},
        )
    return store, embedder


def test_image_to_image_retrieval_finds_itself_first(
    populated: tuple[InMemoryVectorStore, FakeEmbedder],
) -> None:
    """Querying img_vec with an item's own image embedding must rank it top."""
    store, embedder = populated
    hits = store.search(embedder.embed_image(b"o2"), using=IMAGE_VECTOR)
    assert hits[0].id == "o2"
    assert hits[0].score == pytest.approx(1.0)


def test_text_to_text_retrieval_uses_the_caption_vector(
    populated: tuple[InMemoryVectorStore, FakeEmbedder],
) -> None:
    """The same query must resolve against cap_vec, not img_vec."""
    store, embedder = populated
    query = embedder.embed_text("an empire-line kurta in ivory")
    assert store.search(query, using=CAPTION_VECTOR)[0].id == "o3"
    # The identical vector against img_vec should not produce that near-perfect match,
    # confirming the two channels are genuinely separate.
    assert store.search(query, using=IMAGE_VECTOR)[0].score < 0.99


def test_filter_is_a_hard_constraint(
    populated: tuple[InMemoryVectorStore, FakeEmbedder],
) -> None:
    """An excluded body shape must never appear, however similar the vector is."""
    store, embedder = populated
    hits = store.search(
        embedder.embed_image(b"o3"),
        using=IMAGE_VECTOR,
        where=Filter(must_equal={"body_shape": "pear"}),
    )
    assert {h.id for h in hits} == {"o1", "o2"}
    assert "o3" not in {h.id for h in hits}


def test_must_be_in_filter(populated: tuple[InMemoryVectorStore, FakeEmbedder]) -> None:
    store, embedder = populated
    hits = store.search(
        embedder.embed_image(b"o1"),
        where=Filter(must_be_in={"body_shape": ("apple",)}),
    )
    assert {h.id for h in hits} == {"o3"}


def test_limit_is_respected(populated: tuple[InMemoryVectorStore, FakeEmbedder]) -> None:
    store, embedder = populated
    assert len(store.search(embedder.embed_image(b"o1"), limit=2)) == 2


def test_upsert_rejects_wrong_dimension() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection(64)
    with pytest.raises(ValueError, match="img_vec has dim 3"):
        store.upsert("x", OutfitVectors([0.0] * 3, [0.0] * 64), {})


def test_upsert_replaces_rather_than_duplicates(
    populated: tuple[InMemoryVectorStore, FakeEmbedder],
) -> None:
    store, embedder = populated
    before = store.count()
    store.upsert(
        "o1",
        OutfitVectors(embedder.embed_image(b"new"), embedder.embed_text("new")),
        {"body_shape": "rectangle"},
    )
    assert store.count() == before
    payload = store.get("o1")
    assert payload is not None
    assert payload["body_shape"] == "rectangle"


def test_dimension_mismatch_at_query_time_is_explicit(
    populated: tuple[InMemoryVectorStore, FakeEmbedder],
) -> None:
    store, _ = populated
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.search([0.1, 0.2])
