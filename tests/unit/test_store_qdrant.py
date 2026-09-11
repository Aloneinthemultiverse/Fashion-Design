"""Qdrant adapter.

Runs against `QdrantClient(":memory:")` -- qdrant-client's local mode, which is the real
client and the real query semantics without a server. That matters because the things
most likely to break here are Qdrant-specific: named-vector selection, filter
translation, and id handling. A hand-written mock would validate none of them.
"""

from __future__ import annotations

import pytest

from fashion.adapters.embed_fake import FakeEmbedder
from fashion.adapters.store_qdrant import QdrantVectorStore, point_id
from fashion.ports.vectorstore import (
    CAPTION_VECTOR,
    IMAGE_VECTOR,
    Filter,
    OutfitVectors,
    VectorStore,
)

pytest.importorskip("qdrant_client")


@pytest.fixture
def store() -> QdrantVectorStore:
    from qdrant_client import QdrantClient

    embedder = FakeEmbedder()
    store = QdrantVectorStore(client=QdrantClient(":memory:"), collection="test_outfits")
    store.ensure_collection(embedder.dim)
    for item_id, shape, culture, caption in [
        ("Q1-0", "pear", "ethnic", "a flowing a-line anarkali in crimson"),
        ("Q2-0", "pear", "western", "a structured bodycon gown in black"),
        ("Q3-0", "apple", "ethnic", "an empire-line kurta in ivory"),
    ]:
        store.upsert(
            item_id,
            OutfitVectors(
                img_vec=embedder.embed_image(item_id.encode()),
                cap_vec=embedder.embed_text(caption),
            ),
            {"body_shape": shape, "culture": culture, "caption": caption},
        )
    return store


def test_adapter_satisfies_the_port(store: QdrantVectorStore) -> None:
    assert isinstance(store, VectorStore)


def test_all_points_are_stored(store: QdrantVectorStore) -> None:
    assert store.count() == 3


def test_image_to_image_search_finds_the_right_point(store: QdrantVectorStore) -> None:
    hits = store.search(FakeEmbedder().embed_image(b"Q2-0"), using=IMAGE_VECTOR, limit=3)
    assert hits[0].id == "Q2-0"
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)


def test_caption_search_uses_the_other_named_vector(store: QdrantVectorStore) -> None:
    """Proves the two named vectors are genuinely separate inside one collection."""
    query = FakeEmbedder().embed_text("an empire-line kurta in ivory")
    assert store.search(query, using=CAPTION_VECTOR, limit=3)[0].id == "Q3-0"
    assert store.search(query, using=IMAGE_VECTOR, limit=3)[0].score < 0.99


def test_filter_excludes_non_matching_shapes(store: QdrantVectorStore) -> None:
    """The body-shape constraint must be enforced by the engine, not post-hoc."""
    hits = store.search(
        FakeEmbedder().embed_image(b"Q3-0"),
        using=IMAGE_VECTOR,
        limit=10,
        where=Filter(must_equal={"body_shape": "pear"}),
    )
    assert {h.id for h in hits} == {"Q1-0", "Q2-0"}


def test_multiple_filter_conditions_are_combined(store: QdrantVectorStore) -> None:
    hits = store.search(
        FakeEmbedder().embed_image(b"Q1-0"),
        limit=10,
        where=Filter(must_equal={"body_shape": "pear", "culture": "ethnic"}),
    )
    assert {h.id for h in hits} == {"Q1-0"}


def test_must_be_in_filter(store: QdrantVectorStore) -> None:
    hits = store.search(
        FakeEmbedder().embed_image(b"Q1-0"),
        limit=10,
        where=Filter(must_be_in={"culture": ("western",)}),
    )
    assert {h.id for h in hits} == {"Q2-0"}


def test_string_outfit_ids_survive_the_uuid_mapping(store: QdrantVectorStore) -> None:
    """Qdrant rejects arbitrary string ids, so the original must live in the payload."""
    payload = store.get("Q1-0")
    assert payload is not None
    assert payload["outfit_id"] == "Q1-0"


def test_upsert_is_idempotent(store: QdrantVectorStore) -> None:
    """Re-running the index builder must update rows, not duplicate them."""
    embedder = FakeEmbedder()
    before = store.count()
    store.upsert(
        "Q1-0",
        OutfitVectors(embedder.embed_image(b"new"), embedder.embed_text("new caption")),
        {"body_shape": "rectangle", "culture": "fusion"},
    )
    assert store.count() == before
    payload = store.get("Q1-0")
    assert payload is not None
    assert payload["body_shape"] == "rectangle"


def test_point_ids_are_deterministic_across_processes() -> None:
    assert point_id("Q1-0") == point_id("Q1-0")
    assert point_id("Q1-0") != point_id("Q2-0")


def test_get_returns_none_for_an_unknown_id(store: QdrantVectorStore) -> None:
    assert store.get("does-not-exist") is None


def test_limit_is_respected(store: QdrantVectorStore) -> None:
    assert len(store.search(FakeEmbedder().embed_image(b"Q1-0"), limit=2)) == 2
