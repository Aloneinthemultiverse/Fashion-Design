"""Hybrid retrieval.

The load-bearing guarantee is that body shape is a constraint, not a preference. Most of
these tests exist to stop that from silently degrading into a score term.
"""

from __future__ import annotations

import pytest

from fashion.adapters.embed_fake import FakeEmbedder
from fashion.adapters.store_memory import InMemoryVectorStore
from fashion.core.models import (
    BodyMetrics,
    BodyShape,
    Build,
    CelebrityProfile,
    Culture,
    HeightBand,
    Neckline,
    Occasion,
    Silhouette,
    UserQuery,
    WaistEmphasis,
)
from fashion.pipeline.index import IndexBuilder
from fashion.pipeline.retrieve import Retriever
from tests.conftest import make_outfit


def profile(cid: str, shape: BodyShape, name: str = "Someone") -> CelebrityProfile:
    return CelebrityProfile(
        id=cid,
        name=name,
        region="indian",
        shape=shape,
        build=Build.SLIM,
        height_band=HeightBand.AVERAGE,
    )


@pytest.fixture
def indexed(tmp_path):  # type: ignore[no-untyped-def]
    """A small corpus spanning two body shapes and two cultures."""
    embedder = FakeEmbedder()
    store = InMemoryVectorStore()

    specs = [
        ("pear1", "cPear", BodyShape.PEAR, Culture.ETHNIC, Silhouette.A_LINE, "anarkali"),
        ("pear2", "cPear", BodyShape.PEAR, Culture.WESTERN, Silhouette.BODYCON, "gown"),
        ("apple1", "cApple", BodyShape.APPLE, Culture.ETHNIC, Silhouette.EMPIRE, "kurta"),
        ("apple2", "cApple", BodyShape.APPLE, Culture.WESTERN, Silhouette.STRAIGHT, "suit"),
    ]
    items = []
    profiles = {}
    for oid, cid, shape, culture, silhouette, garment in specs:
        image = tmp_path / f"{oid}.jpg"
        image.write_bytes(f"image-bytes-{oid}".encode())
        items.append(
            make_outfit(
                oid,
                celebrity_id=cid,
                garment_type=garment,
                silhouette=silhouette,
                culture=culture,
                caption=f"a {garment} with {silhouette.value} cut",
            ).model_copy(update={"image_path": str(image)})
        )
        profiles[cid] = profile(cid, shape, name=f"Celeb {shape.value}")

    stats = IndexBuilder(embedder, store).build(items, profiles)
    assert stats.indexed == 4
    return embedder, store


def metrics(shape: BodyShape) -> BodyMetrics:
    return BodyMetrics(
        shape=shape, build=Build.SLIM, height_band=HeightBand.AVERAGE, user_confirmed=True
    )


def test_only_matching_body_shapes_are_returned(indexed) -> None:  # type: ignore[no-untyped-def]
    """The central guarantee: shape filters, it does not merely influence."""
    embedder, store = indexed
    result = Retriever(embedder, store).retrieve(
        metrics(BodyShape.PEAR), UserQuery(text="a flowing ethnic outfit")
    )
    assert result.recommendations
    assert not result.relaxed
    assert all(r.outfit.id.startswith("pear") for r in result.recommendations)


def test_culture_filter_is_honoured(indexed) -> None:  # type: ignore[no-untyped-def]
    embedder, store = indexed
    result = Retriever(embedder, store).retrieve(
        metrics(BodyShape.PEAR), UserQuery(text="something", culture=Culture.ETHNIC)
    )
    assert {r.outfit.id for r in result.recommendations} == {"pear1"}


def test_text_query_uses_both_text_channels(indexed) -> None:  # type: ignore[no-untyped-def]
    """caption (text-to-text) and cross (text-to-image) are genuinely separate paths."""
    embedder, store = indexed
    result = Retriever(embedder, store).retrieve(
        metrics(BodyShape.PEAR), UserQuery(text="an a-line anarkali")
    )
    assert set(result.channels_used) == {"caption", "cross"}


def test_photo_adds_the_image_to_image_channel(indexed) -> None:  # type: ignore[no-untyped-def]
    embedder, store = indexed
    result = Retriever(embedder, store).retrieve(
        metrics(BodyShape.PEAR), UserQuery(text="ethnic"), photo=b"image-bytes-pear1"
    )
    assert "image" in result.channels_used


def test_image_only_query_works_without_any_text(indexed) -> None:  # type: ignore[no-untyped-def]
    """Pure image-to-image retrieval must not depend on the user typing anything."""
    embedder, store = indexed
    result = Retriever(embedder, store).retrieve(
        metrics(BodyShape.PEAR), UserQuery(text=""), photo=b"image-bytes-pear1"
    )
    assert result.channels_used == ("image",)
    assert result.recommendations[0].outfit.id == "pear1"


def test_filter_is_relaxed_only_when_nothing_matches(indexed) -> None:  # type: ignore[no-untyped-def]
    """Returning cross-shape options clearly flagged beats returning nothing."""
    embedder, store = indexed
    result = Retriever(embedder, store).retrieve(
        metrics(BodyShape.HOURGLASS), UserQuery(text="anything")
    )
    assert result.relaxed
    assert result.recommendations


def test_better_fitting_cuts_outrank_worse_ones(indexed) -> None:  # type: ignore[no-untyped-def]
    """A-line suits a pear frame; bodycon does not. Fit must affect the order."""
    embedder, store = indexed
    result = Retriever(embedder, store).retrieve(
        metrics(BodyShape.PEAR), UserQuery(text="an outfit")
    )
    ids = [r.outfit.id for r in result.recommendations]
    assert ids.index("pear1") < ids.index("pear2")


def test_results_carry_a_rationale_and_channel_provenance(indexed) -> None:  # type: ignore[no-untyped-def]
    embedder, store = indexed
    result = Retriever(embedder, store).retrieve(
        metrics(BodyShape.PEAR), UserQuery(text="ethnic wear")
    )
    top = result.recommendations[0]
    assert top.rationale
    assert top.matched_on
    assert top.outfit.garment_type in top.rationale


def test_poor_fits_come_with_concrete_adjustments(indexed) -> None:  # type: ignore[no-untyped-def]
    embedder, store = indexed
    result = Retriever(embedder, store).retrieve(
        metrics(BodyShape.PEAR), UserQuery(text="a gown", culture=Culture.WESTERN)
    )
    bodycon = next(r for r in result.recommendations if r.outfit.id == "pear2")
    assert bodycon.adjustments


def test_top_k_is_respected(indexed) -> None:  # type: ignore[no-untyped-def]
    embedder, store = indexed
    result = Retriever(embedder, store).retrieve(
        metrics(BodyShape.PEAR), UserQuery(text="outfit", top_k=1)
    )
    assert len(result.recommendations) == 1


def test_celebrity_mode_restricts_to_that_person(indexed) -> None:  # type: ignore[no-untyped-def]
    embedder, store = indexed
    result = Retriever(embedder, store).retrieve(
        metrics(BodyShape.PEAR),
        UserQuery(text="outfit", celebrity_name="Celeb pear"),
    )
    assert all(r.outfit.celebrity_id == "cPear" for r in result.recommendations)


def test_index_skips_outfits_without_a_profile(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """No profile means no body shape, so the item could never be matched to a user."""
    image = tmp_path / "x.jpg"
    image.write_bytes(b"bytes")
    item = make_outfit("orphan", celebrity_id="missing").model_copy(
        update={"image_path": str(image)}
    )
    stats = IndexBuilder(FakeEmbedder(), InMemoryVectorStore()).build([item], {})
    assert stats.indexed == 0
    assert stats.skipped_no_profile == 1


def test_index_skips_outfits_whose_image_is_gone(tmp_path) -> None:  # type: ignore[no-untyped-def]
    item = make_outfit("ghost", celebrity_id="c1").model_copy(
        update={"image_path": str(tmp_path / "absent.jpg")}
    )
    stats = IndexBuilder(FakeEmbedder(), InMemoryVectorStore()).build(
        [item], {"c1": profile("c1", BodyShape.PEAR)}
    )
    assert stats.skipped_no_image == 1


def test_payload_carries_provenance(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Anything served must be attributable, so licence travels into the index."""
    from fashion.pipeline.index import build_payload

    payload = build_payload(make_outfit("o1"), profile("c1", BodyShape.PEAR))
    assert payload["license"] == "CC0"
    assert payload["body_shape"] == "pear"


def test_occasion_filter_is_honoured(indexed) -> None:  # type: ignore[no-untyped-def]
    embedder, store = indexed
    result = Retriever(embedder, store).retrieve(
        metrics(BodyShape.PEAR), UserQuery(text="x", occasion=Occasion.WEDDING)
    )
    # Fixtures are all FESTIVE, so a WEDDING filter matches nothing and relaxing the
    # body filter cannot help either -- occasion stays applied.
    assert all(r.outfit.occasion == Occasion.WEDDING for r in result.recommendations)


def test_unreadable_payload_is_skipped_not_fatal() -> None:
    from fashion.pipeline.retrieve import _outfit_from_payload

    assert _outfit_from_payload("x", {"silhouette": "nonsense"}) is None


def test_neckline_and_waist_survive_the_index_round_trip(indexed) -> None:  # type: ignore[no-untyped-def]
    """Rationale quality depends on these, so they must not be lost in the payload."""
    embedder, store = indexed
    result = Retriever(embedder, store).retrieve(metrics(BodyShape.PEAR), UserQuery(text="outfit"))
    top = result.recommendations[0].outfit
    assert top.neckline is Neckline.V_NECK
    assert top.waist_emphasis is WaistEmphasis.HIGH


def _mixed_region_corpus(tmp_path):  # type: ignore[no-untyped-def]
    """Same body shape, same garments, different celebrity regions.

    Body shape is held constant so that anything the region filter changes is
    attributable to the region filter alone.
    """
    embedder = FakeEmbedder()
    store = InMemoryVectorStore()
    items, profiles = [], {}
    for region in ("indian", "american", "british"):
        image = tmp_path / f"{region}.jpg"
        image.write_bytes(f"outfit-{region}".encode())
        items.append(
            make_outfit(f"o-{region}", celebrity_id=f"c-{region}").model_copy(
                update={"image_path": str(image)}
            )
        )
        profiles[f"c-{region}"] = CelebrityProfile(
            id=f"c-{region}",
            name=f"Celeb {region}",
            region=region,
            shape=BodyShape.PEAR,
            build=Build.SLIM,
            height_band=HeightBand.AVERAGE,
        )
    IndexBuilder(embedder, store).build(items, profiles)
    return embedder, store


def test_region_restricts_the_wardrobe(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The product premise: a Western body matched to an *Indian* wardrobe.

    Without this filter the composition of the corpus decides which wardrobe a user
    sees, which is how the premise breaks silently rather than loudly.
    """
    embedder, store = _mixed_region_corpus(tmp_path)
    result = Retriever(embedder, store).retrieve(
        metrics(BodyShape.PEAR), UserQuery(text="an outfit", region="indian")
    )
    assert result.recommendations
    assert all(r.outfit.celebrity_id == "c-indian" for r in result.recommendations)


def test_omitting_region_searches_every_wardrobe(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The engine stays general; the product default lives at the edge."""
    embedder, store = _mixed_region_corpus(tmp_path)
    result = Retriever(embedder, store).retrieve(
        metrics(BodyShape.PEAR), UserQuery(text="an outfit")
    )
    assert {r.outfit.celebrity_id for r in result.recommendations} == {
        "c-indian",
        "c-american",
        "c-british",
    }


def test_region_survives_the_body_shape_relaxation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Relaxing the shape filter must not quietly widen the wardrobe as well.

    Showing a cross-shape Indian outfit is a labelled compromise; showing an American
    one to a user who asked for Indian is a different product.
    """
    embedder, store = _mixed_region_corpus(tmp_path)
    result = Retriever(embedder, store).retrieve(
        metrics(BodyShape.HOURGLASS),  # no hourglass wearers exist in this corpus
        UserQuery(text="an outfit", region="indian"),
    )
    assert result.relaxed
    assert result.recommendations
    assert all(r.outfit.celebrity_id == "c-indian" for r in result.recommendations)


def _wardrobe_corpus(tmp_path):  # type: ignore[no-untyped-def]
    """Same shape and region throughout, so only wardrobe can explain a difference."""
    from fashion.core.models import Wardrobe

    embedder = FakeEmbedder()
    store = InMemoryVectorStore()
    items, profiles = [], {}
    for wardrobe in Wardrobe:
        image = tmp_path / f"{wardrobe.value}.jpg"
        image.write_bytes(f"outfit-{wardrobe.value}".encode())
        items.append(
            make_outfit(f"o-{wardrobe.value}", celebrity_id=f"c-{wardrobe.value}").model_copy(
                update={"image_path": str(image), "wardrobe": wardrobe}
            )
        )
        profiles[f"c-{wardrobe.value}"] = CelebrityProfile(
            id=f"c-{wardrobe.value}",
            name=f"Celeb {wardrobe.value}",
            region="indian",
            shape=BodyShape.PEAR,
            build=Build.SLIM,
            height_band=HeightBand.AVERAGE,
        )
    IndexBuilder(embedder, store).build(items, profiles)
    return embedder, store


def test_womenswear_is_never_returned_for_a_menswear_request(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A saree returned to someone wanting menswear is a garment they cannot wear.

    This is the failure that produced a generated image of a woman for a male user.
    """
    from fashion.core.models import Wardrobe

    embedder, store = _wardrobe_corpus(tmp_path)
    result = Retriever(embedder, store).retrieve(
        metrics(BodyShape.PEAR), UserQuery(text="an outfit", wardrobe=Wardrobe.MENSWEAR)
    )
    returned = {r.outfit.wardrobe for r in result.recommendations}
    assert Wardrobe.WOMENSWEAR not in returned
    assert returned <= {Wardrobe.MENSWEAR, Wardrobe.UNISEX}


def test_unisex_stays_eligible_for_either_wardrobe(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """That is the whole point of the category; excluding it would gut the corpus."""
    from fashion.core.models import Wardrobe

    embedder, store = _wardrobe_corpus(tmp_path)
    for wanted in (Wardrobe.MENSWEAR, Wardrobe.WOMENSWEAR):
        result = Retriever(embedder, store).retrieve(
            metrics(BodyShape.PEAR), UserQuery(text="an outfit", wardrobe=wanted)
        )
        assert Wardrobe.UNISEX in {r.outfit.wardrobe for r in result.recommendations}


def test_wardrobe_survives_the_body_shape_relaxation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Relaxing shape must not quietly hand back clothes from the other tradition."""
    from fashion.core.models import Wardrobe

    embedder, store = _wardrobe_corpus(tmp_path)
    result = Retriever(embedder, store).retrieve(
        metrics(BodyShape.HOURGLASS),  # no hourglass wearers exist here
        UserQuery(text="an outfit", wardrobe=Wardrobe.MENSWEAR),
    )
    assert result.relaxed
    assert Wardrobe.WOMENSWEAR not in {r.outfit.wardrobe for r in result.recommendations}


def test_the_photo_suggests_a_wardrobe_when_the_user_states_none(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The reading is a default, not a verdict -- but it must actually be applied."""
    from fashion.core.models import BodyMetrics, Wardrobe

    embedder, store = _wardrobe_corpus(tmp_path)
    inferred = BodyMetrics(
        shape=BodyShape.PEAR,
        build=Build.SLIM,
        height_band=HeightBand.AVERAGE,
        wardrobe=Wardrobe.MENSWEAR,
        user_confirmed=True,
    )
    result = Retriever(embedder, store).retrieve(inferred, UserQuery(text="an outfit"))
    assert Wardrobe.WOMENSWEAR not in {r.outfit.wardrobe for r in result.recommendations}


def test_an_explicit_choice_overrides_the_photo(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The user's stated preference outranks anything inferred from their picture."""
    from fashion.core.models import BodyMetrics, Wardrobe

    embedder, store = _wardrobe_corpus(tmp_path)
    inferred = BodyMetrics(
        shape=BodyShape.PEAR,
        build=Build.SLIM,
        height_band=HeightBand.AVERAGE,
        wardrobe=Wardrobe.MENSWEAR,
        user_confirmed=True,
    )
    result = Retriever(embedder, store).retrieve(
        inferred, UserQuery(text="an outfit", wardrobe=Wardrobe.WOMENSWEAR)
    )
    assert Wardrobe.MENSWEAR not in {r.outfit.wardrobe for r in result.recommendations}
