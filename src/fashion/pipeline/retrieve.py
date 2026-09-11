"""Hybrid retrieval.

Three channels, fused by rank:

* `image` -- the user's own photo, or a reference outfit photo, against `img_vec`.
  This is the image-to-image path.
* `caption` -- the user's text preferences against `cap_vec` (text-to-text).
* `cross` -- the same text against `img_vec` (cross-modal text-to-image), which CLIP's
  shared space makes possible and which surfaces different items than `caption` does,
  because one matches how an outfit *looks* and the other how it was *described*.

Running all three and fusing them is worth the extra searches precisely because they
disagree: an item ranked well by several channels is a safer recommendation than one a
single channel loved.

Body shape is applied as a **pre-filter**, never as a score term. An outfit cut for a
pear frame does not become right for an apple frame because its colours match, so it
must not be able to rank its way in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fashion.core.crosscultural import adjustments, explain, fit_score
from fashion.core.models import (
    BodyMetrics,
    Culture,
    Occasion,
    OutfitItem,
    Recommendation,
    UserQuery,
    as_str_tuple,
)
from fashion.core.ranking import reciprocal_rank_fusion
from fashion.ports.embedder import Embedder
from fashion.ports.vectorstore import (
    CAPTION_VECTOR,
    IMAGE_VECTOR,
    Filter,
    VectorStore,
)

log = logging.getLogger(__name__)

# The image channel is weighted below the text channels because the user's query photo
# shows what they currently wear, which is weaker evidence of what they *want* than the
# preferences they typed.
CHANNEL_WEIGHTS = {"caption": 1.0, "cross": 0.9, "image": 0.6}

# Over-fetch per channel so fusion has enough overlap to be meaningful, and so the
# post-filter on fit score has candidates to discard.
CHANNEL_DEPTH = 40


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    recommendations: tuple[Recommendation, ...]
    considered: int
    channels_used: tuple[str, ...]
    relaxed: bool = False


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def _filter_for(self, metrics: BodyMetrics, query: UserQuery, *, strict: bool) -> Filter:
        must: dict[str, str] = {}
        if strict:
            must["body_shape"] = metrics.shape.value
        if query.region:
            # Region is a constraint, not a preference. Without it the corpus order
            # decides which wardrobe a user is shown, which silently breaks the
            # cross-cultural premise as soon as non-Indian celebrities are ingested.
            must["region"] = query.region
        if query.culture is not None:
            must["culture"] = query.culture.value
        if query.occasion is not None:
            must["occasion"] = query.occasion.value
        if query.celebrity_name:
            must["celebrity_name"] = query.celebrity_name
        return Filter(must_equal=must)

    def _search_channels(
        self, metrics: BodyMetrics, query: UserQuery, photo: bytes | None, where: Filter
    ) -> tuple[dict[str, list[str]], dict[str, dict[str, object]]]:
        channels: dict[str, list[str]] = {}
        payloads: dict[str, dict[str, object]] = {}

        def record(name: str, vector: list[float], using: str) -> None:
            hits = self._store.search(vector, using=using, limit=CHANNEL_DEPTH, where=where)
            channels[name] = [h.id for h in hits]
            for hit in hits:
                payloads.setdefault(hit.id, hit.payload)

        text = query.text.strip()
        if text:
            text_vec = self._embedder.embed_text(text)
            record("caption", text_vec, CAPTION_VECTOR)
            record("cross", text_vec, IMAGE_VECTOR)

        if photo is not None:
            record("image", self._embedder.embed_image(photo), IMAGE_VECTOR)

        if not channels:
            # No text and no photo: fall back to the wearer's own caption space so the
            # filter alone still returns something sensible rather than nothing.
            record("caption", self._embedder.embed_text(metrics.shape.value), CAPTION_VECTOR)

        return channels, payloads

    def retrieve(
        self,
        metrics: BodyMetrics,
        query: UserQuery,
        *,
        photo: bytes | None = None,
    ) -> RetrievalResult:
        where = self._filter_for(metrics, query, strict=True)
        channels, payloads = self._search_channels(metrics, query, photo, where)
        relaxed = False

        if not any(channels.values()):
            # A sparse corpus can have no outfits at all for a given shape. Returning
            # nothing is a worse answer than returning cross-shape options that are
            # clearly labelled as needing adjustment, so the shape filter is dropped
            # and `relaxed` tells the UI to say so.
            log.info("no items for shape %s; relaxing the body filter", metrics.shape.value)
            where = self._filter_for(metrics, query, strict=False)
            channels, payloads = self._search_channels(metrics, query, photo, where)
            relaxed = True

        fused = reciprocal_rank_fusion(channels, weights=CHANNEL_WEIGHTS)

        recommendations: list[Recommendation] = []
        for item_id, score, matched in fused:
            payload = payloads.get(item_id)
            if payload is None:
                continue
            item = _outfit_from_payload(item_id, payload)
            if item is None:
                continue
            recommendations.append(
                Recommendation(
                    outfit=item,
                    # Blend retrieval rank with how well the cut actually suits the
                    # shape, so a visually similar but badly-cut garment ranks below a
                    # slightly less similar one that flatters.
                    score=round(score * (0.5 + 0.5 * fit_score(item, metrics.shape)), 6),
                    rationale=explain(item, metrics.shape),
                    adjustments=adjustments(item, metrics.shape),
                    matched_on=matched,
                )
            )
            if len(recommendations) >= query.top_k:
                break

        recommendations.sort(key=lambda r: (-r.score, r.outfit.id))
        return RetrievalResult(
            recommendations=tuple(recommendations),
            considered=len(fused),
            channels_used=tuple(channels),
            relaxed=relaxed,
        )


def _outfit_from_payload(item_id: str, payload: dict[str, object]) -> OutfitItem | None:
    """Rebuild an OutfitItem from a stored payload.

    Returns None rather than raising: a payload written by an older index version should
    be skipped, not crash a user's request.
    """
    from fashion.core.models import Neckline, Silhouette, WaistEmphasis

    try:
        return OutfitItem(
            id=item_id,
            celebrity_id=str(payload.get("celebrity_id", "unknown")),
            image_path=str(payload.get("image_path", "")) or "unknown",
            garment_type=str(payload.get("garment_type", "")) or "unknown",
            silhouette=Silhouette(str(payload.get("silhouette"))),
            neckline=Neckline(str(payload.get("neckline"))),
            waist_emphasis=WaistEmphasis(str(payload.get("waist_emphasis"))),
            culture=Culture(str(payload.get("culture"))),
            occasion=Occasion(str(payload.get("occasion"))),
            colors=as_str_tuple(payload.get("colors")),
            patterns=as_str_tuple(payload.get("patterns")),
            style_tags=as_str_tuple(payload.get("style_tags")),
            caption=str(payload.get("caption", "")),
            source=str(payload.get("source", "")) or "unknown",
            license=str(payload.get("license", "")) or "unknown",
        )
    except (ValueError, TypeError):
        log.warning("skipping unreadable payload for %s", item_id)
        return None
