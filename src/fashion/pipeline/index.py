"""Index building.

Computes the two named vectors per outfit and writes them to the vector store, with the
structured payload that retrieval pre-filters on.

The payload is denormalised deliberately: it carries the celebrity's body shape, build
and height band alongside the garment attributes. Body match is applied as a hard filter
during search, and making the store join back to a profile table per query would turn a
single filtered search into N lookups.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from fashion.core.models import CelebrityProfile, OutfitItem
from fashion.ports.embedder import Embedder
from fashion.ports.vectorstore import OutfitVectors, VectorStore

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexStats:
    indexed: int = 0
    skipped_no_profile: int = 0
    skipped_no_image: int = 0
    failed: int = 0


def build_payload(item: OutfitItem, profile: CelebrityProfile) -> dict[str, object]:
    """Flatten an outfit plus its wearer into one filterable payload."""
    return {
        "outfit_id": item.id,
        "celebrity_id": item.celebrity_id,
        "celebrity_name": profile.name,
        "region": profile.region,
        # Wearer geometry -- the hard filter for recommendations.
        "body_shape": profile.shape.value,
        "build": profile.build.value,
        "height_band": profile.height_band.value,
        # Garment attributes.
        "garment_type": item.garment_type,
        "silhouette": item.silhouette.value,
        "neckline": item.neckline.value,
        "waist_emphasis": item.waist_emphasis.value,
        "culture": item.culture.value,
        "occasion": item.occasion.value,
        "colors": list(item.colors),
        "patterns": list(item.patterns),
        "style_tags": list(item.style_tags),
        "caption": item.caption,
        "image_path": item.image_path,
        # Provenance travels with the row so anything served can be attributed.
        "source": item.source,
        "license": item.license,
    }


class IndexBuilder:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def build(
        self,
        items: Iterable[OutfitItem],
        profiles: dict[str, CelebrityProfile],
    ) -> IndexStats:
        self._store.ensure_collection(self._embedder.dim)

        indexed = skipped_profile = skipped_image = failed = 0

        for item in items:
            profile = profiles.get(item.celebrity_id)
            if profile is None:
                # Without a profile there is no body shape, so the item could never be
                # matched to a user and would only add noise to similarity search.
                skipped_profile += 1
                continue

            path = Path(item.image_path)
            if not path.exists():
                skipped_image += 1
                continue

            try:
                image_bytes = path.read_bytes()
                img_vec = self._embedder.embed_image(image_bytes)
                # Fall back to the garment type when a caption is missing, so the item
                # still has a usable text vector rather than an embedding of "".
                caption = item.caption.strip() or f"a {item.garment_type}"
                cap_vec = self._embedder.embed_text(caption)
            except Exception:
                log.warning("failed to embed %s", item.id, exc_info=True)
                failed += 1
                continue

            self._store.upsert(
                item.id,
                OutfitVectors(img_vec=img_vec, cap_vec=cap_vec),
                build_payload(item, profile),
            )
            indexed += 1

        return IndexStats(
            indexed=indexed,
            skipped_no_profile=skipped_profile,
            skipped_no_image=skipped_image,
            failed=failed,
        )
