"""Ingest: roster + images -> labelled corpus.

Turns Wikidata roster rows into `OutfitItem` and `CelebrityProfile` records by fetching
each person's Commons image and running one VLM pass over it.

Two properties matter more than throughput here:

* **Resumability.** A full pass over 2,000 images spans more than one day of free VLM
  quota, so it will be interrupted. Already-labelled ids are skipped and each result is
  appended immediately, so a re-run continues rather than restarts.
* **Honest provenance.** Every item carries the source URL and licence returned by
  Commons. Nothing is written without them.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from fashion.core.body import Proportions, classify
from fashion.core.dataset import CelebrityRepository, OutfitRepository
from fashion.core.models import (
    CelebrityProfile,
    Culture,
    Neckline,
    Occasion,
    OutfitItem,
    Silhouette,
    WaistEmphasis,
    as_str_tuple,
)
from fashion.ports.vision import VisionModel

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RosterRow:
    """A roster entry as written by scripts/fetch_roster.py."""

    id: str
    name: str
    region: str
    image_url: str

    @classmethod
    def from_dict(cls, row: dict[str, object]) -> RosterRow | None:
        qid = str(row.get("qid", ""))
        ident = qid.rsplit("/", 1)[-1] if qid else ""
        name = str(row.get("name", ""))
        image_url = str(row.get("image_url", ""))
        if not (ident and name and image_url):
            return None
        return cls(id=ident, name=name, region=str(row.get("region", "")), image_url=image_url)


@dataclass(frozen=True, slots=True)
class IngestStats:
    seen: int = 0
    skipped_existing: int = 0
    skipped_no_image: int = 0
    skipped_unclear: int = 0
    failed: int = 0
    written: int = 0

    def merge(self, **deltas: int) -> IngestStats:
        current = {
            "seen": self.seen,
            "skipped_existing": self.skipped_existing,
            "skipped_no_image": self.skipped_no_image,
            "skipped_unclear": self.skipped_unclear,
            "failed": self.failed,
            "written": self.written,
        }
        for key, value in deltas.items():
            current[key] += value
        return IngestStats(**current)


def _enum_or(value: object, enum_cls: type, default: object) -> object:
    """Coerce a VLM string to an enum member, falling back rather than failing.

    The response schema constrains these fields, but schema adherence is not guaranteed
    and one bad value should not discard an otherwise-usable record.
    """
    try:
        return enum_cls(str(value))
    except ValueError:
        return default


class Ingestor:
    """Labels roster rows into the corpus."""

    def __init__(
        self,
        vision: VisionModel,
        image_source: object,
        outfits: OutfitRepository,
        celebrities: CelebrityRepository,
        *,
        require_clear_outfit: bool = True,
    ) -> None:
        self._vision = vision
        self._source = image_source
        self._outfits = outfits
        self._celebrities = celebrities
        self._require_clear_outfit = require_clear_outfit

    def run(self, rows: Iterable[RosterRow], *, limit: int | None = None) -> IngestStats:
        stats = IngestStats()
        already = self._outfits.labelled_ids()
        profiles: dict[str, CelebrityProfile] = self._celebrities.index()

        for row in self._take(rows, limit):
            stats = stats.merge(seen=1)
            outfit_id = f"{row.id}-0"
            if outfit_id in already:
                stats = stats.merge(skipped_existing=1)
                continue

            sourced = self._source.fetch_one(row.image_url, celebrity=row.name)  # type: ignore[attr-defined]
            if sourced is None:
                stats = stats.merge(skipped_no_image=1)
                continue

            try:
                image = Path(sourced.local_path).read_bytes()
            except OSError:
                stats = stats.merge(failed=1)
                continue

            try:
                tags = self._vision.tag_outfit(image)
                metrics = self._vision.analyze_body(image)
                caption = self._vision.caption_outfit(image)
            except Exception:
                log.warning("VLM failed for %s", row.name, exc_info=True)
                stats = stats.merge(failed=1)
                continue

            if self._require_clear_outfit and tags.get("outfit_clearly_visible") is False:
                # A face close-up yields confident-looking but meaningless garment tags,
                # which is worse for retrieval than having no record at all.
                stats = stats.merge(skipped_unclear=1)
                continue

            try:
                item = OutfitItem(
                    id=outfit_id,
                    celebrity_id=row.id,
                    image_path=sourced.local_path,
                    garment_type=str(tags.get("garment_type") or "unknown"),
                    silhouette=_enum_or(  # type: ignore[arg-type]
                        tags.get("silhouette"), Silhouette, Silhouette.STRAIGHT
                    ),
                    neckline=_enum_or(  # type: ignore[arg-type]
                        tags.get("neckline"), Neckline, Neckline.ROUND
                    ),
                    waist_emphasis=_enum_or(  # type: ignore[arg-type]
                        tags.get("waist_emphasis"), WaistEmphasis, WaistEmphasis.NATURAL
                    ),
                    culture=_enum_or(  # type: ignore[arg-type]
                        tags.get("culture"), Culture, Culture.WESTERN
                    ),
                    occasion=_enum_or(  # type: ignore[arg-type]
                        tags.get("occasion"), Occasion, Occasion.CASUAL
                    ),
                    colors=as_str_tuple(tags.get("colors")),
                    patterns=as_str_tuple(tags.get("patterns")),
                    fabric=str(tags.get("fabric")) if tags.get("fabric") else None,
                    style_tags=as_str_tuple(tags.get("style_tags")),
                    caption=caption,
                    source=sourced.source,
                    license=sourced.license,
                )
            except ValidationError:
                log.warning("invalid outfit record for %s", row.name, exc_info=True)
                stats = stats.merge(failed=1)
                continue

            self._outfits.append(item)
            profile = CelebrityProfile(
                id=row.id,
                name=row.name,
                region=row.region,
                shape=metrics.shape,
                build=metrics.build,
                height_band=metrics.height_band,
                style_tags=item.style_tags,
                full_body=bool(tags.get("full_body_visible", False)),
                shape_confidence=metrics.confidence,
            )
            profiles[row.id] = profile
            # Flush profiles as they are produced, not only at the end. A full pass
            # takes hours; losing every body shape to a crash in the final hour would
            # be unrecoverable, and rewriting the file per item is cheap next to the
            # model call that produced it.
            self._celebrities.save(profiles.values())
            stats = stats.merge(written=1)

        self._celebrities.save(profiles.values())
        return stats

    @staticmethod
    def _take(rows: Iterable[RosterRow], limit: int | None) -> Iterator[RosterRow]:
        for index, row in enumerate(rows):
            if limit is not None and index >= limit:
                return
            yield row


def body_shape_from_widths(shoulder: float, waist: float, hip: float) -> object:
    """Convenience wrapper used by the labelling UI when a human overrides measurements."""
    return classify(Proportions(shoulder, waist, hip))
