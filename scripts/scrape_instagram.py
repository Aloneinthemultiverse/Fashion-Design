"""Build corpus from Instagram: the dynamic, current-season layer.

This works straight from the Wikidata roster rather than from already-labelled
celebrities, which is the point. Commons turned out to be the weak link: red-carpet
portraits, 0-18% full-body depending on the sample, 87% Western clothing, and exactly one
menswear item across 146 outfits. Instagram is current, far more often full-length, and
44% ethnic on the sample measured so far.

So a celebrity needs neither a Commons photo nor a prior profile here. Their body shape
comes from whichever Instagram post shows the most of them, and their outfits from the
rest. That is what makes a menswear corpus possible at all, which Commons could not
supply.

Prerequisites
-------------
None beyond the mirror being reachable: no Instagram account, no session, no cookies.
See fashion.adapters.source_imginn for why that route is closed and this one is not.

Usage
-----
    # Build the menswear corpus: Indian men, 5 posts each
    uv run python scripts/scrape_instagram.py --wardrobe menswear --limit-celebrities 20

    # See who would be scraped, without touching the network
    uv run python scripts/scrape_instagram.py --wardrobe menswear --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from fashion.adapters.source_imginn import ImginnImageSource
from fashion.config import load_settings
from fashion.core.dataset import (
    AlreadyRunningError,
    CelebrityRepository,
    OutfitRepository,
    RunLock,
    iter_jsonl,
)
from fashion.core.models import (
    CelebrityProfile,
    Culture,
    Neckline,
    Occasion,
    OutfitItem,
    Silhouette,
    WaistEmphasis,
    Wardrobe,
    as_str_tuple,
)
from fashion.pipeline.ingest import _enum_or
from fashion.ports.vision import VisionModel

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RosterEntry:
    qid: str
    name: str
    region: str
    handle: str
    wardrobe: str


def roster_entries(roster_path: Path) -> list[RosterEntry]:
    """Roster rows carrying an Instagram handle."""
    out: list[RosterEntry] = []
    for row in iter_jsonl(roster_path):
        qid = str(row.get("qid", "")).rsplit("/", 1)[-1]
        handle = str(row.get("instagram", "")).strip().lstrip("@")
        if qid and handle:
            out.append(
                RosterEntry(
                    qid=qid,
                    name=str(row.get("name", "")),
                    region=str(row.get("region", "")),
                    handle=handle,
                    wardrobe=str(row.get("wardrobe", "")),
                )
            )
    return out


def build_vision(kind: str) -> VisionModel:
    if kind == "fake":
        from fashion.adapters.vision_fake import FakeVisionModel

        return FakeVisionModel()

    from fashion.adapters.vision_proxy import ProxyVisionModel

    settings = load_settings()
    proxy = ProxyVisionModel(
        settings.proxy_url,
        model=settings.proxy_model,
        cache_dir=settings.cache_dir / "proxy",
    )
    if not proxy.available():
        raise SystemExit(
            f"the proxy at {settings.proxy_url} is not reachable. Start it with "
            "`npx antigravity-claude-proxy@latest start`."
        )
    return proxy


def select(
    entries: list[RosterEntry],
    *,
    region: str | None,
    wardrobe: str | None,
    names: list[str],
    already: set[str],
    limit: int | None,
) -> list[RosterEntry]:
    """Choose who to scrape.

    `wardrobe` comes from Wikidata P21 and is used only to target *sourcing* -- it
    answers "whose Instagram should I read to build a menswear corpus", never anything
    about a user.
    """
    wanted = {n.casefold() for n in names}
    chosen = [
        e
        for e in entries
        if (not wanted or e.name.casefold() in wanted)
        and (not region or e.region == region)
        and (not wardrobe or e.wardrobe == wardrobe)
        # Skip anyone already scraped, so a resumed run extends the corpus rather than
        # re-paying for profiles it already has.
        and e.qid not in already
    ]
    return chosen[:limit] if limit else chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/seed"))
    parser.add_argument("--region", default="indian")
    parser.add_argument("--wardrobe", choices=["menswear", "womenswear"], default=None)
    parser.add_argument("--celebrity", action="append", default=[], help="repeatable")
    parser.add_argument("--limit-celebrities", type=int, default=10)
    parser.add_argument("--posts-per-celebrity", type=int, default=5)
    parser.add_argument("--vision", choices=["fake", "proxy"], default="proxy")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    entries = roster_entries(args.data_dir / "celebrities.jsonl")
    if not entries:
        raise SystemExit("no roster entries with handles; run scripts/fetch_roster.py")

    outfits = OutfitRepository(args.data_dir / "labels.jsonl")
    celebrities = CelebrityRepository(args.data_dir / "celebrity_profiles.jsonl")
    profiles = celebrities.index()
    labelled = outfits.labelled_ids()
    scraped = {i.split("-ig-")[0] for i in labelled if "-ig-" in i}

    targets = select(
        entries,
        region=args.region,
        wardrobe=args.wardrobe,
        names=args.celebrity,
        already=scraped,
        limit=args.limit_celebrities,
    )
    if not targets:
        raise SystemExit(
            f"nothing to scrape. {len(entries)} roster entries have handles; "
            f"{len(scraped)} already scraped."
        )

    print(f"{len(entries)} roster entries with handles, {len(scraped)} already scraped")
    print(f"targeting {len(targets)}, {args.posts_per_celebrity} posts each:\n")
    for entry in targets:
        print(f"  {entry.name[:26]:<28} @{entry.handle:<26} {entry.wardrobe or 'unknown'}")
    if args.dry_run:
        print("\ndry run: nothing requested")
        return 0

    source = ImginnImageSource(args.data_dir / "instagram")
    vision = build_vision(args.vision)
    written = skipped = failed = new_profiles = 0

    try:
        lock = RunLock(args.data_dir / ".scrape.lock")
    except AlreadyRunningError as exc:
        raise SystemExit(str(exc)) from exc

    with lock:
        for entry in targets:
            print(f"\n@{entry.handle}  ({entry.name})")
            best_confidence = -1.0

            for sourced in source.fetch(entry.handle, limit=args.posts_per_celebrity):
                path = Path(sourced.local_path)
                outfit_id = f"{entry.qid}-ig-{path.stem.rsplit('_', 1)[-1]}"
                if outfit_id in labelled:
                    skipped += 1
                    continue

                try:
                    image = path.read_bytes()
                    tags = vision.tag_outfit(image)
                    caption = vision.caption_outfit(image)
                except Exception:
                    failed += 1
                    continue

                # Body shape comes from whichever post shows the most of the body. A
                # celebrity with no usable full-length post gets no profile at all,
                # rather than one resting on an estimated hip width.
                try:
                    metrics = vision.analyze_body(image)
                    if metrics.full_body and metrics.confidence > best_confidence:
                        best_confidence = metrics.confidence
                        profiles[entry.qid] = CelebrityProfile(
                            id=entry.qid,
                            name=entry.name,
                            region=entry.region,
                            shape=metrics.shape,
                            build=metrics.build,
                            height_band=metrics.height_band,
                            full_body=True,
                            shape_confidence=metrics.confidence,
                        )
                except Exception:
                    # Unmeasurable is normal and not a reason to discard the outfit:
                    # the garment is still a valid corpus item for someone else.
                    pass

                if not tags or not tags.get("garment_type"):
                    skipped += 1
                    continue
                if tags.get("outfit_clearly_visible") is False:
                    skipped += 1
                    continue

                try:
                    outfits.append(
                        OutfitItem(
                            id=outfit_id,
                            celebrity_id=entry.qid,
                            image_path=str(path),
                            garment_type=str(tags.get("garment_type") or "unknown"),
                            silhouette=_enum_or(  # type: ignore[arg-type]
                                tags.get("silhouette"), Silhouette, Silhouette.STRAIGHT
                            ),
                            neckline=_enum_or(  # type: ignore[arg-type]
                                tags.get("neckline"), Neckline, Neckline.ROUND
                            ),
                            waist_emphasis=_enum_or(  # type: ignore[arg-type]
                                tags.get("waist_emphasis"),
                                WaistEmphasis,
                                WaistEmphasis.NATURAL,
                            ),
                            culture=_enum_or(  # type: ignore[arg-type]
                                tags.get("culture"), Culture, Culture.ETHNIC
                            ),
                            occasion=_enum_or(  # type: ignore[arg-type]
                                tags.get("occasion"), Occasion, Occasion.CASUAL
                            ),
                            wardrobe=_enum_or(  # type: ignore[arg-type]
                                tags.get("wardrobe"), Wardrobe, Wardrobe.UNISEX
                            ),
                            colors=as_str_tuple(tags.get("colors")),
                            patterns=as_str_tuple(tags.get("patterns")),
                            style_tags=as_str_tuple(tags.get("style_tags")),
                            caption=caption,
                            source=sourced.source,
                            license=sourced.license,
                        )
                    )
                except Exception:
                    failed += 1
                    continue

                labelled.add(outfit_id)
                written += 1
                print(
                    f"  + {tags.get('garment_type')} ({tags.get('culture')}/{tags.get('wardrobe')})"
                )

            if entry.qid in profiles and best_confidence >= 0:
                new_profiles += 1
                celebrities.save(profiles.values())
                print(f"  profile: {profiles[entry.qid].shape.value} (full-body measured)")

    print(f"\nwritten {written}  profiles {new_profiles}  skipped {skipped}  failed {failed}")
    print("re-run scripts/build_index.py to index")
    return 0


if __name__ == "__main__":
    sys.exit(main())
