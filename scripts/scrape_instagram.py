"""Dynamic layer: pull recent Instagram outfits for celebrities already in the corpus.

This is the architecture document's dynamic scraping step. The static Wikimedia corpus
supplies who exists and their body geometry; this supplies what they are wearing now.

Two reasons it is worth the trouble, both measured rather than assumed:

* Commons is archival. Its photographs are red-carpet portraits, and on this corpus only
  about one in eight shows the whole body — so most Wikimedia-derived body shapes rest on
  an estimated hip width. Instagram posts are far more often full-length.
* Commons has no notion of a season. Instagram is current by construction.

Targeting is deliberate, not a sweep. You scrape the celebrities a user was actually
matched to, which is a handful, rather than the whole roster — that is what keeps the
request count survivable.

Prerequisites
-------------
Anonymous access returns 401; a session is required. Create one from an existing browser
login, which never exposes a password to this process:

    uv run instaloader --load-cookies chrome --sessionfile ig.session :feed

Usage
-----
    # Everyone matching a body shape, 5 recent posts each:
    uv run python scripts/scrape_instagram.py --shape pear --limit-celebrities 10

    # Specific people:
    uv run python scripts/scrape_instagram.py --celebrity "Vidya Malvade"

    # See who would be scraped, without touching Instagram:
    uv run python scripts/scrape_instagram.py --shape pear --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fashion.adapters.source_instagram import (
    InstagramImageSource,
    SessionRequiredError,
)
from fashion.config import load_settings
from fashion.core.dataset import CelebrityRepository, OutfitRepository, iter_jsonl
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
from fashion.pipeline.ingest import _enum_or
from fashion.ports.vision import VisionModel

log = logging.getLogger(__name__)


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


def handles_by_id(roster_path: Path) -> dict[str, str]:
    """Q-id -> Instagram handle.

    Handles live in the roster and body shapes live in the profiles; joining on Q-id
    keeps the handle out of the profile schema, so adding this layer does not require
    re-running the multi-hour labelling pass.
    """
    out: dict[str, str] = {}
    for row in iter_jsonl(roster_path):
        qid = str(row.get("qid", "")).rsplit("/", 1)[-1]
        handle = str(row.get("instagram", "")).strip().lstrip("@")
        if qid and handle:
            out[qid] = handle
    return out


def select(
    profiles: dict[str, CelebrityProfile],
    handles: dict[str, str],
    *,
    shape: str | None,
    region: str | None,
    names: list[str],
    limit: int | None,
) -> list[tuple[CelebrityProfile, str]]:
    """Choose who to scrape, newest-measurable first."""
    chosen: list[tuple[CelebrityProfile, str]] = []
    wanted = {n.casefold() for n in names}

    for qid, profile in profiles.items():
        handle = handles.get(qid)
        if not handle:
            continue
        if wanted and profile.name.casefold() not in wanted:
            continue
        if shape and profile.shape.value != shape:
            continue
        if region and profile.region != region:
            continue
        chosen.append((profile, handle))

    # Celebrities whose recorded shape came from a full-body photo first: their geometry
    # is measured rather than estimated, so fresh outfits for them are worth more.
    chosen.sort(key=lambda row: (not row[0].shape_is_reliable, row[0].name))
    return chosen[:limit] if limit else chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/seed"))
    parser.add_argument("--session-file", type=Path, default=Path("ig.session"))
    parser.add_argument("--shape", default=None, help="only this body shape")
    parser.add_argument("--region", default="indian")
    parser.add_argument("--celebrity", action="append", default=[], help="repeatable")
    parser.add_argument("--limit-celebrities", type=int, default=10)
    parser.add_argument("--posts-per-celebrity", type=int, default=5)
    parser.add_argument("--vision", choices=["fake", "proxy"], default="proxy")
    parser.add_argument("--dry-run", action="store_true", help="list targets only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    profiles = CelebrityRepository(args.data_dir / "celebrity_profiles.jsonl").index()
    handles = handles_by_id(args.data_dir / "celebrities.jsonl")
    if not profiles:
        raise SystemExit("no celebrity profiles; run scripts/ingest.py first")

    targets = select(
        profiles,
        handles,
        shape=args.shape,
        region=args.region,
        names=args.celebrity,
        limit=args.limit_celebrities,
    )
    if not targets:
        raise SystemExit(
            "no celebrities matched with a known Instagram handle. "
            f"{len(handles)} of {len(profiles)} profiles have one."
        )

    with_handles = sum(1 for qid in profiles if qid in handles)
    print(f"{with_handles} of {len(profiles)} labelled celebrities have an Instagram handle")
    print(f"targeting {len(targets)} celebrities, {args.posts_per_celebrity} posts each:\n")
    for profile, handle in targets:
        flag = "measured" if profile.shape_is_reliable else "estimated"
        print(f"  {profile.name[:26]:<28} @{handle:<24} {profile.shape.value} ({flag})")

    if args.dry_run:
        print("\ndry run: nothing was requested from Instagram")
        return 0

    try:
        source = InstagramImageSource(
            args.data_dir / "instagram",
            i_accept_terms_risk=True,
            session_file=args.session_file,
            recent_days=30,
        )
    except SessionRequiredError as exc:
        raise SystemExit(f"\n{exc}") from exc

    vision = build_vision(args.vision)
    outfits = OutfitRepository(args.data_dir / "labels.jsonl")
    already = outfits.labelled_ids()

    written = skipped = failed = 0
    for profile, handle in targets:
        print(f"\n@{handle}")
        for sourced in source.fetch(handle, limit=args.posts_per_celebrity):
            shortcode = Path(sourced.local_path).stem.rsplit("_", 1)[-1]
            outfit_id = f"{profile.id}-ig-{shortcode}"
            if outfit_id in already:
                skipped += 1
                continue

            try:
                image = Path(sourced.local_path).read_bytes()
                tags = vision.tag_outfit(image)
                caption = vision.caption_outfit(image)
            except Exception:
                log.warning("analysis failed for %s", shortcode, exc_info=True)
                failed += 1
                continue

            if tags.get("outfit_clearly_visible") is False:
                skipped += 1
                continue

            try:
                outfits.append(
                    OutfitItem(
                        id=outfit_id,
                        celebrity_id=profile.id,
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
                        culture=_enum_or(tags.get("culture"), Culture, Culture.ETHNIC),  # type: ignore[arg-type]
                        occasion=_enum_or(  # type: ignore[arg-type]
                            tags.get("occasion"), Occasion, Occasion.CASUAL
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
                log.warning("invalid record for %s", shortcode, exc_info=True)
                failed += 1
                continue

            written += 1
            print(f"  + {tags.get('garment_type')} ({tags.get('culture')}) {sourced.source}")

    print(f"\nwritten {written}  skipped {skipped}  failed {failed}")
    print("re-run scripts/build_index.py to index the new outfits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
