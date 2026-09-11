"""Label the roster into the outfit corpus.

Downloads each celebrity's Commons image and runs one VLM pass over it, writing
`data/seed/labels.jsonl` and `data/seed/celebrity_profiles.jsonl`.

Resumable: already-labelled ids are skipped, so an interrupted run continues where it
stopped. That matters because a full 2,000-image pass exceeds one day of free Gemini
quota.

Usage:
    # Validate the pipeline with no API key and no quota spend:
    uv run python scripts/ingest.py --limit 25 --vision fake

    # Real labelling (needs FASHION_GEMINI_API_KEY):
    uv run python scripts/ingest.py --limit 1400 --vision gemini
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fashion.adapters.source_wikimedia import WikimediaImageSource
from fashion.adapters.vision_fake import FakeVisionModel
from fashion.config import load_settings
from fashion.core.dataset import CelebrityRepository, OutfitRepository, iter_jsonl
from fashion.pipeline.ingest import Ingestor, RosterRow
from fashion.ports.vision import VisionModel


def build_vision(kind: str) -> VisionModel:
    if kind == "fake":
        return FakeVisionModel()

    from fashion.adapters.vision_gemini import GeminiVisionModel

    settings = load_settings()
    if not settings.gemini_api_key:
        raise SystemExit(
            "FASHION_GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey, put it in .env, and re-run. "
            "Use --vision fake to validate the pipeline without one."
        )
    return GeminiVisionModel(
        settings.gemini_api_key,
        model=settings.gemini_model,
        cache_dir=settings.cache_dir / "vlm",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=Path("data/seed/celebrities.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/seed"))
    parser.add_argument("--vision", choices=["fake", "gemini"], default="fake")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--keep-unclear",
        action="store_true",
        help="Keep images where the outfit is obscured (default: drop them).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows = [
        row
        for row in (RosterRow.from_dict(raw) for raw in iter_jsonl(args.roster))
        if row is not None
    ]
    if not rows:
        raise SystemExit(f"no roster rows in {args.roster}. Run scripts/fetch_roster.py first.")

    ingestor = Ingestor(
        build_vision(args.vision),
        WikimediaImageSource(args.data_dir / "images"),
        OutfitRepository(args.data_dir / "labels.jsonl"),
        CelebrityRepository(args.data_dir / "celebrity_profiles.jsonl"),
        require_clear_outfit=not args.keep_unclear,
    )

    stats = ingestor.run(rows, limit=args.limit)

    print(f"\nseen              {stats.seen}")
    print(f"written           {stats.written}")
    print(f"skipped (existing){stats.skipped_existing:>4}")
    print(f"skipped (no image){stats.skipped_no_image:>4}")
    print(f"skipped (unclear) {stats.skipped_unclear:>4}")
    print(f"failed            {stats.failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
