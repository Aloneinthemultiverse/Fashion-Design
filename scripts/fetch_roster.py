"""Build the celebrity roster from Wikidata.

Writes `data/seed/celebrities.jsonl`, one candidate per line. This is the *directory*
only -- names, regions and a Commons image URL. Body shape, build and height band are
visual judgements and are filled in by the labelling pass, not guessed here.

Usage:
    uv run python scripts/fetch_roster.py --indian 1000 --american 600 --british 400
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from fashion.adapters.source_wikidata import WikidataCelebrityDirectory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indian", type=int, default=1000)
    parser.add_argument("--american", type=int, default=600)
    parser.add_argument("--british", type=int, default=400)
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--out", type=Path, default=Path("data/seed/celebrities.jsonl"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    targets = {
        region: count
        for region, count in (
            ("indian", args.indian),
            ("american", args.american),
            ("british", args.british),
        )
        if count > 0
    }

    directory = WikidataCelebrityDirectory()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Write region by region rather than all at the end. A roster run takes minutes and
    # depends on a public endpoint that intermittently 502s; losing everything because
    # the last region failed would waste all the preceding work.
    roster = []
    by_region: dict[str, int] = {}
    by_wardrobe: dict[str, int] = {}
    seen: set[str] = set()

    with args.out.open("w", encoding="utf-8") as fh:
        for region, wanted in targets.items():
            print(f"fetching {wanted} from {region}...", flush=True)
            found = directory.fetch_roster({region: wanted}, page_size=args.page_size)
            for candidate in found:
                if candidate.id in seen:
                    continue
                seen.add(candidate.id)
                roster.append(candidate)
                fh.write(json.dumps(asdict(candidate), ensure_ascii=False) + "\n")
            fh.flush()
            by_region[region] = len(found)
            for candidate in found:
                key = candidate.wardrobe or "unknown"
                by_wardrobe[key] = by_wardrobe.get(key, 0) + 1
            print(f"  {region}: {len(found)}", flush=True)

    print(f"wrote {len(roster)} celebrities to {args.out}")
    for region, count in sorted(by_region.items()):
        print(f"  {region:<10} {count:>5}")
    print("by wardrobe:")
    for key, count in sorted(by_wardrobe.items()):
        print(f"  {key:<12} {count:>5}")
    with_ig = sum(1 for c in roster if c.instagram)
    print(f"with Instagram handle: {with_ig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
