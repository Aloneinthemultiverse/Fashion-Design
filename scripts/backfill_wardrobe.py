"""Assign a wardrobe to outfits labelled before the field existed.

Re-labelling the corpus to add one field would cost a full pass of model quota. Garment
type already determines the answer for the cases that matter -- a saree is womenswear and
a sherwani is menswear regardless of who is photographed in it -- so this derives it
offline and leaves genuinely ambiguous garments as unisex.

Deliberately conservative: anything not clearly one tradition stays unisex, because a
unisex item is eligible in every search while a wrongly-assigned one is invisible in half
of them. Guessing wrong is worse than not guessing.

Usage:
    uv run python scripts/backfill_wardrobe.py --dry-run
    uv run python scripts/backfill_wardrobe.py
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from fashion.core.dataset import OutfitRepository
from fashion.core.models import Wardrobe

# Substrings, matched against the garment type. Indian traditional wear is strongly
# gendered by garment name, which is what makes this reliable at all.
WOMENSWEAR = (
    "saree",
    "sari",
    "lehenga",
    "anarkali",
    "salwar",
    "churidar",
    "sharara",
    "gharara",
    "choli",
    "dupatta",
    "kurti",
    "gown",
    "dress",
    "skirt",
    "blouse",
    "crop top",
    "jumpsuit",
    "bodysuit",
    "peplum",
)

MENSWEAR = (
    "sherwani",
    "bandhgala",
    "dhoti",
    "lungi",
    "veshti",
    "mundu",
    "kurta pyjama",
    "kurta-pyjama",
    "nehru jacket",
    "achkan",
    "tuxedo",
    "suit and tie",
    "necktie",
    "bow tie",
)


def classify(garment_type: str) -> Wardrobe:
    name = garment_type.casefold()
    # Menswear first: "kurta pyjama" would otherwise be caught by nothing, while several
    # womenswear terms are substrings of longer menswear names.
    if any(term in name for term in MENSWEAR):
        return Wardrobe.MENSWEAR
    if any(term in name for term in WOMENSWEAR):
        return Wardrobe.WOMENSWEAR
    return Wardrobe.UNISEX


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/seed"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = OutfitRepository(args.data_dir / "labels.jsonl")
    items, report = repo.load()
    if not items:
        raise SystemExit("no labelled outfits")
    if not report.ok:
        print(f"warning: skipped {report.skipped} malformed rows")

    updated = []
    changes: collections.Counter[str] = collections.Counter()
    examples: dict[str, list[str]] = collections.defaultdict(list)

    for item in items:
        # Never overwrite a wardrobe the model actually produced.
        if item.wardrobe is not Wardrobe.UNISEX:
            updated.append(item)
            changes["already set"] += 1
            continue
        wardrobe = classify(item.garment_type)
        updated.append(item.model_copy(update={"wardrobe": wardrobe}))
        changes[wardrobe.value] += 1
        if len(examples[wardrobe.value]) < 5:
            examples[wardrobe.value].append(item.garment_type)

    print(f"{len(items)} outfits\n")
    for key, count in changes.most_common():
        print(f"  {key:<14} {count:>4}")
        for sample in examples.get(key, []):
            print(f"      {sample}")

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    repo.save(updated)
    print(f"\nwrote {len(updated)} outfits")
    print("re-run scripts/build_index.py to reindex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
