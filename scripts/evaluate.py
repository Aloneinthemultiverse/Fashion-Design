"""Retrieval quality check.

Answers one question: does the embedder actually understand the images, or is it just
returning something? The fake embedder is hash-derived, so it passes every structural
test in the suite while being semantically meaningless. Nothing in the unit tests can
tell the two apart -- that is what this is for.

Two measures, both computed against the real corpus:

* **Self-retrieval** -- embed an item's own image, search, and check it ranks first.
  Any working index passes this, including the fake one. It is a wiring check.
* **Cross-modal agreement** -- embed each item's *caption* and check the item's own
  image ranks highly. Only a real dual-encoder can do this, because it requires the two
  towers to share a meaningful space. The fake embedder derives text and image vectors
  from unrelated bytes and scores at chance.

The gap between the two is the signal. A high self-retrieval rate with chance-level
cross-modal agreement means the index is wired correctly and understands nothing.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

from fashion.adapters.embed_fake import FakeEmbedder
from fashion.adapters.store_memory import InMemoryVectorStore
from fashion.core.dataset import CelebrityRepository, OutfitRepository
from fashion.pipeline.index import IndexBuilder
from fashion.ports.embedder import Embedder
from fashion.ports.vectorstore import IMAGE_VECTOR


def build_embedder(kind: str) -> Embedder:
    if kind == "fake":
        return FakeEmbedder()
    from fashion.adapters.embed_openclip import OpenClipEmbedder

    return OpenClipEmbedder()


def rank_of(hits: list, item_id: str) -> int | None:
    for position, hit in enumerate(hits, start=1):
        if hit.id == item_id:
            return position
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/seed"))
    parser.add_argument("--embed", choices=["fake", "openclip"], default="openclip")
    parser.add_argument("--limit", type=int, default=None, help="cap items, for speed")
    args = parser.parse_args()

    items, _ = OutfitRepository(args.data_dir / "labels.jsonl").load()
    profiles = CelebrityRepository(args.data_dir / "celebrity_profiles.jsonl").index()
    items = [i for i in items if Path(i.image_path).exists()]
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("no labelled outfits with images; run scripts/ingest.py first")

    print(f"embedder: {args.embed}")
    print(f"corpus:   {len(items)} outfits\n")

    embedder = build_embedder(args.embed)
    store = InMemoryVectorStore()
    IndexBuilder(embedder, store).build(items, profiles)

    self_hits = 0
    caption_ranks: list[int] = []
    n = store.count()

    for item in items:
        image = Path(item.image_path).read_bytes()

        hits = store.search(embedder.embed_image(image), using=IMAGE_VECTOR, limit=1)
        if hits and hits[0].id == item.id:
            self_hits += 1

        caption = item.caption.strip() or f"a {item.garment_type}"
        # Query the IMAGE vectors with a TEXT embedding: this only works if the two
        # towers share a space, which is exactly what the fake embedder lacks.
        cross = store.search(embedder.embed_text(caption), using=IMAGE_VECTOR, limit=n)
        rank = rank_of(cross, item.id)
        if rank is not None:
            caption_ranks.append(rank)

    chance = (n + 1) / 2  # expected rank under a uniformly random ordering
    median_rank = statistics.median(caption_ranks) if caption_ranks else float("nan")
    top5 = sum(1 for r in caption_ranks if r <= 5) / len(caption_ranks) if caption_ranks else 0.0

    print(f"self-retrieval rank-1     {self_hits}/{len(items)} ({self_hits / len(items):.0%})")
    print("  (wiring check -- any working index passes this)\n")
    print(f"cross-modal median rank   {median_rank:.1f}  (chance = {chance:.1f})")
    print(f"cross-modal top-5 rate    {top5:.0%}")
    print("  (semantic check -- only a real dual encoder beats chance)\n")

    if median_rank < chance * 0.6:
        print("VERDICT: the embedder understands the images.")
        return 0
    print(
        "VERDICT: cross-modal retrieval is at or near chance. The index is wired\n"
        "         correctly but carries no semantic signal -- expected for --embed fake."
    )
    return 0


if __name__ == "__main__":
    random.seed(0)
    sys.exit(main())
