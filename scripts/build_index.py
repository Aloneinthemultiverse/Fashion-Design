"""Build the retrieval index from the labelled corpus.

Computes both named vectors per outfit and upserts them with the filterable payload.

Usage:
    uv run python scripts/build_index.py --embed fake --verify
    uv run python scripts/build_index.py --embed openclip   # needs `uv sync --extra embed`
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fashion.adapters.embed_fake import FakeEmbedder
from fashion.adapters.store_memory import InMemoryVectorStore
from fashion.core.dataset import CelebrityRepository, OutfitRepository
from fashion.pipeline.index import IndexBuilder
from fashion.ports.embedder import Embedder
from fashion.ports.vectorstore import CAPTION_VECTOR, IMAGE_VECTOR, VectorStore


def build_embedder(kind: str) -> Embedder:
    if kind == "fake":
        return FakeEmbedder()
    from fashion.adapters.embed_openclip import OpenClipEmbedder

    return OpenClipEmbedder()


def build_store(kind: str) -> VectorStore:
    if kind == "memory":
        return InMemoryVectorStore()
    from fashion.adapters.store_qdrant import QdrantVectorStore
    from fashion.config import load_settings

    settings = load_settings()
    return QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        collection=settings.qdrant_collection,
    )


def _tops(hits: list, item_id: str) -> bool:
    """True when `item_id` is among the joint-highest-scoring hits.

    Two outfits with identical captions embed to the same vector, so demanding strict
    rank 1 would report a failure for what is a genuine tie rather than an indexing
    fault. Accepting anything at the top score keeps the check meaningful without
    flagging ties.
    """
    if not hits:
        return False
    best = hits[0].score
    return any(h.id == item_id for h in hits if h.score >= best - 1e-9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/seed"))
    parser.add_argument("--embed", choices=["fake", "openclip"], default="fake")
    parser.add_argument("--store", choices=["memory", "qdrant"], default="memory")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After building, check each item is retrievable by its own vectors.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    outfits = OutfitRepository(args.data_dir / "labels.jsonl")
    items, report = outfits.load()
    if not items:
        raise SystemExit(f"no labelled outfits in {outfits.path}. Run scripts/ingest.py first.")
    if not report.ok:
        print(f"warning: skipped {report.skipped} malformed rows")

    profiles = CelebrityRepository(args.data_dir / "celebrity_profiles.jsonl").index()

    embedder = build_embedder(args.embed)
    store = build_store(args.store)
    stats = IndexBuilder(embedder, store).build(items, profiles)

    print(f"\nindexed             {stats.indexed}")
    print(f"skipped (no profile){stats.skipped_no_profile:>4}")
    print(f"skipped (no image)  {stats.skipped_no_image:>4}")
    print(f"failed              {stats.failed}")

    if args.verify and args.store == "memory":
        print("\nverifying round-trip...")
        ok = fail = 0
        for item in items[: stats.indexed]:
            path = Path(item.image_path)
            if not path.exists():
                continue
            image_ok = _tops(
                store.search(embedder.embed_image(path.read_bytes()), using=IMAGE_VECTOR, limit=5),
                item.id,
            )
            caption = item.caption.strip() or f"a {item.garment_type}"
            caption_ok = _tops(
                store.search(embedder.embed_text(caption), using=CAPTION_VECTOR, limit=5),
                item.id,
            )
            if image_ok and caption_ok:
                ok += 1
            else:
                fail += 1
                channel = "image" if not image_ok else "caption"
                print(f"  MISMATCH {item.id} ({channel} vector)")
        print(f"  top-scoring on both vectors: {ok} ok, {fail} failed")
        if fail:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
