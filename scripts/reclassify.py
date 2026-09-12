"""Recompute body shapes from cached measurements.

Classification thresholds are tuning parameters, and tuning them should not cost a
second pass over the vision model. Every analysis is cached with its raw widths, so
shapes can be re-derived offline: change `BALANCED_TOLERANCE`, run this, done.

It also repairs profiles written before a threshold change, which is the situation this
was written for -- a long labelling run was already in flight when the tolerance was
corrected, and restarting it would have thrown away hours of completed work for no
reason.

Usage:
    uv run python scripts/reclassify.py --dry-run
    uv run python scripts/reclassify.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

from fashion.core.body import BALANCED_TOLERANCE, Proportions, classify, confidence_from_ratios
from fashion.core.dataset import CelebrityRepository, OutfitRepository
from fashion.core.models import BodyShape, CelebrityProfile


def measurements_by_image(cache_dir: Path) -> dict[str, dict[str, Any]]:
    """Map image content-hash -> body measurements.

    The cache is keyed by a hash of (task, model, prompt, image), which cannot be
    reversed to a file path. So entries are matched back to outfits by re-deriving the
    same digest from the image on disk.
    """
    out: dict[str, dict[str, Any]] = {}
    for path in cache_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        body = data.get("body")
        if isinstance(body, dict) and body.get("hip_width"):
            out[path.stem] = body
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/seed"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/proxy"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profiles_repo = CelebrityRepository(args.data_dir / "celebrity_profiles.jsonl")
    profiles = profiles_repo.index()
    if not profiles:
        raise SystemExit("no profiles to reclassify")

    outfits, _ = OutfitRepository(args.data_dir / "labels.jsonl").load()
    image_by_celebrity = {o.celebrity_id: Path(o.image_path) for o in outfits}

    # Rebuild the cache key the adapter uses, so a cached analysis can be found from
    # the image it came from.
    from fashion.adapters.prompts import ANALYSIS_PROMPT, ANALYSIS_SYSTEM
    from fashion.adapters.vision_proxy import ProxyVisionModel
    from fashion.config import load_settings

    settings = load_settings()
    probe = ProxyVisionModel(model=settings.proxy_model)
    cached = measurements_by_image(args.cache_dir)

    before = collections.Counter(p.shape.value for p in profiles.values())
    updated: dict[str, CelebrityProfile] = {}
    matched = missing = 0

    for cid, profile in profiles.items():
        image_path = image_by_celebrity.get(cid)
        if image_path is None or not image_path.exists():
            updated[cid] = profile
            missing += 1
            continue

        key = probe._cache_key(
            "analyze_all", ANALYSIS_PROMPT + ANALYSIS_SYSTEM, (image_path.read_bytes(),)
        )
        body = cached.get(key)
        if body is None:
            updated[cid] = profile
            missing += 1
            continue

        proportions = Proportions(
            shoulder=float(body.get("shoulder_width") or 1.0),
            waist=float(body.get("waist_width") or 1.0),
            hip=float(body.get("hip_width") or 1.0),
        )
        confidence = confidence_from_ratios(proportions)
        if not body.get("full_body_visible", True):
            confidence *= 0.5

        updated[cid] = profile.model_copy(
            update={
                "shape": classify(proportions),
                "shape_confidence": round(confidence, 3),
                "full_body": bool(body.get("full_body_visible", False)),
            }
        )
        matched += 1

    after = collections.Counter(p.shape.value for p in updated.values())

    print(f"balance tolerance: {BALANCED_TOLERANCE}")
    print(f"profiles: {len(profiles)}  recomputed: {matched}  no cached measurement: {missing}\n")
    print(f"{'shape':<20}{'before':>8}{'after':>8}")
    for shape in BodyShape:
        print(f"{shape.value:<20}{before.get(shape.value, 0):>8}{after.get(shape.value, 0):>8}")

    changed = sum(1 for c, p in profiles.items() if p.shape != updated[c].shape)
    print(f"\n{changed} profiles changed shape")

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    profiles_repo.save(updated.values())
    print(f"wrote {len(updated)} profiles")
    print("re-run scripts/build_index.py to reindex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
