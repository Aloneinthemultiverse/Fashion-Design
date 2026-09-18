"""Deploy the Streamlit app to a Hugging Face Space (Docker SDK, free CPU tier).

Vercel cannot host this: Streamlit needs a long-lived server, torch exceeds its function
size limit, and generation outlasts its request timeout. A Space runs the container as-is.

The outfit images are not in git (539MB, licence-encumbered), so this stages a compacted
copy of only the images the index uses -- resized to 768px -- and uploads that with the
code. Paths are rewritten with forward slashes because the labels were written on
Windows and the Space runs Linux.

Prerequisite, once: `uv run hf auth login` with a write token.

Usage:
    uv run python scripts/deploy_space.py --space <user>/fashion-atelier
Then set the secrets FASHION_GEMINI_API_KEY and FASHION_POLLINATIONS_TOKEN in the
Space settings.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MAX_SIDE = 768
SPACE_HEADER = """---
title: Fashion Atelier
emoji: 👔
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

"""


def compact(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.thumbnail((MAX_SIDE, MAX_SIDE))
        if src.suffix.lower() in {".jpg", ".jpeg"}:
            image.convert("RGB").save(dst, quality=82, optimize=True)
        else:
            image.save(dst, optimize=True)


def stage(out: Path) -> tuple[int, int]:
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, out / name)
    shutil.copytree(ROOT / "src", out / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "deploy/space/Dockerfile", out / "Dockerfile")
    (out / "README.md").write_text(
        SPACE_HEADER + (ROOT / "README.md").read_text(encoding="utf-8"), encoding="utf-8"
    )

    seed = out / "data/seed"
    seed.mkdir(parents=True)
    for name in ("celebrities.jsonl", "celebrity_profiles.jsonl"):
        shutil.copy2(ROOT / "data/seed" / name, seed / name)

    images = missing = 0
    with (seed / "labels.jsonl").open("w", encoding="utf-8") as fh:
        for line in (ROOT / "data/seed/labels.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rel = str(row.get("image_path", "")).replace("\\", "/")
            src = ROOT / rel
            if not src.exists():
                missing += 1
                continue
            compact(src, out / rel)
            row["image_path"] = rel
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            images += 1
    return images, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", required=True, help="e.g. yourname/fashion-atelier")
    parser.add_argument("--public", action="store_true", help="default is private")
    parser.add_argument("--stage-only", type=Path, help="write the bundle here, no upload")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        out = args.stage_only or Path(tmp)
        out.mkdir(parents=True, exist_ok=True)
        images, missing = stage(out)
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
        print(f"staged {images} images ({missing} missing), {size:.0f}MB total")
        if args.stage_only:
            return 0

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            args.space,
            repo_type="space",
            space_sdk="docker",
            private=not args.public,
            exist_ok=True,
        )
        api.upload_folder(folder_path=str(out), repo_id=args.space, repo_type="space")
        print(f"https://huggingface.co/spaces/{args.space}")
        print("now add secrets FASHION_GEMINI_API_KEY and FASHION_POLLINATIONS_TOKEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
