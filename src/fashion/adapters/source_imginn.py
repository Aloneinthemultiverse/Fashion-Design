"""Instagram content without a login, via a public mirror.

Instagram itself is closed to anonymous clients: a stealth browser fetch gets the login
wall, `web_profile_info` with the public app id returns 401, and instaloader 4.15.3
anonymous returns the same. Getting in needs the operator's own session, which means
their account carries the ban risk, and on Windows even importing that session is blocked
by Chrome's App-Bound Encryption.

Public mirrors re-serve the same public profiles and are reachable without any of that.
This adapter reads one, which removes the two practical objections at once: no credential
handling, and no risk to the operator's account.

What it does *not* remove is copyright. These are still photographs owned by the
photographer or agency. They can be analysed locally; serving them from a gallery is
redistribution. Items are recorded as `all-rights-reserved` so downstream code can keep
them out of anything public.

A mirror is third-party infrastructure with no stability guarantee -- it may rate-limit,
change markup, or vanish. Every failure path here returns empty rather than raising, so
the dynamic layer degrades to the static corpus instead of taking a request down.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fashion.ports.imagesource import SourcedImage

log = logging.getLogger(__name__)

MIRROR = "https://imginn.com"
LICENSE = "all-rights-reserved"

# The mirror re-hosts post images on its own CDN; anything else on the page is chrome.
POST_IMAGE_HOST = "imginn.com"
ASSET_MARKER = "assets.imginn.com"

# The profile avatar is served at a fixed thumbnail size and is not an outfit.
AVATAR_WIDTH = "77"

MIN_SECONDS_BETWEEN_PROFILES = 4.0
DOWNLOAD_TIMEOUT = 45.0

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Post images carry a numeric media id at the start of the filename; it is stable and
# unique, which makes it a better item id than a hash of the URL, whose query string
# contains expiring tokens.
MEDIA_ID = re.compile(r"/(\d{6,})_")


@dataclass(frozen=True, slots=True)
class MirrorPost:
    image_url: str
    caption: str

    @property
    def media_id(self) -> str:
        match = MEDIA_ID.search(self.image_url)
        return match.group(1) if match else str(abs(hash(self.image_url)))


class ImginnImageSource:
    """Reads a public Instagram mirror. No account, no session, no cookies."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        min_gap: float = MIN_SECONDS_BETWEEN_PROFILES,
        fetcher: Any = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._min_gap = min_gap
        self._last = 0.0
        self._fetcher = fetcher

    def _get_fetcher(self) -> Any:
        if self._fetcher is not None:
            return self._fetcher
        try:
            from scrapling.fetchers import StealthyFetcher
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "scrapling is not installed. Run:\n"
                "  uv pip install 'scrapling[fetchers]'\n"
                "  uv run scrapling install"
            ) from exc
        self._fetcher = StealthyFetcher
        return self._fetcher

    def _throttle(self) -> None:
        gap = time.monotonic() - self._last
        if self._last and gap < self._min_gap:
            time.sleep(self._min_gap - gap)
        self._last = time.monotonic()

    def _posts(self, handle: str) -> list[MirrorPost]:
        """Scrape one profile page into post image URLs and captions."""
        self._throttle()
        try:
            page = self._get_fetcher().fetch(
                f"{MIRROR}/{handle}/",
                headless=True,
                solve_cloudflare=True,
                network_idle=True,
                timeout=90000,
            )
        except Exception:
            log.warning("mirror fetch failed for %r", handle, exc_info=True)
            return []

        if getattr(page, "status", 0) != 200:
            log.warning("mirror returned %s for %r", getattr(page, "status", "?"), handle)
            return []

        posts: list[MirrorPost] = []
        for element in page.css("img"):
            attrs = element.attrib
            src = str(attrs.get("src") or "")
            if not src or ASSET_MARKER in src or POST_IMAGE_HOST not in src:
                continue
            # The avatar is the only image carrying an explicit width attribute.
            if str(attrs.get("width") or "") == AVATAR_WIDTH:
                continue
            posts.append(MirrorPost(image_url=src, caption=str(attrs.get("alt") or "").strip()))
        return posts

    def fetch(self, celebrity: str, *, limit: int = 10) -> Iterator[SourcedImage]:
        """Yield recent post images for an Instagram handle.

        `celebrity` is the handle. Recency is implicit: the mirror lists newest first, so
        taking the head of the list is the last-30-days window in practice. Post dates
        are not exposed reliably, so this cannot filter on them and does not pretend to.
        """
        handle = celebrity.strip().lstrip("@")
        if not handle:
            return

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        yielded = 0

        for post in self._posts(handle):
            if yielded >= limit:
                return

            dest = self._cache_dir / f"{handle}_{post.media_id}.jpg"
            if not dest.exists():
                data = self._download(post.image_url)
                if data is None:
                    continue
                try:
                    dest.write_bytes(data)
                except OSError:
                    log.warning("could not write %s", dest, exc_info=True)
                    continue

            yield SourcedImage(
                local_path=str(dest),
                # Credit the original platform, not the mirror: the mirror is how it was
                # reached, but the post is where it came from.
                source=f"https://www.instagram.com/{handle}/",
                license=LICENSE,
                celebrity_hint=handle,
            )
            yielded += 1

    def _download(self, url: str) -> bytes | None:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                data: bytes = response.read()
        except (urllib.error.URLError, TimeoutError, OSError):
            log.warning("image download failed", exc_info=True)
            return None

        # A short body or a non-JPEG magic number is an error page, and writing it would
        # put a file on disk that only fails later when something tries to decode it.
        if len(data) < 2048 or not data.startswith((b"\xff\xd8", b"\x89PNG")):
            log.warning("mirror returned %d bytes that are not an image", len(data))
            return None
        return data

    def fetch_one(self, url: str, *, celebrity: str | None = None) -> SourcedImage | None:
        """Not meaningful here: this is a per-profile source, not a per-URL one."""
        return None
