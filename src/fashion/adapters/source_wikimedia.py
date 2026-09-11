"""Wikimedia Commons image source.

Commons hosts genuine photographs of public figures under explicit, machine-readable
licences, which makes it the one corpus we can build on without foreclosing a public
launch. Per-file metadata carries the licence, the author and the usage terms, so
`OutfitItem.source` and `.license` can be populated truthfully at ingest.

Two filters carry most of the weight:

* a **licence allowlist**, because Commons also hosts fair-use and non-commercial files
  that we must not redistribute; anything not explicitly permissive is skipped.
* a **portrait-shape and size filter**, because the goal is full-body outfit shots and
  a category contains logos, signatures, posters and head-and-shoulders crops too.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from fashion.ports.imagesource import SourcedImage

log = logging.getLogger(__name__)

API = "https://commons.wikimedia.org/w/api.php"

# Wikimedia's robot policy requires a User-Agent identifying the client with a contact
# route. Note that the stdlib client is used deliberately: Wikimedia's edge rejects
# httpx and several other third-party clients on TLS fingerprint alone, returning 403
# with a robot-policy notice regardless of headers. urllib is accepted, and using it
# also keeps this adapter dependency-free.
USER_AGENT = (
    "FashionRecommender/0.1 "
    "(https://github.com/example/fashion-design; research prototype) urllib/python"
)

# Only licences that permit redistribution with attribution. Everything else -- fair
# use, non-commercial, no-derivatives -- is skipped rather than downloaded and sorted
# out later, because an unusable file in the corpus is worse than a missing one.
ALLOWED_LICENCES = frozenset(
    {
        "cc0",
        "cc-by-1.0",
        "cc-by-2.0",
        "cc-by-2.5",
        "cc-by-3.0",
        "cc-by-4.0",
        "cc-by-sa-1.0",
        "cc-by-sa-2.0",
        "cc-by-sa-2.5",
        "cc-by-sa-3.0",
        "cc-by-sa-4.0",
        "pd",
        "public domain",
    }
)

# Filename fragments that reliably indicate something other than an outfit photograph.
NON_PHOTO_HINTS = (
    "signature",
    "logo",
    "poster",
    "autograph",
    "icon",
    "map",
    "chart",
    ".svg",
    ".ogv",
    ".webm",
    ".pdf",
)

MIN_WIDTH = 400
MIN_HEIGHT = 600  # Portrait orientation: full-body shots are taller than they are wide.


@dataclass(frozen=True, slots=True)
class CommonsFile:
    title: str
    url: str
    width: int
    height: int
    licence: str
    author: str
    mediatype: str = ""


def _plain(html: str) -> str:
    """Strip the HTML Commons wraps around author and credit fields."""
    out: list[str] = []
    depth = 0
    for ch in html:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return " ".join("".join(out).split())


class WikimediaImageSource:
    """Fetches licence-cleared photographs of a named person from Commons."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        delay_seconds: float = 0.2,
        min_width: int = MIN_WIDTH,
        min_height: int = MIN_HEIGHT,
        timeout: float = 30.0,
    ) -> None:
        self._cache_dir = cache_dir
        self._delay = delay_seconds
        self._min_width = min_width
        self._min_height = min_height
        self._timeout = timeout

    # -- Transport ---------------------------------------------------------------

    def _get(self, url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        time.sleep(self._delay)  # Be a polite API citizen.
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            data: bytes = response.read()
        return data

    def _api(self, params: dict[str, str]) -> dict[str, object]:
        payload = self._get(f"{API}?{urllib.parse.urlencode(params)}")
        parsed: dict[str, object] = json.loads(payload)
        return parsed

    # -- API calls ---------------------------------------------------------------

    def _category_files(self, celebrity: str, limit: int) -> list[str]:
        """List file titles in a person's Commons category."""
        data = self._api(
            {
                "action": "query",
                "format": "json",
                "list": "categorymembers",
                "cmtitle": f"Category:{celebrity}",
                "cmtype": "file",
                # Over-fetch: licence and size filters reject a large fraction.
                "cmlimit": str(min(limit * 8, 200)),
            }
        )
        query = data.get("query", {})
        members = query.get("categorymembers", []) if isinstance(query, dict) else []
        return [m["title"] for m in members]

    def _file_info(self, titles: list[str]) -> list[CommonsFile]:
        """Resolve URL, dimensions and licence for up to 50 files in one call."""
        if not titles:
            return []
        data = self._api(
            {
                "action": "query",
                "format": "json",
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|size|mediatype",
                "iiurlwidth": "1024",
                "titles": "|".join(titles[:50]),
            }
        )
        query = data.get("query", {})
        pages = query.get("pages", {}) if isinstance(query, dict) else {}

        out: list[CommonsFile] = []
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            if not info:
                continue
            meta = info.get("extmetadata", {})
            out.append(
                CommonsFile(
                    title=page.get("title", ""),
                    # thumburl is a scaled render; it keeps downloads small and avoids
                    # pulling 20MB originals we would immediately resize anyway.
                    url=info.get("thumburl") or info.get("url", ""),
                    width=int(info.get("thumbwidth") or info.get("width") or 0),
                    height=int(info.get("thumbheight") or info.get("height") or 0),
                    licence=str(meta.get("LicenseShortName", {}).get("value", "")).strip(),
                    author=_plain(str(meta.get("Artist", {}).get("value", ""))) or "unknown",
                    mediatype=str(info.get("mediatype", "")).upper(),
                )
            )
        return out

    # -- Filtering ---------------------------------------------------------------

    def _is_redistributable(self, licence: str) -> bool:
        return licence.strip().lower().replace(" ", "-") in {
            lic.replace(" ", "-") for lic in ALLOWED_LICENCES
        } or licence.strip().lower().startswith("public domain")

    def _is_plausible_outfit_photo(self, f: CommonsFile) -> bool:
        # Check mediatype before dimensions. Commons reports a synthetic 1024x1024
        # thumbnail for audio and video files, so a size-only filter lets a .wav through
        # while its real width/height are reported as 0.
        if f.mediatype not in {"BITMAP", "DRAWING", ""}:
            return False
        lowered = f.title.lower()
        if any(hint in lowered for hint in NON_PHOTO_HINTS):
            return False
        if f.width < self._min_width or f.height < self._min_height:
            return False
        # Landscape images are group shots or crops far more often than full-body
        # outfit photographs, which are portrait by construction.
        return f.height >= f.width

    # -- Port implementation -----------------------------------------------------

    def fetch(self, celebrity: str, *, limit: int = 10) -> Iterator[SourcedImage]:
        """Yield licence-cleared, downloaded photographs of `celebrity`."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        titles = self._category_files(celebrity, limit)
        if not titles:
            log.warning("no Commons category found for %r", celebrity)
            return

        yielded = 0
        for batch_start in range(0, len(titles), 50):
            if yielded >= limit:
                return
            for f in self._file_info(titles[batch_start : batch_start + 50]):
                if yielded >= limit:
                    return
                if not f.url or not self._is_redistributable(f.licence):
                    continue
                if not self._is_plausible_outfit_photo(f):
                    continue

                local = self._download(f)
                if local is None:
                    continue
                yield SourcedImage(
                    local_path=str(local),
                    source=f"https://commons.wikimedia.org/wiki/{quote(f.title)}",
                    license=f"{f.licence} (author: {f.author})",
                    celebrity_hint=celebrity,
                )
                yielded += 1

    def fetch_one(self, file_url: str, *, celebrity: str | None = None) -> SourcedImage | None:
        """Resolve and download a single Commons file given its FilePath URL.

        Wikidata's P18 property stores images as
        `http://commons.wikimedia.org/wiki/Special:FilePath/<name>`, which is a redirect
        and carries no licence information. Going back to the Commons API for the file's
        metadata is what lets provenance be recorded truthfully instead of assumed.

        Returns None when the file is not redistributable or is not a usable photograph,
        so callers can simply skip it.
        """
        name = urllib.parse.unquote(file_url.rsplit("/", 1)[-1])
        if not name:
            return None

        info = self._file_info([f"File:{name}"])
        if not info:
            return None
        f = info[0]
        if not f.url or not self._is_redistributable(f.licence):
            return None
        if not self._is_plausible_outfit_photo(f):
            return None

        local = self._download(f)
        if local is None:
            return None
        return SourcedImage(
            local_path=str(local),
            source=f"https://commons.wikimedia.org/wiki/{quote(f.title)}",
            license=f"{f.licence} (author: {f.author})",
            celebrity_hint=celebrity,
        )

    def _download(self, f: CommonsFile) -> Path | None:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in f.title)
        dest = self._cache_dir / safe
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        try:
            dest.write_bytes(self._get(f.url))
        except (urllib.error.URLError, OSError):
            log.warning("failed to download %s", f.title, exc_info=True)
            return None
        return dest
