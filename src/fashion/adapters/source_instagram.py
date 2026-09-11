"""Instagram image source (opt-in).

Supplies what Wikimedia Commons cannot: current-season outfits. Commons is archival and
red-carpet, so it anchors body-geometry matching well but says nothing about what is
being worn now. This adapter fills the recency layer.

It is **not enabled by default**, and callers must pass `i_accept_terms_risk=True`. Three
things the operator is accepting by doing so:

* Automated collection violates Instagram's Terms of Service. The realistic consequence
  is rate-limiting or a ban on the account and IP that run it -- which is why this needs
  the operator's own session, never a shared or throwaway one.
* Retrieved photographs are copyrighted by the photographer or agency. They may be
  analysed locally, but redistributing them -- which is what serving a recommendation
  gallery does -- requires rights we do not have. `OutfitItem.license` is recorded as
  `all-rights-reserved` so downstream code can filter these out of anything public.
* As of 2026 anonymous profile browsing is largely blocked, so a logged-in session is
  required in practice.

Rate limits here are deliberately conservative. The architecture document's 100
requests/hour is already aggressive for an authenticated session; exceeding it is the
single most common way to get an account restricted.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fashion.ports.imagesource import SourcedImage

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    pass

log = logging.getLogger(__name__)

LICENSE = "all-rights-reserved"
DEFAULT_REQUESTS_PER_HOUR = 100
RECENT_WINDOW_DAYS = 30


class TermsNotAcceptedError(RuntimeError):
    """Raised when the adapter is used without explicit acknowledgement."""


class InstaloaderNotInstalledError(RuntimeError):
    """Raised when the optional instaloader dependency is missing."""


class _RateLimiter:
    """Simple sliding-window limiter.

    Tracks actual request timestamps rather than sleeping a fixed interval, so a burst
    followed by idle time does not over-throttle, and a sustained run cannot exceed the
    hourly budget.
    """

    def __init__(self, requests_per_hour: int) -> None:
        if requests_per_hour <= 0:
            raise ValueError("requests_per_hour must be positive")
        self._budget = requests_per_hour
        self._window = 3600.0
        self._timestamps: list[float] = []

    def acquire(self) -> None:
        now = time.monotonic()
        self._timestamps = [t for t in self._timestamps if now - t < self._window]
        if len(self._timestamps) >= self._budget:
            sleep_for = self._window - (now - self._timestamps[0])
            log.warning("instagram rate limit reached; sleeping %.0fs", sleep_for)
            time.sleep(max(0.0, sleep_for))
            self._timestamps = []
        self._timestamps.append(time.monotonic())


class InstagramImageSource:
    """Fetches recent posts from a public profile using Instaloader."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        i_accept_terms_risk: bool = False,
        session_user: str | None = None,
        requests_per_hour: int = DEFAULT_REQUESTS_PER_HOUR,
        recent_days: int = RECENT_WINDOW_DAYS,
        loader: Any = None,
    ) -> None:
        if not i_accept_terms_risk:
            raise TermsNotAcceptedError(
                "InstagramImageSource requires i_accept_terms_risk=True. Automated "
                "collection breaches Instagram's Terms of Service, risks banning the "
                "account and IP used, and returns copyrighted images that cannot be "
                "redistributed. Use WikimediaImageSource unless you have decided to "
                "accept those consequences."
            )
        self._cache_dir = cache_dir
        self._recent_days = recent_days
        self._limiter = _RateLimiter(requests_per_hour)
        self._loader = loader if loader is not None else self._build_loader(session_user)

    @staticmethod
    def _build_loader(session_user: str | None) -> Any:
        try:
            import instaloader
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise InstaloaderNotInstalledError(
                "instaloader is not installed. Install the optional extra with "
                "`uv sync --extra instagram`."
            ) from exc

        loader = instaloader.Instaloader(
            download_videos=False,
            download_video_thumbnails=False,
            download_comments=False,
            save_metadata=False,
            compress_json=False,
            quiet=True,
        )
        if session_user:
            # Reuses a session established out-of-band via `instaloader -l <user>`, so no
            # credentials are ever handled by this process.
            try:
                loader.load_session_from_file(session_user)
                log.info("loaded instagram session for %s", session_user)
            except FileNotFoundError:
                log.warning(
                    "no saved session for %s; run `instaloader -l %s` first. "
                    "Anonymous access is largely blocked and will likely fail.",
                    session_user,
                    session_user,
                )
        return loader

    def fetch(self, celebrity: str, *, limit: int = 10) -> Iterator[SourcedImage]:
        """Yield recent post images for an Instagram handle.

        `celebrity` is the handle, not the display name -- the caller maps one to the
        other, because Wikidata knows display names and Instagram knows handles.
        """
        import instaloader

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cutoff = datetime.now(UTC) - timedelta(days=self._recent_days)

        self._limiter.acquire()
        try:
            profile = instaloader.Profile.from_username(self._loader.context, celebrity)
        except Exception:
            # Profile lookup fails for private, renamed, deleted or rate-limited
            # accounts. None of those should abort a batch run over many celebrities.
            log.warning("could not open instagram profile %r", celebrity, exc_info=True)
            return

        yielded = 0
        for post in profile.get_posts():
            if yielded >= limit:
                return
            post_date = post.date_utc.replace(tzinfo=UTC)
            if post_date < cutoff:
                # Posts arrive newest-first, so the first old one ends the window.
                return
            if post.is_video:
                continue

            self._limiter.acquire()
            dest = self._cache_dir / f"{celebrity}_{post.shortcode}.jpg"
            if not dest.exists():
                try:
                    self._loader.download_pic(str(dest.with_suffix("")), post.url, post.date_utc)
                except Exception:
                    log.warning("failed to download %s", post.shortcode, exc_info=True)
                    continue

            yield SourcedImage(
                local_path=str(dest),
                source=f"https://www.instagram.com/p/{post.shortcode}/",
                license=LICENSE,
                celebrity_hint=celebrity,
            )
            yielded += 1
