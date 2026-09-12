"""Instagram image source: the dynamic, current-season layer.

Wikimedia anchors body geometry but is archival, and -- measured on this corpus -- only
about one image in eight shows the whole body, because Commons is red-carpet portraiture.
Instagram is the opposite: current, and far more often full-length. That is what this is
for.

**Anonymous access does not work.** Verified three ways against live Instagram: a stealth
browser fetch returns the login wall, `web_profile_info` with the public app id returns
401 `require_login`, and instaloader 4.15.3 anonymous returns the same 401. This is a
standing issue on instaloader's tracker (#2487, #2501, #2508, #2511, #2540), and reported
anonymous rate limits are 1-2 requests per 30 seconds where they work at all. So a
session is required, full stop.

The session is established out-of-band and never handled here:

    uv run instaloader --load-cookies chrome --sessionfile ig.session :feed

That imports an existing browser login. No password passes through this process, and none
is stored in the repository.

What the operator accepts by enabling this, stated once, plainly:

* It breaches Instagram's Terms of Service, and the realistic consequence is a restricted
  account or IP — the operator's own, which is why it must be their session and not a
  shared one.
* The photographs are copyrighted. They can be analysed locally, but serving them from a
  gallery is redistribution we have no right to. Items are recorded as
  `all-rights-reserved` so downstream code can keep them out of anything public.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fashion.ports.imagesource import SourcedImage

log = logging.getLogger(__name__)

LICENSE = "all-rights-reserved"

# Deliberately below the architecture document's 100/hour. That figure predates the
# current restrictions, and exceeding the real limit is the single most common way to
# get an account restricted.
DEFAULT_REQUESTS_PER_HOUR = 60
RECENT_WINDOW_DAYS = 30

# Instagram now throttles aggressively even with a session; a hard floor between
# requests matters more than the hourly budget for avoiding a soft ban.
MIN_SECONDS_BETWEEN_REQUESTS = 3.0


class TermsNotAcceptedError(RuntimeError):
    """Raised when the adapter is used without explicit acknowledgement."""


class InstaloaderNotInstalledError(RuntimeError):
    """Raised when the optional instaloader dependency is missing."""


class SessionRequiredError(RuntimeError):
    """Raised when no usable session is available.

    A distinct type because it is the expected failure and the only actionable one:
    every other error here is transient, this one needs the operator to log in.
    """


class _RateLimiter:
    """Sliding window plus a hard minimum gap between requests.

    The window alone permits a burst that empties the budget in seconds, which is what
    actually triggers throttling; the floor spreads requests out regardless.
    """

    def __init__(
        self, requests_per_hour: int, min_gap: float = MIN_SECONDS_BETWEEN_REQUESTS
    ) -> None:
        if requests_per_hour <= 0:
            raise ValueError("requests_per_hour must be positive")
        self._budget = requests_per_hour
        self._window = 3600.0
        self._min_gap = min_gap
        self._timestamps: list[float] = []
        self._last = 0.0

    def acquire(self) -> None:
        gap = time.monotonic() - self._last
        if self._last and gap < self._min_gap:
            time.sleep(self._min_gap - gap)

        now = time.monotonic()
        self._timestamps = [t for t in self._timestamps if now - t < self._window]
        if len(self._timestamps) >= self._budget:
            sleep_for = self._window - (now - self._timestamps[0])
            log.warning("instagram hourly budget reached; sleeping %.0fs", sleep_for)
            time.sleep(max(0.0, sleep_for))
            self._timestamps = []

        self._last = time.monotonic()
        self._timestamps.append(self._last)


class InstagramImageSource:
    """Fetches recent posts for a handle, using an operator-supplied session."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        i_accept_terms_risk: bool = False,
        session_file: Path | None = None,
        session_user: str | None = None,
        requests_per_hour: int = DEFAULT_REQUESTS_PER_HOUR,
        recent_days: int = RECENT_WINDOW_DAYS,
        loader: Any = None,
    ) -> None:
        if not i_accept_terms_risk:
            raise TermsNotAcceptedError(
                "InstagramImageSource requires i_accept_terms_risk=True. Automated "
                "collection breaches Instagram's Terms of Service, risks restricting "
                "the account and IP used, and returns copyrighted images that cannot "
                "be redistributed. Use WikimediaImageSource unless you have decided to "
                "accept those consequences."
            )
        self._cache_dir = cache_dir
        self._recent_days = recent_days
        self._limiter = _RateLimiter(requests_per_hour)
        self._loader = (
            loader if loader is not None else self._build_loader(session_file, session_user)
        )

    @staticmethod
    def _build_loader(session_file: Path | None, session_user: str | None) -> Any:
        try:
            import instaloader
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise InstaloaderNotInstalledError(
                "instaloader is not installed. Run `uv pip install instaloader`."
            ) from exc

        loader = instaloader.Instaloader(
            download_videos=False,
            download_video_thumbnails=False,
            download_comments=False,
            download_geotags=False,
            save_metadata=False,
            compress_json=False,
            quiet=True,
        )

        if session_file and Path(session_file).exists():
            # load_session_from_file keys sessions by username; passing an empty one
            # with an explicit path loads whatever that file holds, which is what the
            # --load-cookies flow produces.
            try:
                loader.load_session_from_file("", str(session_file))
                log.info("loaded instagram session from %s", session_file)
                return loader
            except Exception as exc:
                raise SessionRequiredError(
                    f"session file {session_file} could not be loaded: {exc}"
                ) from exc

        if session_user:
            try:
                loader.load_session_from_file(session_user)
                log.info("loaded instagram session for %s", session_user)
                return loader
            except FileNotFoundError as exc:
                raise SessionRequiredError(
                    f"no saved session for {session_user!r}. Create one with:\n"
                    f"  instaloader --load-cookies chrome --sessionfile ig.session :feed"
                ) from exc

        raise SessionRequiredError(
            "Instagram requires a session; anonymous access returns 401. Create one "
            "from an existing browser login with:\n"
            "  instaloader --load-cookies chrome --sessionfile ig.session :feed\n"
            "then pass session_file=Path('ig.session')."
        )

    def fetch(self, celebrity: str, *, limit: int = 10) -> Iterator[SourcedImage]:
        """Yield recent photo posts for an Instagram handle.

        `celebrity` is the handle, not the display name: Wikidata knows display names
        and Instagram knows handles, and the caller joins the two.
        """
        import instaloader

        handle = celebrity.strip().lstrip("@")
        if not handle:
            return

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cutoff = datetime.now(UTC) - timedelta(days=self._recent_days)

        self._limiter.acquire()
        try:
            profile = instaloader.Profile.from_username(self._loader.context, handle)
        except Exception:
            # Private, renamed, deleted or rate-limited. None of these should abort a
            # batch running over hundreds of celebrities.
            log.warning("could not open instagram profile %r", handle, exc_info=True)
            return

        if profile.is_private:
            log.info("skipping private profile %r", handle)
            return

        yielded = 0
        for post in profile.get_posts():
            if yielded >= limit:
                return
            if post.date_utc.replace(tzinfo=UTC) < cutoff:
                # Posts arrive newest-first, so the first old one ends the window.
                return
            if post.is_video:
                continue

            dest = self._cache_dir / f"{handle}_{post.shortcode}.jpg"
            if not dest.exists():
                self._limiter.acquire()
                try:
                    self._loader.download_pic(str(dest.with_suffix("")), post.url, post.date_utc)
                except Exception:
                    log.warning("failed to download %s", post.shortcode, exc_info=True)
                    continue
            if not dest.exists():
                continue

            yield SourcedImage(
                local_path=str(dest),
                source=f"https://www.instagram.com/p/{post.shortcode}/",
                license=LICENSE,
                celebrity_hint=handle,
            )
            yielded += 1

    def fetch_one(self, url: str, *, celebrity: str | None = None) -> SourcedImage | None:
        """Present so this satisfies the same shape the ingest pipeline expects.

        Instagram is a per-profile source, not a per-URL one, so single-URL fetching is
        not meaningful here; callers use `fetch`.
        """
        return None
