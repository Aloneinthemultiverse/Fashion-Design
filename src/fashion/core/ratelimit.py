"""Rate limiting and TTL caching.

Two mechanisms that exist for the same reason: the Gemini free tier allows roughly 1,500
requests a day, and without limits a handful of users -- or one loop -- exhausts it in
minutes and the system is dead until midnight UTC.

Both are in-process. That is correct for a single node and wrong for several, where the
counters would be per-node and the effective limit would multiply. Redis-backed versions
belong behind the same interfaces when the deployment grows; the note is here so the
limitation is visible rather than discovered.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Decision:
    """Whether a request may proceed, and when to retry if not."""

    allowed: bool
    remaining: int
    retry_after_seconds: float = 0.0


class SlidingWindowLimiter:
    """Per-key sliding window.

    Tracks actual request timestamps rather than resetting a counter on a fixed
    boundary, which would let a caller spend the whole budget twice across the
    boundary -- exactly the burst that gets an API key throttled.
    """

    def __init__(self, limit: int, window_seconds: float) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._limit = limit
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, *, now: float | None = None) -> Decision:
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self._window:
                hits.popleft()

            if len(hits) >= self._limit:
                return Decision(
                    allowed=False,
                    remaining=0,
                    retry_after_seconds=round(self._window - (now - hits[0]), 3),
                )

            hits.append(now)
            return Decision(allowed=True, remaining=self._limit - len(hits))

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)


class DailyQuota:
    """A hard daily ceiling, for the VLM budget specifically.

    Separate from the sliding window because the failure modes differ: bursting is a
    fairness problem, while exhausting the day's quota is an outage that lasts until
    the provider resets it. Callers check this before spending, not after.
    """

    def __init__(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self._limit = limit
        self._used = 0
        self._day = self._today()
        self._lock = threading.Lock()

    @staticmethod
    def _today() -> int:
        # Gemini's free tier resets on UTC midnight, so the day is derived from UTC.
        return int(time.time() // 86400)

    def spend(self, amount: int = 1) -> Decision:
        with self._lock:
            today = self._today()
            if today != self._day:
                self._day, self._used = today, 0

            if self._used + amount > self._limit:
                seconds_left = 86400 - (time.time() % 86400)
                return Decision(
                    allowed=False,
                    remaining=max(0, self._limit - self._used),
                    retry_after_seconds=round(seconds_left, 1),
                )

            self._used += amount
            return Decision(allowed=True, remaining=self._limit - self._used)

    @property
    def used(self) -> int:
        with self._lock:
            return self._used


@dataclass
class TtlCache:
    """Bounded LRU cache with per-entry expiry.

    The architecture document's cache table gives different TTLs per result kind
    (24 hours for body-type matches, 7 days for celebrity-specific), so the TTL is a
    per-set argument rather than a fixed property of the cache.

    Bounded because an unbounded result cache in a long-lived process is a memory leak
    with extra steps.
    """

    max_entries: int = 512
    _data: OrderedDict[str, tuple[float, Any]] = field(default_factory=OrderedDict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    hits: int = 0
    misses: int = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires_at, value = entry
            if time.time() >= expires_at:
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            return
        with self._lock:
            self._data[key] = (time.time() + ttl_seconds, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 3) if total else 0.0

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


def cache_key(*parts: Any) -> str:
    """Stable key from arbitrary parts.

    Sorted JSON so that two equivalent queries whose dict keys were built in a
    different order still hit the same entry -- otherwise the cache silently misses on
    requests that are identical in every way that matters.
    """
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()
