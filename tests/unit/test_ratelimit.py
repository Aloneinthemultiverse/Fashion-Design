"""Rate limiting, quota and caching.

These exist because the Gemini free tier is ~1,500 requests/day. Without them one loop
exhausts it and the system is dead until midnight UTC, so the bounds are the product
behaviour, not an optimisation.
"""

from __future__ import annotations

import pytest

from fashion.core.feedback import FeedbackEvent, FeedbackLog, Verdict
from fashion.core.ratelimit import (
    DailyQuota,
    SlidingWindowLimiter,
    TtlCache,
    cache_key,
)

# -- sliding window ---------------------------------------------------------------


def test_requests_inside_the_budget_are_allowed() -> None:
    limiter = SlidingWindowLimiter(3, 60)
    assert [limiter.check("ip", now=float(i)).allowed for i in range(3)] == [True] * 3


def test_exceeding_the_budget_is_refused_with_a_retry_hint() -> None:
    limiter = SlidingWindowLimiter(2, 60)
    limiter.check("ip", now=0.0)
    limiter.check("ip", now=1.0)
    decision = limiter.check("ip", now=2.0)
    assert not decision.allowed
    assert decision.retry_after_seconds == pytest.approx(58.0)


def test_the_window_slides_rather_than_resetting_on_a_boundary() -> None:
    """A fixed-boundary counter lets a caller spend the budget twice across it."""
    limiter = SlidingWindowLimiter(2, 60)
    limiter.check("ip", now=0.0)
    limiter.check("ip", now=1.0)
    assert not limiter.check("ip", now=30.0).allowed
    # Hits age out one at a time as the window slides past each, rather than the whole
    # budget being handed back at a fixed boundary.
    assert limiter.check("ip", now=60.5).allowed  # the t=0.0 hit has aged out
    assert not limiter.check("ip", now=60.9).allowed  # t=1.0 and t=60.5 still in window
    assert limiter.check("ip", now=61.5).allowed  # now t=1.0 ages out too


def test_limits_are_tracked_per_key() -> None:
    limiter = SlidingWindowLimiter(1, 60)
    assert limiter.check("a", now=0.0).allowed
    assert limiter.check("b", now=0.0).allowed
    assert not limiter.check("a", now=0.0).allowed


def test_remaining_counts_down() -> None:
    limiter = SlidingWindowLimiter(3, 60)
    assert limiter.check("ip", now=0.0).remaining == 2
    assert limiter.check("ip", now=0.1).remaining == 1


def test_limiter_rejects_nonsense_configuration() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        SlidingWindowLimiter(0, 60)
    with pytest.raises(ValueError, match="window_seconds must be positive"):
        SlidingWindowLimiter(1, 0)


# -- daily quota ------------------------------------------------------------------


def test_quota_allows_spending_up_to_the_ceiling() -> None:
    quota = DailyQuota(3)
    assert all(quota.spend().allowed for _ in range(3))
    assert quota.used == 3


def test_quota_refuses_past_the_ceiling_and_says_when_it_resets() -> None:
    quota = DailyQuota(1)
    quota.spend()
    decision = quota.spend()
    assert not decision.allowed
    assert 0 < decision.retry_after_seconds <= 86400


def test_quota_rejects_an_oversized_single_spend() -> None:
    """Partial spends must not sneak past the ceiling."""
    quota = DailyQuota(5)
    assert not quota.spend(6).allowed
    assert quota.used == 0


# -- TTL cache --------------------------------------------------------------------


def test_cache_returns_what_was_stored() -> None:
    cache = TtlCache()
    cache.set("k", {"v": 1}, 60)
    assert cache.get("k") == {"v": 1}


def test_expired_entries_are_misses() -> None:
    cache = TtlCache()
    cache.set("k", "v", 0.001)
    import time

    time.sleep(0.01)
    assert cache.get("k") is None


def test_a_zero_ttl_is_not_stored() -> None:
    cache = TtlCache()
    cache.set("k", "v", 0)
    assert cache.get("k") is None


def test_cache_is_bounded_and_evicts_least_recently_used() -> None:
    """An unbounded result cache in a long-lived process is a memory leak."""
    cache = TtlCache(max_entries=2)
    cache.set("a", 1, 60)
    cache.set("b", 2, 60)
    cache.get("a")  # 'a' becomes most recently used, so 'b' is next out
    cache.set("c", 3, 60)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_hit_rate_is_reported_for_metrics() -> None:
    cache = TtlCache()
    cache.set("k", "v", 60)
    cache.get("k")
    cache.get("missing")
    assert cache.hit_rate == 0.5


def test_cache_key_ignores_dict_ordering() -> None:
    """Two identical queries built in a different order must hit the same entry."""
    assert cache_key({"a": 1, "b": 2}) == cache_key({"b": 2, "a": 1})


def test_cache_key_distinguishes_different_inputs() -> None:
    assert cache_key("photo-hash", {"text": "saree"}) != cache_key("photo-hash", {"text": "gown"})


# -- feedback ---------------------------------------------------------------------


def event(verdict: Verdict, *, confirmed: bool, shape: str = "pear") -> FeedbackEvent:
    return FeedbackEvent(
        outfit_id="o1", verdict=verdict, body_shape=shape, shape_was_confirmed=confirmed
    )


def test_feedback_round_trips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    log.record(event(Verdict.UP, confirmed=True))
    stored = list(log.read())
    assert len(stored) == 1
    assert stored[0].verdict is Verdict.UP


def test_summary_separates_confirmed_shape_satisfaction(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Mixing them lets bad shape inference masquerade as bad recommendations."""
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    log.record(event(Verdict.UP, confirmed=True))
    log.record(event(Verdict.DOWN, confirmed=False))
    log.record(event(Verdict.DOWN, confirmed=False))

    summary = log.summary()
    assert summary["total"] == 3
    # Overall satisfaction looks poor...
    assert summary["satisfaction"] == pytest.approx(0.333, abs=0.01)
    # ...but every rejection came from an unconfirmed shape reading.
    assert summary["satisfaction_confirmed_shape"] == 1.0


def test_summary_breaks_down_by_shape(tmp_path) -> None:  # type: ignore[no-untyped-def]
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    log.record(event(Verdict.UP, confirmed=True, shape="pear"))
    log.record(event(Verdict.DOWN, confirmed=True, shape="apple"))
    assert log.summary()["by_shape"] == {
        "pear": {"up": 1, "down": 0},
        "apple": {"up": 0, "down": 1},
    }


def test_empty_log_summarises_without_dividing_by_zero(tmp_path) -> None:  # type: ignore[no-untyped-def]
    summary = FeedbackLog(tmp_path / "absent.jsonl").summary()
    assert summary["total"] == 0
    assert summary["satisfaction"] is None


def test_corrupt_rows_are_skipped(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "feedback.jsonl"
    log = FeedbackLog(path)
    log.record(event(Verdict.UP, confirmed=True))
    with path.open("a", encoding="utf-8") as fh:
        fh.write("not json\n")
    assert len(list(log.read())) == 1


def test_recording_never_raises_on_an_unwritable_path(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Losing a thumbs-up is trivial; failing a user's request over it is not."""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory", encoding="utf-8")
    FeedbackLog(blocker / "feedback.jsonl").record(event(Verdict.UP, confirmed=True))
