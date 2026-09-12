"""Instagram adapter guardrails.

No network and no instaloader install are needed: what is tested here is the opt-in gate
and the rate limiter, which are the two mechanisms that exist to stop this adapter from
being used accidentally or from getting the operator's account restricted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fashion.adapters.source_instagram import (
    LICENSE,
    InstagramImageSource,
    TermsNotAcceptedError,
    _RateLimiter,
)


def test_adapter_refuses_to_construct_without_explicit_acknowledgement(
    tmp_path: Path,
) -> None:
    """Defaulting to off is the point: this must never be reachable by accident."""
    with pytest.raises(TermsNotAcceptedError, match="i_accept_terms_risk=True"):
        InstagramImageSource(tmp_path)


def test_refusal_message_states_the_actual_consequences(tmp_path: Path) -> None:
    """A gate the operator cannot understand is a gate they will bypass blindly."""
    with pytest.raises(TermsNotAcceptedError) as exc:
        InstagramImageSource(tmp_path)
    message = str(exc.value).lower()
    assert "terms of service" in message
    assert "restricting" in message or "ban" in message
    assert "copyright" in message or "redistributed" in message


def test_constructs_when_acknowledged_and_given_a_loader(tmp_path: Path) -> None:
    """An injected loader keeps construction offline and instaloader-free."""
    source = InstagramImageSource(tmp_path, i_accept_terms_risk=True, loader=object())
    assert isinstance(source, InstagramImageSource)


def test_images_are_recorded_as_all_rights_reserved() -> None:
    """Downstream code filters on this to keep copyrighted images out of public output."""
    assert LICENSE == "all-rights-reserved"


def test_rate_limiter_spaces_requests_even_inside_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hourly budget alone permits a burst, which is what actually gets throttled.

    A floor between consecutive requests spreads them out regardless of how much budget
    remains, so all but the first call waits.
    """
    slept: list[float] = []
    monkeypatch.setattr("fashion.adapters.source_instagram.time.sleep", lambda s: slept.append(s))
    limiter = _RateLimiter(5, min_gap=3.0)
    for _ in range(5):
        limiter.acquire()
    assert len(slept) == 4
    assert all(0 < s <= 3.0 for s in slept)


def test_rate_limiter_sleeps_out_the_window_once_the_budget_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exceeding the hourly budget is the most common way to get an account restricted."""
    slept: list[float] = []
    monkeypatch.setattr("fashion.adapters.source_instagram.time.sleep", lambda s: slept.append(s))
    limiter = _RateLimiter(2, min_gap=0.0)
    for _ in range(3):
        limiter.acquire()
    # One long sleep: the remainder of the hour, not the short inter-request gap.
    assert len(slept) == 1
    assert slept[0] > 60


def test_rate_limiter_rejects_a_nonsensical_budget() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _RateLimiter(0)


def test_adapter_satisfies_the_image_source_port(tmp_path: Path) -> None:
    from fashion.ports.imagesource import ImageSource

    source: Any = InstagramImageSource(tmp_path, i_accept_terms_risk=True, loader=object())
    assert isinstance(source, ImageSource)


def test_anonymous_use_is_refused_with_the_fix(tmp_path: Path) -> None:
    """Anonymous access returns 401 from Instagram, so this must fail early and say how.

    A vague failure here sends the operator hunting for a bug that does not exist; the
    message names the exact command that creates a session.
    """
    from fashion.adapters.source_instagram import SessionRequiredError

    with pytest.raises(SessionRequiredError) as exc:
        InstagramImageSource(tmp_path, i_accept_terms_risk=True)
    assert "--load-cookies" in str(exc.value)


def test_a_missing_session_file_is_reported_not_ignored(tmp_path: Path) -> None:
    from fashion.adapters.source_instagram import SessionRequiredError

    with pytest.raises(SessionRequiredError):
        InstagramImageSource(
            tmp_path, i_accept_terms_risk=True, session_file=tmp_path / "absent.session"
        )
