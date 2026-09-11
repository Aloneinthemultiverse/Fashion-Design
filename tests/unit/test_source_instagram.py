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
    assert "ban" in message
    assert "copyright" in message or "redistributed" in message


def test_constructs_when_acknowledged_and_given_a_loader(tmp_path: Path) -> None:
    """An injected loader keeps construction offline and instaloader-free."""
    source = InstagramImageSource(
        tmp_path, i_accept_terms_risk=True, loader=object()
    )
    assert isinstance(source, InstagramImageSource)


def test_images_are_recorded_as_all_rights_reserved() -> None:
    """Downstream code filters on this to keep copyrighted images out of public output."""
    assert LICENSE == "all-rights-reserved"


def test_rate_limiter_allows_up_to_its_budget_without_sleeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []
    monkeypatch.setattr(
        "fashion.adapters.source_instagram.time.sleep", lambda s: slept.append(s)
    )
    limiter = _RateLimiter(5)
    for _ in range(5):
        limiter.acquire()
    assert slept == []


def test_rate_limiter_sleeps_once_the_budget_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exceeding the hourly budget is the most common way to get an account restricted."""
    slept: list[float] = []
    monkeypatch.setattr(
        "fashion.adapters.source_instagram.time.sleep", lambda s: slept.append(s)
    )
    limiter = _RateLimiter(2)
    for _ in range(3):
        limiter.acquire()
    assert len(slept) == 1
    assert slept[0] > 0


def test_rate_limiter_rejects_a_nonsensical_budget() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _RateLimiter(0)


def test_adapter_satisfies_the_image_source_port(tmp_path: Path) -> None:
    from fashion.ports.imagesource import ImageSource

    source: Any = InstagramImageSource(tmp_path, i_accept_terms_risk=True, loader=object())
    assert isinstance(source, ImageSource)
