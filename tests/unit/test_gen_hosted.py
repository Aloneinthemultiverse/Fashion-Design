"""Hosted generation provider.

Offline: the HTTP call is stubbed. What matters here is not that the service works but
that this adapter is honest about what it cannot do, because the ImageRAG loop makes
decisions based on `supports_references`.
"""

from __future__ import annotations

import urllib.error
from typing import Any

import pytest

from fashion.adapters.gen_pollinations import PollinationsGenerationProvider
from fashion.ports.generation import GenerationProvider

JPEG = b"\xff\xd8\xff\xe0" + b"x" * 4096
PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 4096


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def stub_urlopen(payload: bytes, captured: list[str] | None = None):  # type: ignore[no-untyped-def]
    def _open(request: Any, timeout: float = 0) -> FakeResponse:
        if captured is not None:
            captured.append(request.full_url)
        return FakeResponse(payload)

    return _open


def test_adapter_satisfies_the_port() -> None:
    assert isinstance(PollinationsGenerationProvider(), GenerationProvider)


def test_it_declares_no_reference_support() -> None:
    """The ImageRAG loop reads this to decide whether it can honestly proceed.

    Claiming support would make the loop present a plain text prompt as
    reference-guided generation.
    """
    assert PollinationsGenerationProvider().supports_references is False


def test_it_reports_available_without_a_probe() -> None:
    assert PollinationsGenerationProvider().available is True


def test_a_jpeg_response_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fashion.adapters.gen_pollinations.urllib.request.urlopen", stub_urlopen(JPEG)
    )
    assert PollinationsGenerationProvider().generate("a red saree") == JPEG


def test_a_png_response_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fashion.adapters.gen_pollinations.urllib.request.urlopen", stub_urlopen(PNG)
    )
    assert PollinationsGenerationProvider().generate("a red saree") == PNG


def test_an_html_error_page_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing a non-image through would surface a broken picture to the user."""
    monkeypatch.setattr(
        "fashion.adapters.gen_pollinations.urllib.request.urlopen",
        stub_urlopen(b"<html>rate limited</html>"),
    )
    assert PollinationsGenerationProvider().generate("a red saree") is None


def test_a_truncated_response_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fashion.adapters.gen_pollinations.urllib.request.urlopen",
        stub_urlopen(b"\xff\xd8\xff\xe0short"),
    )
    assert PollinationsGenerationProvider().generate("a red saree") is None


def test_network_failure_returns_none_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None is the contract for an unavailable backend; the pipeline already handles it."""

    def boom(request: Any, timeout: float = 0) -> None:
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr("fashion.adapters.gen_pollinations.urllib.request.urlopen", boom)
    assert PollinationsGenerationProvider().generate("a red saree") is None


def test_the_prompt_reaches_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []
    monkeypatch.setattr(
        "fashion.adapters.gen_pollinations.urllib.request.urlopen",
        stub_urlopen(JPEG, captured),
    )
    PollinationsGenerationProvider().generate("a crimson banarasi saree")
    assert "crimson%20banarasi%20saree" in captured[0]
    # Styling hints are appended so the result is a usable full-body fashion shot.
    assert "full%20body%20fashion%20photograph" in captured[0]


def test_a_seed_is_forwarded_for_reproducibility(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []
    monkeypatch.setattr(
        "fashion.adapters.gen_pollinations.urllib.request.urlopen",
        stub_urlopen(JPEG, captured),
    )
    PollinationsGenerationProvider().generate("a saree", seed=42)
    assert "seed=42" in captured[0]


def test_references_are_discarded_loudly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Reaching here with references means a caller ignored supports_references."""
    monkeypatch.setattr(
        "fashion.adapters.gen_pollinations.urllib.request.urlopen", stub_urlopen(JPEG)
    )
    with caplog.at_level("WARNING"):
        PollinationsGenerationProvider().generate("a saree", references=(b"ref",))
    assert "cannot condition on" in caplog.text
