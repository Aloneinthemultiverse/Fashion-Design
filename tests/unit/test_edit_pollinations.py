"""Hosted edit adapter: offline behaviour only; no request leaves the machine."""

from __future__ import annotations

from fashion.adapters.edit_pollinations import (
    PollinationsEditClient,
    PollinationsEditGenerationProvider,
    PollinationsTryOnProvider,
)


def test_unconfigured_client_is_unavailable_and_returns_none() -> None:
    client = PollinationsEditClient("")
    assert not PollinationsTryOnProvider(client).available
    assert PollinationsTryOnProvider(client).try_on(b"p", b"g") is None


def test_generation_claims_reference_support() -> None:
    assert PollinationsEditGenerationProvider(PollinationsEditClient("k")).supports_references


class _Fallback:
    def generate(self, prompt: str, *, seed: int | None = None) -> bytes:
        return b"text-only"


def test_unreferenced_generation_uses_the_text_fallback() -> None:
    provider = PollinationsEditGenerationProvider(PollinationsEditClient("k"), _Fallback())
    assert provider.generate("a kurta") == b"text-only"
