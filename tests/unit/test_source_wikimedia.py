"""Wikimedia Commons filtering.

Network calls are not exercised here; the filters are. They are what decide whether an
unusable or unlicensed file enters the corpus, and a licence-filter regression is the
kind of bug that is invisible until someone asks where an image came from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fashion.adapters.source_wikimedia import CommonsFile, WikimediaImageSource, _plain


@pytest.fixture
def source(tmp_path: Path) -> WikimediaImageSource:
    return WikimediaImageSource(tmp_path, delay_seconds=0.0)


def photo(**kwargs: object) -> CommonsFile:
    defaults: dict[str, object] = {
        "title": "File:Someone at an event.jpg",
        "url": "https://upload.wikimedia.org/x.jpg",
        "width": 800,
        "height": 1200,
        "licence": "CC BY-SA 4.0",
        "author": "A Photographer",
        "mediatype": "BITMAP",
    }
    defaults.update(kwargs)
    return CommonsFile(**defaults)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "licence",
    ["CC0", "cc0", "CC BY 4.0", "CC BY-SA 3.0", "Public domain", "PD"],
)
def test_permissive_licences_are_accepted(source: WikimediaImageSource, licence: str) -> None:
    assert source._is_redistributable(licence)


@pytest.mark.parametrize(
    "licence",
    ["Fair use", "CC BY-NC 4.0", "CC BY-ND 4.0", "All rights reserved", ""],
)
def test_non_redistributable_licences_are_rejected(
    source: WikimediaImageSource, licence: str
) -> None:
    """Anything not explicitly permissive must be skipped, not downloaded and triaged."""
    assert not source._is_redistributable(licence)


def test_audio_is_rejected_despite_a_plausible_thumbnail_size(
    source: WikimediaImageSource,
) -> None:
    """Commons reports a synthetic 1024x1024 thumbnail for audio files.

    A size-only filter therefore accepts a .wav -- this happened in practice, pulling a
    name-pronunciation recording into the corpus. mediatype is the reliable signal.
    """
    wav = photo(
        title="File:LL-Q9610-Titodutta-name.wav",
        width=1024,
        height=1024,
        mediatype="AUDIO",
    )
    assert not source._is_plausible_outfit_photo(wav)


def test_video_is_rejected(source: WikimediaImageSource) -> None:
    assert not source._is_plausible_outfit_photo(photo(mediatype="VIDEO"))


def test_landscape_images_are_rejected(source: WikimediaImageSource) -> None:
    """Full-body outfit shots are portrait; landscape is usually a group shot or crop."""
    assert not source._is_plausible_outfit_photo(photo(width=1600, height=900))


def test_small_images_are_rejected(source: WikimediaImageSource) -> None:
    assert not source._is_plausible_outfit_photo(photo(width=200, height=300))


@pytest.mark.parametrize(
    "title",
    [
        "File:Deepika-Padukone-Signature.jpg",
        "File:Some Movie Poster.jpg",
        "File:Studio logo.jpg",
        "File:Diagram.svg",
    ],
)
def test_non_photographs_are_rejected_by_title(
    source: WikimediaImageSource, title: str
) -> None:
    assert not source._is_plausible_outfit_photo(photo(title=title))


def test_a_genuine_portrait_photo_is_accepted(source: WikimediaImageSource) -> None:
    assert source._is_plausible_outfit_photo(photo())


def test_plain_strips_the_html_commons_wraps_around_attribution() -> None:
    html = '<a href="//commons.wikimedia.org/wiki/User:Sudheerbs" title="x">Sudheerbs</a>'
    assert _plain(html) == "Sudheerbs"


def test_plain_collapses_whitespace() -> None:
    assert _plain("<span>  Own   work </span>") == "Own work"
