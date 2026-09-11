"""Ingest pipeline.

Runs entirely against fakes. The behaviours under test are the ones that protect scarce
VLM quota and corpus integrity: resume without re-spending quota, never write a record
without provenance, and drop images whose garment tags would be meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from fashion.adapters.vision_fake import FakeVisionModel
from fashion.core.dataset import CelebrityRepository, OutfitRepository
from fashion.pipeline.ingest import Ingestor, RosterRow
from fashion.ports.imagesource import SourcedImage


@dataclass
class StubSource:
    """Returns a real file on disk for every row, recording how often it was called."""

    image_path: Path
    calls: int = 0
    return_none: bool = False

    def fetch_one(self, url: str, *, celebrity: str | None = None) -> SourcedImage | None:
        self.calls += 1
        if self.return_none:
            return None
        return SourcedImage(
            local_path=str(self.image_path),
            source="https://commons.wikimedia.org/wiki/File:X.jpg",
            license="CC BY-SA 4.0 (author: Someone)",
            celebrity_hint=celebrity,
        )


class UnclearVision(FakeVisionModel):
    def tag_outfit(self, image: bytes) -> dict[str, object]:
        tags = super().tag_outfit(image)
        tags["outfit_clearly_visible"] = False
        return tags


class ExplodingVision(FakeVisionModel):
    def tag_outfit(self, image: bytes) -> dict[str, object]:
        raise RuntimeError("quota exhausted")


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"fake-jpeg-bytes")
    return path


@pytest.fixture
def repos(tmp_path: Path) -> tuple[OutfitRepository, CelebrityRepository]:
    return (
        OutfitRepository(tmp_path / "labels.jsonl"),
        CelebrityRepository(tmp_path / "celebs.jsonl"),
    )


def rows(n: int = 2) -> list[RosterRow]:
    return [
        RosterRow(
            id=f"Q{i}",
            name=f"Person {i}",
            region="indian",
            image_url=f"https://commons.wikimedia.org/wiki/Special:FilePath/p{i}.jpg",
        )
        for i in range(n)
    ]


def test_writes_outfits_and_profiles(
    image: Path, repos: tuple[OutfitRepository, CelebrityRepository]
) -> None:
    outfits, celebrities = repos
    stats = Ingestor(FakeVisionModel(), StubSource(image), outfits, celebrities).run(rows(2))

    assert stats.written == 2
    items, report = outfits.load()
    assert report.ok
    assert len(items) == 2
    assert set(celebrities.index()) == {"Q0", "Q1"}


def test_every_written_item_carries_provenance(
    image: Path, repos: tuple[OutfitRepository, CelebrityRepository]
) -> None:
    """An item without source and licence can never be audited or published."""
    outfits, celebrities = repos
    Ingestor(FakeVisionModel(), StubSource(image), outfits, celebrities).run(rows(2))
    items, _ = outfits.load()
    assert all(i.source.startswith("https://") and i.license for i in items)


def test_resume_skips_already_labelled_without_touching_the_source(
    image: Path, repos: tuple[OutfitRepository, CelebrityRepository]
) -> None:
    """The scarce resource is VLM quota; a resumed run must not re-spend it."""
    outfits, celebrities = repos
    source = StubSource(image)
    Ingestor(FakeVisionModel(), source, outfits, celebrities).run(rows(2))
    first_calls = source.calls

    stats = Ingestor(FakeVisionModel(), source, outfits, celebrities).run(rows(2))
    assert stats.skipped_existing == 2
    assert stats.written == 0
    assert source.calls == first_calls  # no re-fetch, therefore no re-analysis


def test_unclear_outfits_are_dropped(
    image: Path, repos: tuple[OutfitRepository, CelebrityRepository]
) -> None:
    """Face close-ups produce confident but meaningless garment tags."""
    outfits, celebrities = repos
    stats = Ingestor(UnclearVision(), StubSource(image), outfits, celebrities).run(rows(2))
    assert stats.skipped_unclear == 2
    assert stats.written == 0


def test_unclear_outfits_can_be_kept_when_requested(
    image: Path, repos: tuple[OutfitRepository, CelebrityRepository]
) -> None:
    outfits, celebrities = repos
    ingestor = Ingestor(
        UnclearVision(), StubSource(image), outfits, celebrities, require_clear_outfit=False
    )
    assert ingestor.run(rows(1)).written == 1


def test_missing_or_unlicensed_images_are_counted_not_fatal(
    image: Path, repos: tuple[OutfitRepository, CelebrityRepository]
) -> None:
    outfits, celebrities = repos
    source = StubSource(image, return_none=True)
    stats = Ingestor(FakeVisionModel(), source, outfits, celebrities).run(rows(3))
    assert stats.skipped_no_image == 3
    assert stats.written == 0


def test_a_vlm_failure_does_not_abort_the_batch(
    image: Path, repos: tuple[OutfitRepository, CelebrityRepository]
) -> None:
    """One bad response must not end a run that spans a day of quota."""
    outfits, celebrities = repos
    stats = Ingestor(ExplodingVision(), StubSource(image), outfits, celebrities).run(rows(3))
    assert stats.failed == 3
    assert stats.seen == 3


def test_limit_caps_the_run(
    image: Path, repos: tuple[OutfitRepository, CelebrityRepository]
) -> None:
    outfits, celebrities = repos
    stats = Ingestor(FakeVisionModel(), StubSource(image), outfits, celebrities).run(
        rows(10), limit=3
    )
    assert stats.seen == 3


def test_roster_row_parsing_requires_the_essential_fields() -> None:
    assert RosterRow.from_dict({"qid": "http://www.wikidata.org/entity/Q42"}) is None
    assert RosterRow.from_dict({"name": "X", "image_url": "u"}) is None

    row = RosterRow.from_dict(
        {
            "qid": "http://www.wikidata.org/entity/Q42",
            "name": "Someone",
            "region": "indian",
            "image_url": "https://example.org/x.jpg",
        }
    )
    assert row is not None
    assert row.id == "Q42"  # bare Q-id, not the full entity URL
