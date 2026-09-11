"""Dataset persistence.

The corpus represents hours of VLM quota once labelling has run, so the behaviours that
matter most here are the ones that protect it: never truncate on an interrupted write,
never lose the whole file to one malformed row, and never silently re-spend quota.
"""

from __future__ import annotations

from pathlib import Path

from fashion.core.dataset import CelebrityRepository, OutfitRepository, iter_jsonl
from fashion.core.models import BodyShape, Build, CelebrityProfile, HeightBand
from tests.conftest import make_outfit


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    repo = OutfitRepository(tmp_path / "labels.jsonl")
    original = make_outfit("o1")
    repo.save([original])
    loaded, report = repo.load()
    assert report.ok
    assert loaded == [original]


def test_missing_file_loads_empty_rather_than_raising(tmp_path: Path) -> None:
    """A first run has no corpus yet; that is normal, not an error."""
    items, report = OutfitRepository(tmp_path / "nope.jsonl").load()
    assert items == []
    assert report.loaded == 0


def test_one_malformed_row_does_not_discard_the_rest(tmp_path: Path) -> None:
    """A labelling run over thousands of images will produce some bad rows."""
    path = tmp_path / "labels.jsonl"
    path.write_text(
        "\n".join(
            [
                make_outfit("good1").model_dump_json(),
                '{"id": "bad", "silhouette": "not-a-real-silhouette"}',
                "{ this is not json",
                make_outfit("good2").model_dump_json(),
            ]
        ),
        encoding="utf-8",
    )
    items, report = OutfitRepository(path).load()
    assert {i.id for i in items} == {"good1", "good2"}
    assert report.skipped == 2
    assert not report.ok


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(f"\n{make_outfit('o1').model_dump_json()}\n\n", encoding="utf-8")
    items, report = OutfitRepository(path).load()
    assert len(items) == 1
    assert report.ok


def test_save_does_not_leave_a_temp_file_behind(tmp_path: Path) -> None:
    repo = OutfitRepository(tmp_path / "labels.jsonl")
    repo.save([make_outfit("o1")])
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_replaces_atomically(tmp_path: Path) -> None:
    """An interrupted rewrite must not truncate a corpus that cost hours of quota."""
    path = tmp_path / "labels.jsonl"
    repo = OutfitRepository(path)
    repo.save([make_outfit("o1"), make_outfit("o2")])
    repo.save([make_outfit("o3")])
    items, _ = repo.load()
    assert [i.id for i in items] == ["o3"]


def test_append_adds_without_rewriting(tmp_path: Path) -> None:
    """Incremental labelling must be interruptible without losing prior work."""
    repo = OutfitRepository(tmp_path / "labels.jsonl")
    repo.append(make_outfit("o1"))
    repo.append(make_outfit("o2"))
    items, report = repo.load()
    assert report.ok
    assert [i.id for i in items] == ["o1", "o2"]


def test_labelled_ids_lets_a_resumed_run_skip_work(tmp_path: Path) -> None:
    repo = OutfitRepository(tmp_path / "labels.jsonl")
    repo.append(make_outfit("o1"))
    repo.append(make_outfit("o2"))
    assert repo.labelled_ids() == {"o1", "o2"}


def test_labelled_ids_counts_malformed_rows_as_attempted(tmp_path: Path) -> None:
    """Re-labelling a row that already failed would burn quota on every resume."""
    path = tmp_path / "labels.jsonl"
    path.write_text('{"id": "attempted", "garment_type": 12345}\n', encoding="utf-8")
    assert OutfitRepository(path).labelled_ids() == {"attempted"}


def test_celebrity_repository_round_trip(tmp_path: Path) -> None:
    repo = CelebrityRepository(tmp_path / "celebs.jsonl")
    profile = CelebrityProfile(
        id="Q123",
        name="Example Person",
        region="indian",
        shape=BodyShape.HOURGLASS,
        build=Build.SLIM,
        height_band=HeightBand.AVERAGE,
        style_tags=("ethnic", "glam"),
    )
    repo.save([profile])
    assert repo.index() == {"Q123": profile}


def test_iter_jsonl_skips_corrupt_rows(tmp_path: Path) -> None:
    path = tmp_path / "roster.jsonl"
    path.write_text('{"name": "A"}\nnot json\n{"name": "B"}\n', encoding="utf-8")
    assert [r["name"] for r in iter_jsonl(path)] == ["A", "B"]


def test_iter_jsonl_on_missing_file_yields_nothing(tmp_path: Path) -> None:
    assert list(iter_jsonl(tmp_path / "absent.jsonl")) == []


def test_as_str_tuple_handles_the_shapes_a_vlm_actually_returns() -> None:
    """Payload fields are untyped; a bare string must not be split into characters."""
    from fashion.core.models import as_str_tuple

    assert as_str_tuple(None) == ()
    assert as_str_tuple("crimson") == ("crimson",)
    assert as_str_tuple(["crimson", "gold"]) == ("crimson", "gold")
    assert as_str_tuple(("a",)) == ("a",)
    assert as_str_tuple([1, 2]) == ("1", "2")
    assert as_str_tuple(42) == ()
