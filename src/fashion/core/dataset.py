"""Dataset persistence.

JSONL rather than a database: the corpus is append-mostly, needs to be diffable in git
so that label corrections are reviewable, and must stay readable without running
anything. At a few thousand rows the whole file loads in well under a second.

Loading is deliberately tolerant of bad rows -- a labelling run over thousands of images
will produce some malformed output, and losing the entire corpus because one row has an
unknown enum value would be worse than skipping it and reporting the count.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from fashion.core.models import CelebrityProfile, OutfitItem

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoadReport:
    """What happened during a load. Surfaced so silent data loss is impossible."""

    loaded: int
    skipped: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.skipped == 0


def _write_jsonl(path: Path, rows: Iterable[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file and replace, so an interrupted run cannot truncate an
    # existing corpus that took hours of API quota to label.
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row + "\n")
            count += 1
    tmp.replace(path)
    return count


class OutfitRepository:
    """Reads and writes the labelled outfit corpus."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> tuple[list[OutfitItem], LoadReport]:
        if not self._path.exists():
            return [], LoadReport(0, 0, ())

        items: list[OutfitItem] = []
        errors: list[str] = []
        with self._path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(OutfitItem.model_validate_json(line))
                except (ValidationError, ValueError) as exc:
                    errors.append(f"line {lineno}: {type(exc).__name__}")
                    log.warning("skipping malformed outfit at line %d", lineno)

        return items, LoadReport(len(items), len(errors), tuple(errors))

    def save(self, items: Iterable[OutfitItem]) -> int:
        return _write_jsonl(self._path, (i.model_dump_json() for i in items))

    def append(self, item: OutfitItem) -> None:
        """Append one item, for incremental labelling that can be interrupted safely."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(item.model_dump_json() + "\n")

    def labelled_ids(self) -> set[str]:
        """IDs already in the corpus, so a resumed labelling run skips them.

        Reads ids only, without full validation, so that a malformed row still counts as
        "already attempted" and does not get re-labelled on every resume -- re-labelling
        costs VLM quota, which is the scarce resource.
        """
        if not self._path.exists():
            return set()
        ids: set[str] = set()
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line).get("id")
                except json.JSONDecodeError:
                    continue
                if isinstance(value, str):
                    ids.add(value)
        return ids


class CelebrityRepository:
    """Reads and writes celebrity profiles."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> tuple[list[CelebrityProfile], LoadReport]:
        if not self._path.exists():
            return [], LoadReport(0, 0, ())

        profiles: list[CelebrityProfile] = []
        errors: list[str] = []
        with self._path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    profiles.append(CelebrityProfile.model_validate_json(line))
                except (ValidationError, ValueError) as exc:
                    errors.append(f"line {lineno}: {type(exc).__name__}")

        return profiles, LoadReport(len(profiles), len(errors), tuple(errors))

    def save(self, profiles: Iterable[CelebrityProfile]) -> int:
        return _write_jsonl(self._path, (p.model_dump_json() for p in profiles))

    def index(self) -> dict[str, CelebrityProfile]:
        profiles, _ = self.load()
        return {p.id: p for p in profiles}


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    """Read raw JSONL rows, for files whose schema is not an OutfitItem.

    Used for the Wikidata roster, which is a pre-labelling staging file.
    """
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


class AlreadyRunningError(RuntimeError):
    """Raised when another run holds the lock."""


class RunLock:
    """Exclusive lock for a long ingest run.

    Two concurrent runs over the same corpus corrupt it in a way that is easy to miss:
    both append outfits, so the file grows with duplicates, and both hold their own
    in-memory profile dict which they alternately write over the whole profiles file.
    The result is an outfit count that looks healthy beside a profile count that
    silently lags. That happened; this prevents it.

    Uses exclusive file creation, which is atomic on every platform, rather than a
    check-then-write that races.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._held = False

    def __enter__(self) -> RunLock:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._path.open("x", encoding="utf-8") as fh:
                fh.write(f"pid={os.getpid()} started={datetime.now(UTC).isoformat()}")
        except FileExistsError:
            raise AlreadyRunningError(
                f"another run holds {self._path}. Stop it first, or delete the file if "
                "no run is active."
            ) from None
        self._held = True
        return self

    def __exit__(self, *exc: object) -> None:
        if self._held:
            self._path.unlink(missing_ok=True)
            self._held = False
