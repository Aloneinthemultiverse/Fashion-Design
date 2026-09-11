"""Feedback capture.

Thumbs up/down on a recommendation, appended to JSONL. Deliberately the simplest thing
that stores the signal durably: the value here is in *having* the data when there is
enough of it to learn from, and any storage choice made now would be premature.

What is recorded is chosen so the feedback is analysable later. A thumbs-down is only
interpretable alongside the body shape it was given for and whether that shape was
user-confirmed -- a rejection on a shape the model guessed wrong says nothing about the
recommendation, and without the flag the two cases are indistinguishable in the data.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

log = logging.getLogger(__name__)


class Verdict(StrEnum):
    UP = "up"
    DOWN = "down"


@dataclass(frozen=True, slots=True)
class FeedbackEvent:
    outfit_id: str
    verdict: Verdict
    body_shape: str
    shape_was_confirmed: bool
    query_text: str = ""
    matched_on: tuple[str, ...] = ()
    relaxed: bool = False
    note: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class FeedbackLog:
    """Append-only feedback store."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def record(self, event: FeedbackEvent) -> None:
        """Append one event.

        Failures are logged and swallowed: losing a thumbs-up is a trivial loss, while
        failing a user's request because the feedback file is read-only is not.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            row = asdict(event)
            row["matched_on"] = list(event.matched_on)
            with self._lock, self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            log.warning("could not record feedback for %s", event.outfit_id, exc_info=True)

    def read(self) -> Iterator[FeedbackEvent]:
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    row["matched_on"] = tuple(row.get("matched_on", ()))
                    row["verdict"] = Verdict(row["verdict"])
                    yield FeedbackEvent(**row)
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

    def summary(self) -> dict[str, object]:
        """Aggregate signal, for the metrics endpoint.

        Confirmed and unconfirmed feedback are counted separately because mixing them
        would let bad body-shape inference masquerade as bad recommendations.
        """
        up = down = confirmed_up = confirmed_down = 0
        per_shape: dict[str, dict[str, int]] = {}

        for event in self.read():
            bucket = per_shape.setdefault(event.body_shape, {"up": 0, "down": 0})
            bucket[event.verdict.value] += 1
            if event.verdict is Verdict.UP:
                up += 1
                confirmed_up += int(event.shape_was_confirmed)
            else:
                down += 1
                confirmed_down += int(event.shape_was_confirmed)

        total = up + down
        confirmed_total = confirmed_up + confirmed_down
        return {
            "total": total,
            "up": up,
            "down": down,
            "satisfaction": round(up / total, 3) if total else None,
            # The number that actually reflects recommendation quality.
            "satisfaction_confirmed_shape": (
                round(confirmed_up / confirmed_total, 3) if confirmed_total else None
            ),
            "by_shape": per_shape,
        }
