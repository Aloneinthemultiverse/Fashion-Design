"""Job and stage model.

ADR 0002: the pipeline takes 45-90 seconds, which no synchronous HTTP request survives.
Work is submitted, and progress is polled.

Stages publish results as they finish rather than at the end. The user sees body analysis
at ~2s and recommendations at ~10s instead of a 90-second spinner, so perceived latency
improves even though total wall-clock does not. It also makes partial failure
expressible: try-on can fail while recommendations succeed, which is exactly the
degradation the architecture calls for.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class StageName(StrEnum):
    ANALYZE = "analyze"
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    TRYON = "tryon"


class StageState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class Stage(BaseModel):
    """One pipeline step.

    `SKIPPED` is distinct from `FAILED` on purpose: no GPU configured is a normal
    outcome the UI explains, while a crash is not. Collapsing them would make an outage
    look like a configuration choice.
    """

    model_config = ConfigDict(extra="forbid")

    name: StageName
    state: StageState = StageState.PENDING
    detail: str = ""
    result: dict[str, Any] | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    state: JobState = JobState.QUEUED
    stages: list[Stage] = Field(default_factory=lambda: [Stage(name=n) for n in StageName])
    error: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def stage(self, name: StageName) -> Stage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)

    def start_stage(self, name: StageName) -> None:
        stage = self.stage(name)
        stage.state = StageState.RUNNING
        stage.started_at = datetime.now(UTC)
        self.state = JobState.RUNNING
        self.updated_at = datetime.now(UTC)

    def finish_stage(
        self,
        name: StageName,
        *,
        result: dict[str, Any] | None = None,
        state: StageState = StageState.DONE,
        detail: str = "",
    ) -> None:
        stage = self.stage(name)
        stage.state = state
        stage.result = result
        stage.detail = detail
        stage.finished_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def fail(self, message: str) -> None:
        self.state = JobState.FAILED
        self.error = message
        self.updated_at = datetime.now(UTC)

    def complete(self) -> None:
        self.state = JobState.DONE
        self.updated_at = datetime.now(UTC)

    @property
    def is_terminal(self) -> bool:
        return self.state in (JobState.DONE, JobState.FAILED)

    @property
    def progress(self) -> float:
        """Fraction of stages that have reached a terminal state."""
        settled = sum(
            1
            for s in self.stages
            if s.state in (StageState.DONE, StageState.FAILED, StageState.SKIPPED)
        )
        return round(settled / len(self.stages), 3) if self.stages else 0.0


@runtime_checkable
class JobStore(Protocol):
    """Where job state lives between the worker that writes it and the API that reads it."""

    def create(self, job: Job) -> None: ...

    def get(self, job_id: str) -> Job | None: ...

    def save(self, job: Job) -> None: ...
