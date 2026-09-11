"""Job stores.

Two implementations of the same port. The in-memory one is correct only when the API and
the worker share a process, which is true in the Streamlit and single-process dev setups
and false in production -- hence Redis.

Jobs expire. A completed job is interesting for as long as someone might refresh the
page, not forever, and without a TTL the store grows without bound.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from fashion.core.jobs import Job

log = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 3600


@dataclass
class InMemoryJobStore:
    """Process-local job store.

    Guarded by a lock because the API reads while a worker thread writes; without it a
    poll can observe a half-updated job.
    """

    _jobs: dict[str, Job] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def create(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job.model_copy(deep=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            # Hand back a copy so a caller mutating the result cannot corrupt stored
            # state, which would otherwise make a failed job silently look healthy.
            return job.model_copy(deep=True) if job else None

    def save(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job.model_copy(deep=True)

    def count(self) -> int:
        with self._lock:
            return len(self._jobs)


class RedisJobStore:
    """Redis-backed job store, for when the API and worker are separate processes."""

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        client: Any = None,
        prefix: str = "fashion:job:",
    ) -> None:
        self._ttl = ttl_seconds
        self._prefix = prefix
        self._client = client if client is not None else self._build_client(url)

    @staticmethod
    def _build_client(url: str) -> Any:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("redis is not installed. Run `uv sync --extra api`.") from exc
        return redis.Redis.from_url(url, decode_responses=True)

    def _key(self, job_id: str) -> str:
        return f"{self._prefix}{job_id}"

    def create(self, job: Job) -> None:
        self.save(job)

    def get(self, job_id: str) -> Job | None:
        raw = self._client.get(self._key(job_id))
        if not raw:
            return None
        try:
            return Job.model_validate_json(raw)
        except ValueError:
            # A job written by an older schema version should read as absent rather
            # than crash a poll.
            log.warning("discarding unreadable job %s", job_id)
            return None

    def save(self, job: Job) -> None:
        # Refresh the TTL on every write so a long-running job cannot expire mid-flight.
        self._client.set(self._key(job.id), job.model_dump_json(), ex=self._ttl)
