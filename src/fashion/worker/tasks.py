"""RQ worker tasks.

The API's in-process BackgroundTasks is fine for a single node. This is the path for
separate API and worker processes, where job state has to live in Redis because the two
no longer share memory.

Providers are built once per worker process and reused. Rebuilding CLIP weights per job
would dominate the runtime of every job that follows the first.

Run with:
    uv run rq worker --url redis://localhost:6379/0 fashion
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fashion.adapters.jobs_store import RedisJobStore
from fashion.config import Settings, load_settings
from fashion.core.models import BodyShape, UserQuery
from fashion.factory import (
    build_embedder,
    build_generation,
    build_store,
    build_tryon,
    build_vision,
)
from fashion.pipeline.recommend import RecommendationPipeline

log = logging.getLogger(__name__)

QUEUE_NAME = "fashion"


@lru_cache(maxsize=1)
def _pipeline_for(settings: Settings) -> RecommendationPipeline:
    return RecommendationPipeline(
        vision=build_vision(settings),
        embedder=build_embedder(settings),
        store=build_store(settings),
        generator=build_generation(settings),
        tryon=build_tryon(settings),
        jobs=RedisJobStore(settings.redis_url),
    )


def run_recommendation(
    job_id: str,
    photo: bytes,
    query_json: str,
    confirmed_shape: str | None = None,
    want_generation: bool = True,
    want_tryon: bool = True,
) -> str:
    """Execute one recommendation job.

    Arguments are plain serialisable types because RQ pickles them across a process
    boundary; passing live provider objects would not survive the trip.
    """
    settings = load_settings()
    store = RedisJobStore(settings.redis_url)

    job = store.get(job_id)
    if job is None:
        # The job expired or was never written. Re-running would produce a result
        # nobody can poll for, so stop.
        log.warning("job %s is not in the store; skipping", job_id)
        return job_id

    try:
        _pipeline_for(settings).run(
            job,
            photo,
            UserQuery.model_validate_json(query_json),
            confirmed_shape=BodyShape(confirmed_shape) if confirmed_shape else None,
            want_generation=want_generation,
            want_tryon=want_tryon,
        )
    except Exception:
        log.exception("job %s crashed in the worker", job_id)
        job.fail("The pipeline failed unexpectedly.")
        store.save(job)

    return job_id


def enqueue(
    photo: bytes,
    query: UserQuery,
    *,
    settings: Settings | None = None,
    confirmed_shape: BodyShape | None = None,
    want_generation: bool = True,
    want_tryon: bool = True,
) -> str:
    """Create a job, persist it, and queue the work. Returns the job id."""
    from redis import Redis
    from rq import Queue

    from fashion.core.jobs import Job

    settings = settings or load_settings()
    store = RedisJobStore(settings.redis_url)
    job = Job()
    # Persist before enqueueing, so a fast worker cannot look for the job before the
    # API has written it.
    store.create(job)

    queue = Queue(QUEUE_NAME, connection=Redis.from_url(settings.redis_url))
    queue.enqueue(
        run_recommendation,
        job.id,
        photo,
        query.model_dump_json(),
        confirmed_shape.value if confirmed_shape else None,
        want_generation,
        want_tryon,
        job_timeout=600,
    )
    return job.id
