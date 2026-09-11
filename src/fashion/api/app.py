"""HTTP API.

Submit-and-poll, per ADR 0002:

    POST /recommendations      -> 202 Accepted, {job_id}
    GET  /recommendations/{id} -> stage-by-stage progress and partial results

Photos are held in memory for the life of the job and never written to disk. The privacy
note in ADR 0001 promises exactly that, and the cheapest way to keep the promise is to
have no code that could break it.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from fashion.adapters.jobs_store import InMemoryJobStore
from fashion.config import Settings, load_settings
from fashion.core.jobs import Job, JobStore
from fashion.core.models import BodyShape, Culture, Occasion, UserQuery
from fashion.factory import (
    build_embedder,
    build_generation,
    build_store,
    build_tryon,
    build_vision,
    load_corpus,
)
from fashion.pipeline.recommend import RecommendationPipeline

log = logging.getLogger(__name__)

MAX_PHOTO_BYTES = 12 * 1024 * 1024
ALLOWED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})

# Bounded on purpose. The architecture caps concurrent VLM work, and an unbounded pool
# would let a burst of uploads exhaust the daily quota in seconds.
MAX_CONCURRENT_JOBS = 4


class Deps:
    """Process-wide singletons.

    Built once because the embedder loads model weights and the store holds a
    connection; rebuilding either per request would dominate latency.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.jobs: JobStore = InMemoryJobStore()
        self.vision = build_vision(self.settings)
        self.embedder = build_embedder(self.settings)
        self.store = build_store(self.settings)
        self.generator = build_generation(self.settings)
        self.tryon = build_tryon(self.settings)
        self.executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)
        # The in-memory store is process-local and starts empty; without this the API
        # serves an empty corpus and every request returns no recommendations.
        load_corpus(self.embedder, self.store, self.settings)

    def pipeline(self) -> RecommendationPipeline:
        return RecommendationPipeline(
            vision=self.vision,
            embedder=self.embedder,
            store=self.store,
            generator=self.generator,
            tryon=self.tryon,
            jobs=self.jobs,
        )


class SubmitResponse(BaseModel):
    job_id: str
    state: str
    poll_url: str


class JobResponse(BaseModel):
    id: str
    state: str
    progress: float
    error: str = ""
    stages: list[dict[str, Any]]


def create_app(deps: Deps | None = None) -> FastAPI:
    app = FastAPI(
        title="Cross-Cultural Fashion Recommender",
        version="0.1.0",
        description="Body-geometry outfit recommendations across Western and Indian wardrobes.",
    )
    # Closed over rather than injected with Depends. Under `from __future__ import
    # annotations` FastAPI resolves parameter annotations from strings, and an
    # Annotated[...] dependency declared inside this factory is read as a query
    # parameter instead of a dependency. The container is already a per-app singleton,
    # so closing over it is both simpler and correct.
    d = deps or Deps()
    app.state.deps = d

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Reports which optional backends are live, so degraded mode is visible."""
        return {
            "status": "ok",
            "vision": d.settings.vision_provider,
            "embedder": d.settings.embed_provider,
            "store": d.settings.store_provider,
            "corpus_size": d.store.count(),
            "generation_available": d.generator.available,
            "tryon_available": d.tryon.available,
        }

    @app.post("/recommendations", status_code=202, response_model=SubmitResponse)
    async def submit(
        background: BackgroundTasks,
        photo: Annotated[UploadFile, File(description="Full-body photo")],
        text: Annotated[str, Form()] = "",
        culture: Annotated[str | None, Form()] = None,
        occasion: Annotated[str | None, Form()] = None,
        celebrity_name: Annotated[str | None, Form()] = None,
        confirmed_shape: Annotated[str | None, Form()] = None,
        top_k: Annotated[int, Form()] = 10,
        want_generation: Annotated[bool, Form()] = True,
        want_tryon: Annotated[bool, Form()] = True,
    ) -> SubmitResponse:
        if photo.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                415, f"Unsupported image type {photo.content_type!r}. Use JPEG, PNG or WebP."
            )

        data = await photo.read()
        if not data:
            raise HTTPException(400, "The uploaded photo is empty.")
        if len(data) > MAX_PHOTO_BYTES:
            raise HTTPException(413, f"Photo exceeds {MAX_PHOTO_BYTES // (1024 * 1024)}MB.")

        try:
            query = UserQuery(
                text=text,
                culture=Culture(culture) if culture else None,
                occasion=Occasion(occasion) if occasion else None,
                celebrity_name=celebrity_name or None,
                top_k=max(1, min(top_k, 50)),
            )
            shape = BodyShape(confirmed_shape) if confirmed_shape else None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

        job = Job()
        d.jobs.create(job)

        # FastAPI's BackgroundTasks runs after the response is sent, so the client gets
        # its job id immediately rather than waiting on the pipeline.
        background.add_task(_run_job, d, job, data, query, shape, want_generation, want_tryon)

        return SubmitResponse(
            job_id=job.id, state=job.state.value, poll_url=f"/recommendations/{job.id}"
        )

    @app.get("/recommendations/{job_id}", response_model=JobResponse)
    def poll(job_id: str) -> JobResponse:
        job = d.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown or expired job.")
        return JobResponse(
            id=job.id,
            state=job.state.value,
            progress=job.progress,
            error=job.error,
            stages=[s.model_dump(mode="json") for s in job.stages],
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Any, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Internal error."})

    return app


def _run_job(
    deps: Deps,
    job: Job,
    photo: bytes,
    query: UserQuery,
    shape: BodyShape | None,
    want_generation: bool,
    want_tryon: bool,
) -> None:
    try:
        deps.pipeline().run(
            job,
            photo,
            query,
            confirmed_shape=shape,
            want_generation=want_generation,
            want_tryon=want_tryon,
        )
    except Exception:
        # The pipeline handles its own stage failures; this catches anything above it
        # so a job can never be left stuck in RUNNING with no explanation.
        log.exception("job %s crashed", job.id)
        job.fail("The pipeline failed unexpectedly.")
        deps.jobs.save(job)


app = create_app()
