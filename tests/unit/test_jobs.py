"""Job model, stores, and pipeline stage semantics.

The distinction these tests defend is SKIPPED vs FAILED. Collapsing them would make a
deliberately-unconfigured GPU indistinguishable from an outage, which is the difference
between the UI saying "showing real outfits instead" and "something went wrong".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fashion.adapters.embed_fake import FakeEmbedder
from fashion.adapters.gen_null import NullGenerationProvider, NullTryOnProvider
from fashion.adapters.jobs_store import InMemoryJobStore
from fashion.adapters.store_memory import InMemoryVectorStore
from fashion.adapters.vision_fake import FakeVisionModel
from fashion.core.jobs import Job, JobState, JobStore, StageName, StageState
from fashion.core.models import (
    BodyShape,
    Build,
    CelebrityProfile,
    HeightBand,
    UserQuery,
)
from fashion.pipeline.index import IndexBuilder
from fashion.pipeline.recommend import RecommendationPipeline
from tests.conftest import make_outfit


def test_new_job_has_every_stage_pending() -> None:
    job = Job()
    assert job.state is JobState.QUEUED
    assert [s.name for s in job.stages] == list(StageName)
    assert all(s.state is StageState.PENDING for s in job.stages)
    assert job.progress == 0.0


def test_progress_counts_settled_stages() -> None:
    """Skipped stages are settled: a skipped try-on must not stall progress at 75%."""
    job = Job()
    job.finish_stage(StageName.ANALYZE)
    job.finish_stage(StageName.RETRIEVE)
    job.finish_stage(StageName.GENERATE, state=StageState.SKIPPED)
    job.finish_stage(StageName.TRYON, state=StageState.FAILED)
    assert job.progress == 1.0


def test_stage_records_its_duration() -> None:
    job = Job()
    job.start_stage(StageName.ANALYZE)
    job.finish_stage(StageName.ANALYZE)
    duration = job.stage(StageName.ANALYZE).duration_seconds
    assert duration is not None and duration >= 0


def test_failing_a_job_is_terminal() -> None:
    job = Job()
    job.fail("nope")
    assert job.is_terminal
    assert job.error == "nope"


def test_in_memory_store_hands_back_copies() -> None:
    """A caller mutating a retrieved job must not corrupt stored state."""
    store = InMemoryJobStore()
    job = Job()
    store.create(job)

    retrieved = store.get(job.id)
    assert retrieved is not None
    retrieved.fail("mutated by the caller")

    fresh = store.get(job.id)
    assert fresh is not None
    assert fresh.state is JobState.QUEUED


def test_in_memory_store_satisfies_the_port() -> None:
    assert isinstance(InMemoryJobStore(), JobStore)


def test_unknown_job_reads_as_none() -> None:
    assert InMemoryJobStore().get("nope") is None


# -- pipeline ---------------------------------------------------------------------


@pytest.fixture
def pipeline(tmp_path: Path) -> tuple[RecommendationPipeline, InMemoryJobStore]:
    embedder = FakeEmbedder()
    store = InMemoryVectorStore()
    items = []
    profiles = {}
    for shape in BodyShape:
        image = tmp_path / f"{shape.value}.jpg"
        image.write_bytes(f"outfit-{shape.value}".encode())
        items.append(
            make_outfit(f"o-{shape.value}", celebrity_id=f"c-{shape.value}").model_copy(
                update={"image_path": str(image)}
            )
        )
        profiles[f"c-{shape.value}"] = CelebrityProfile(
            id=f"c-{shape.value}",
            name=f"Celeb {shape.value}",
            region="indian",
            shape=shape,
            build=Build.SLIM,
            height_band=HeightBand.AVERAGE,
        )
    IndexBuilder(embedder, store).build(items, profiles)

    jobs = InMemoryJobStore()
    return (
        RecommendationPipeline(
            vision=FakeVisionModel(),
            embedder=embedder,
            store=store,
            generator=NullGenerationProvider(),
            tryon=NullTryOnProvider(),
            jobs=jobs,
        ),
        jobs,
    )


def test_pipeline_completes_with_no_optional_backends(
    pipeline: tuple[RecommendationPipeline, InMemoryJobStore],
) -> None:
    """The whole point of the null providers: a zero-GPU deployment still succeeds."""
    engine, _ = pipeline
    job = engine.run(Job(), b"a-photo", UserQuery(text="ethnic wear"))

    assert job.state is JobState.DONE
    assert job.stage(StageName.ANALYZE).state is StageState.DONE
    assert job.stage(StageName.RETRIEVE).state is StageState.DONE
    assert job.stage(StageName.GENERATE).state is StageState.SKIPPED
    assert job.stage(StageName.TRYON).state is StageState.SKIPPED


def test_skipped_stages_explain_themselves(
    pipeline: tuple[RecommendationPipeline, InMemoryJobStore],
) -> None:
    engine, _ = pipeline
    job = engine.run(Job(), b"a-photo", UserQuery(text="ethnic wear"))
    assert "generation backend" in job.stage(StageName.GENERATE).detail
    assert "side-by-side" in job.stage(StageName.TRYON).detail


def test_each_stage_is_published_as_it_finishes(
    pipeline: tuple[RecommendationPipeline, InMemoryJobStore],
) -> None:
    """Partial results are the reason the job model exists at all."""
    engine, jobs = pipeline
    job = Job()
    jobs.create(job)
    engine.run(job, b"a-photo", UserQuery(text="ethnic wear"))

    stored = jobs.get(job.id)
    assert stored is not None
    assert stored.stage(StageName.ANALYZE).result is not None
    assert stored.stage(StageName.RETRIEVE).result is not None


def test_a_failing_vlm_fails_the_job(
    pipeline: tuple[RecommendationPipeline, InMemoryJobStore],
) -> None:
    """Body analysis is load-bearing: without it there is no useful answer."""
    engine, _ = pipeline

    class Broken(FakeVisionModel):
        def analyze_body(self, image: bytes):  # type: ignore[no-untyped-def]
            raise RuntimeError("quota exhausted")

    engine.vision = Broken()
    job = engine.run(Job(), b"a-photo", UserQuery(text="x"))

    assert job.state is JobState.FAILED
    assert job.stage(StageName.ANALYZE).state is StageState.FAILED
    assert job.stage(StageName.RETRIEVE).state is StageState.PENDING


def test_a_failing_generator_does_not_fail_the_job(
    pipeline: tuple[RecommendationPipeline, InMemoryJobStore],
) -> None:
    """Generation is optional, so its failure must cost only the preview image."""
    engine, _ = pipeline

    class Exploding:
        available = True
        supports_references = True

        def generate(self, prompt: str, *, references=(), seed=None):  # type: ignore[no-untyped-def]
            raise RuntimeError("colab session died")

    engine.generator = Exploding()  # type: ignore[assignment]
    job = engine.run(Job(), b"a-photo", UserQuery(text="x"))

    assert job.state is JobState.DONE
    assert job.stage(StageName.GENERATE).state is StageState.FAILED
    assert job.stage(StageName.RETRIEVE).state is StageState.DONE


def test_confirmed_shape_overrides_inference_and_raises_confidence(
    pipeline: tuple[RecommendationPipeline, InMemoryJobStore],
) -> None:
    engine, _ = pipeline
    job = engine.run(
        Job(), b"a-photo", UserQuery(text="x"), confirmed_shape=BodyShape.HOURGLASS
    )
    result = job.stage(StageName.ANALYZE).result
    assert result is not None
    assert result["shape"] == "hourglass"
    assert result["user_confirmed"] is True
    assert result["needs_confirmation"] is False


def test_opting_out_is_distinguishable_from_being_unavailable(
    pipeline: tuple[RecommendationPipeline, InMemoryJobStore],
) -> None:
    engine, _ = pipeline
    job = engine.run(Job(), b"a-photo", UserQuery(text="x"), want_tryon=False)
    assert job.stage(StageName.TRYON).detail == "Try-on was not requested."


def test_working_tryon_backend_produces_a_result(
    pipeline: tuple[RecommendationPipeline, InMemoryJobStore],
) -> None:
    engine, _ = pipeline

    class Working:
        available = True

        def try_on(self, person: bytes, garment: bytes) -> bytes:
            return b"rendered-image"

    engine.tryon = Working()  # type: ignore[assignment]
    job = engine.run(Job(), b"a-photo", UserQuery(text="x"))

    stage = job.stage(StageName.TRYON)
    assert stage.state is StageState.DONE
    assert stage.result is not None
    assert stage.result["has_image"] is True
