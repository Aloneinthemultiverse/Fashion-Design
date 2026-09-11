"""API contract, driven through FastAPI's real test client.

Uses fake providers, so it needs no key, no GPU and no network. The contract under test
is the one ADR 0002 commits to: submit returns immediately, progress is pollable, and a
missing optional backend degrades the response rather than failing the request.

BackgroundTasks run synchronously inside TestClient, so a job is already complete by the
time submit returns here. That makes the assertions deterministic; it does not change the
contract, because the client polls either way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from fashion.adapters.embed_fake import FakeEmbedder
from fashion.adapters.gen_null import NullGenerationProvider, NullTryOnProvider
from fashion.adapters.jobs_store import InMemoryJobStore
from fashion.adapters.store_memory import InMemoryVectorStore
from fashion.adapters.vision_fake import FakeVisionModel
from fashion.api.app import Deps, create_app
from fashion.config import Settings
from fashion.core.feedback import FeedbackLog
from fashion.core.models import BodyShape, Build, CelebrityProfile, HeightBand
from fashion.core.ratelimit import (
    DailyQuota,
    SlidingWindowLimiter,
    TtlCache,
)
from fashion.pipeline.index import IndexBuilder
from tests.conftest import make_outfit

PNG = b"\x89PNG\r\n\x1a\n" + b"fake-image-payload" * 8


class StubDeps(Deps):
    """Deps wired entirely from fakes, with a small seeded corpus."""

    def __init__(self, tmp_path: Path) -> None:
        self.settings = Settings(
            vision_provider="fake", embed_provider="fake", store_provider="memory"
        )
        self.jobs = InMemoryJobStore()
        self.vision = FakeVisionModel()
        self.embedder = FakeEmbedder()
        self.store = InMemoryVectorStore()
        self.generator = NullGenerationProvider()
        self.tryon = NullTryOnProvider()
        self.executor = None  # type: ignore[assignment]
        # Generous limits: these tests exercise the contract, not the throttle. The
        # limiter has its own unit tests where the bounds are the subject.
        self.per_minute = SlidingWindowLimiter(1000, 60)
        self.per_hour = SlidingWindowLimiter(1000, 3600)
        self.quota = DailyQuota(1000)
        self.results = TtlCache(max_entries=64)
        self.feedback = FeedbackLog(tmp_path / "feedback.jsonl")

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
        IndexBuilder(self.embedder, self.store).build(items, profiles)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(StubDeps(tmp_path)))


def submit(client: TestClient, **form: Any) -> dict[str, Any]:
    response = client.post(
        "/recommendations",
        files={"photo": ("me.jpg", PNG, "image/jpeg")},
        data={"text": "festive ethnic wear", **form},
    )
    assert response.status_code == 202, response.text
    body: dict[str, Any] = response.json()
    return body


def test_health_reports_degraded_backends_honestly(client: TestClient) -> None:
    """Operators need to see that generation is off, not guess from missing images."""
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["corpus_size"] == len(BodyShape)
    assert body["generation_available"] is False
    assert body["tryon_available"] is False


def test_submit_returns_202_with_a_poll_url(client: TestClient) -> None:
    body = submit(client)
    assert body["job_id"]
    assert body["poll_url"] == f"/recommendations/{body['job_id']}"


def test_polling_returns_stage_by_stage_progress(client: TestClient) -> None:
    job_id = submit(client)["job_id"]
    body = client.get(f"/recommendations/{job_id}").json()

    assert body["state"] == "done"
    assert body["progress"] == 1.0
    assert [s["name"] for s in body["stages"]] == ["analyze", "retrieve", "generate", "tryon"]


def test_analyze_stage_publishes_body_metrics(client: TestClient) -> None:
    job_id = submit(client)["job_id"]
    stages = {s["name"]: s for s in client.get(f"/recommendations/{job_id}").json()["stages"]}

    analyze = stages["analyze"]
    assert analyze["state"] == "done"
    assert analyze["result"]["shape"] in {s.value for s in BodyShape}
    assert "needs_confirmation" in analyze["result"]


def test_retrieve_stage_returns_recommendations_with_rationale(client: TestClient) -> None:
    job_id = submit(client)["job_id"]
    stages = {s["name"]: s for s in client.get(f"/recommendations/{job_id}").json()["stages"]}

    retrieve = stages["retrieve"]
    assert retrieve["state"] == "done"
    recommendations = retrieve["result"]["recommendations"]
    assert recommendations
    top = recommendations[0]
    assert top["rationale"]
    assert top["license"]  # provenance survives all the way to the response
    assert top["source"]


def test_missing_backends_are_skipped_not_failed(client: TestClient) -> None:
    """A missing GPU is a configuration state the UI explains, not an outage."""
    job_id = submit(client)["job_id"]
    stages = {s["name"]: s for s in client.get(f"/recommendations/{job_id}").json()["stages"]}

    assert stages["generate"]["state"] == "skipped"
    assert stages["tryon"]["state"] == "skipped"
    assert "side-by-side" in stages["tryon"]["detail"]
    # The job as a whole still succeeds.
    assert client.get(f"/recommendations/{job_id}").json()["state"] == "done"


def test_user_confirmed_shape_overrides_the_model(client: TestClient) -> None:
    """The user's correction is evidence; the model's guess is only a prior."""
    job_id = submit(client, confirmed_shape="hourglass")["job_id"]
    stages = {s["name"]: s for s in client.get(f"/recommendations/{job_id}").json()["stages"]}

    analyze = stages["analyze"]["result"]
    assert analyze["shape"] == "hourglass"
    assert analyze["user_confirmed"] is True
    assert analyze["needs_confirmation"] is False


def test_confirmed_shape_constrains_what_is_returned(client: TestClient) -> None:
    job_id = submit(client, confirmed_shape="pear", top_k="5")["job_id"]
    stages = {s["name"]: s for s in client.get(f"/recommendations/{job_id}").json()["stages"]}
    result = stages["retrieve"]["result"]
    assert not result["relaxed"]
    assert all(r["outfit_id"] == "o-pear" for r in result["recommendations"])


def test_opting_out_of_generation_is_reported_distinctly(client: TestClient) -> None:
    job_id = submit(client, want_generation="false")["job_id"]
    stages = {s["name"]: s for s in client.get(f"/recommendations/{job_id}").json()["stages"]}
    assert stages["generate"]["detail"] == "Generation was not requested."


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/recommendations/does-not-exist").status_code == 404


def test_non_image_upload_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/recommendations",
        files={"photo": ("notes.txt", b"hello", "text/plain")},
        data={"text": "x"},
    )
    assert response.status_code == 415


def test_empty_upload_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/recommendations",
        files={"photo": ("me.jpg", b"", "image/jpeg")},
        data={"text": "x"},
    )
    assert response.status_code == 400


def test_invalid_enum_value_is_422_not_500(client: TestClient) -> None:
    response = client.post(
        "/recommendations",
        files={"photo": ("me.jpg", PNG, "image/jpeg")},
        data={"text": "x", "culture": "klingon"},
    )
    assert response.status_code == 422


def test_top_k_is_clamped_rather_than_rejected(client: TestClient) -> None:
    """An absurd top_k should not fail the request, just bound the work."""
    job_id = submit(client, top_k="9999")["job_id"]
    stages = {s["name"]: s for s in client.get(f"/recommendations/{job_id}").json()["stages"]}
    assert len(stages["retrieve"]["result"]["recommendations"]) <= 50
