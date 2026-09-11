"""End-to-end recommendation pipeline.

Runs the four stages and publishes each result as it lands. The ordering is a dependency
chain -- retrieval needs body metrics, try-on needs a chosen outfit -- but the failure
semantics are deliberately not a chain: a stage that cannot run marks itself SKIPPED and
the pipeline continues, so a missing GPU costs the user a preview image rather than the
whole answer.

Only `analyze` and `retrieve` are load-bearing. If either fails the job fails, because
there is no useful answer without them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from fashion.core.jobs import Job, JobStore, StageName, StageState
from fashion.core.models import (
    BodyMetrics,
    BodyShape,
    CelebrityProfile,
    Recommendation,
    UserQuery,
)
from fashion.core.reference import BodyReferenceResolver
from fashion.pipeline.imagerag import ImageRagGenerator, ImageRagRefiner
from fashion.pipeline.retrieve import Retriever
from fashion.ports.embedder import Embedder
from fashion.ports.generation import GenerationProvider
from fashion.ports.tryon import TryOnProvider
from fashion.ports.vectorstore import Filter, VectorStore
from fashion.ports.vision import VisionModel

log = logging.getLogger(__name__)


def as_payload(rec: Recommendation) -> dict[str, object]:
    """Flatten a recommendation for the job result.

    Shared by the retrieve and refine stages so the two cannot drift into emitting
    differently-shaped records for the same kind of thing.
    """
    return {
        "outfit_id": rec.outfit.id,
        "celebrity_id": rec.outfit.celebrity_id,
        "garment_type": rec.outfit.garment_type,
        "silhouette": rec.outfit.silhouette.value,
        "neckline": rec.outfit.neckline.value,
        "culture": rec.outfit.culture.value,
        "occasion": rec.outfit.occasion.value,
        "colors": list(rec.outfit.colors),
        "image_path": rec.outfit.image_path,
        "score": rec.score,
        "rationale": rec.rationale,
        "adjustments": list(rec.adjustments),
        "matched_on": list(rec.matched_on),
        "source": rec.outfit.source,
        "license": rec.outfit.license,
    }


@dataclass
class RecommendationPipeline:
    vision: VisionModel
    embedder: Embedder
    store: VectorStore
    generator: GenerationProvider
    tryon: TryOnProvider
    jobs: JobStore
    # Profiles for resolving a named body reference. Empty means the feature is simply
    # unavailable, which the stage reports rather than failing over.
    profiles: dict[str, CelebrityProfile] = field(default_factory=dict)
    # Carried between the retrieve and refine stages so refinement works on the objects
    # rather than re-parsing its own output.
    _last_recommendations: tuple[Recommendation, ...] = field(default=(), init=False, repr=False)

    def run(
        self,
        job: Job,
        photo: bytes,
        query: UserQuery,
        *,
        confirmed_shape: BodyShape | None = None,
        reference_image: bytes | None = None,
        want_generation: bool = True,
        want_tryon: bool = True,
    ) -> Job:
        metrics = self._analyze(job, photo, confirmed_shape, query.body_reference)
        if metrics is None:
            return job

        recommendations = self._retrieve(job, metrics, query, photo, reference_image)
        if recommendations is None:
            return job

        recommendations = self._refine(job, metrics, query, recommendations)

        self._generate(job, query, metrics, enabled=want_generation)
        self._tryon(job, photo, recommendations, enabled=want_tryon)

        job.complete()
        self.jobs.save(job)
        return job

    # -- stages -------------------------------------------------------------------

    def _analyze(
        self,
        job: Job,
        photo: bytes,
        confirmed_shape: BodyShape | None,
        body_reference: str | None = None,
    ) -> BodyMetrics | None:
        job.start_stage(StageName.ANALYZE)
        self.jobs.save(job)

        # A named body reference short-circuits photo analysis entirely: the user has
        # stated whose proportions to use, so inferring different ones from a picture
        # and then blending them would only add noise.
        if body_reference:
            resolver = BodyReferenceResolver(self.profiles)
            match = resolver.resolve(body_reference)
            if match is not None:
                metrics = match.to_metrics()
                job.finish_stage(
                    StageName.ANALYZE,
                    result={
                        "shape": metrics.shape.value,
                        "build": metrics.build.value,
                        "height_band": metrics.height_band.value,
                        "confidence": 1.0,
                        "user_confirmed": True,
                        "needs_confirmation": False,
                        "body_reference": match.profile.name,
                        "body_reference_exact": match.exact,
                        "shoulder_waist_ratio": None,
                        "waist_hip_ratio": None,
                    },
                    detail=f"Matched to {match.profile.name}'s proportions.",
                )
                self.jobs.save(job)
                return metrics

            suggestions = resolver.suggest(body_reference)
            job.finish_stage(
                StageName.ANALYZE,
                state=StageState.FAILED,
                detail=(
                    f"No celebrity named {body_reference!r} is in the profile set."
                    + (f" Did you mean: {', '.join(suggestions)}?" if suggestions else "")
                ),
            )
            job.fail(f"Unknown body reference {body_reference!r}.")
            self.jobs.save(job)
            return None

        try:
            metrics = self.vision.analyze_body(photo)
        except Exception as exc:
            log.warning("body analysis failed", exc_info=True)
            job.finish_stage(StageName.ANALYZE, state=StageState.FAILED, detail=str(exc)[:200])
            job.fail("Could not analyse the photo.")
            self.jobs.save(job)
            return None

        if confirmed_shape is not None:
            # The user's correction always wins. Single-photo inference is corrupted by
            # pose, angle and clothing, so their answer is evidence and the model's is
            # a prior.
            metrics = metrics.model_copy(
                update={"shape": confirmed_shape, "user_confirmed": True, "confidence": 1.0}
            )

        job.finish_stage(
            StageName.ANALYZE,
            result={
                "shape": metrics.shape.value,
                "build": metrics.build.value,
                "height_band": metrics.height_band.value,
                "confidence": metrics.confidence,
                "user_confirmed": metrics.user_confirmed,
                # The UI uses this to decide whether to ask for confirmation.
                "needs_confirmation": not metrics.is_trustworthy,
                "shoulder_waist_ratio": metrics.shoulder_waist_ratio,
                "waist_hip_ratio": metrics.waist_hip_ratio,
            },
        )
        self.jobs.save(job)
        return metrics

    def _retrieve(
        self,
        job: Job,
        metrics: BodyMetrics,
        query: UserQuery,
        photo: bytes,
        reference_image: bytes | None = None,
    ) -> list[dict[str, object]] | None:
        job.start_stage(StageName.RETRIEVE)
        self.jobs.save(job)
        try:
            result = Retriever(self.embedder, self.store, vision=self.vision).retrieve(
                metrics, query, photo=photo, reference_image=reference_image
            )
        except Exception as exc:
            log.warning("retrieval failed", exc_info=True)
            job.finish_stage(StageName.RETRIEVE, state=StageState.FAILED, detail=str(exc)[:200])
            job.fail("Could not retrieve recommendations.")
            self.jobs.save(job)
            return None

        self._last_recommendations = result.recommendations
        payload: list[dict[str, object]] = [as_payload(rec) for rec in result.recommendations]

        job.finish_stage(
            StageName.RETRIEVE,
            result={
                "recommendations": payload,
                "considered": result.considered,
                "channels": list(result.channels_used),
                "relaxed": result.relaxed,
            },
            detail=(
                "No outfits matched this body shape, so results span other shapes and "
                "carry adjustment notes."
                if result.relaxed
                else ""
            ),
        )
        self.jobs.save(job)
        return payload

    def _refine(
        self,
        job: Job,
        metrics: BodyMetrics,
        query: UserQuery,
        recommendations: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """ImageRAG's gap loop, run over the result set rather than a generated image."""
        job.start_stage(StageName.REFINE)
        self.jobs.save(job)

        if not query.text.strip():
            job.finish_stage(
                StageName.REFINE,
                state=StageState.SKIPPED,
                detail="No text request, so there is nothing to check the results against.",
            )
            self.jobs.save(job)
            return recommendations

        try:
            result = ImageRagRefiner(self.vision, self.embedder, self.store).refine(
                query.text,
                metrics,
                self._last_recommendations,
                where=Filter(
                    must_equal={
                        key: value
                        for key, value in (
                            ("body_shape", metrics.shape.value),
                            ("region", query.region or ""),
                        )
                        if value
                    }
                ),
            )
        except Exception as exc:
            log.warning("refinement failed", exc_info=True)
            job.finish_stage(StageName.REFINE, state=StageState.FAILED, detail=str(exc)[:200])
            self.jobs.save(job)
            return recommendations

        added = [as_payload(rec) for rec in result.added]
        job.finish_stage(
            StageName.REFINE,
            state=StageState.DONE,
            result={
                "added": added,
                "rounds": result.rounds,
                "gaps": [
                    {"concept": g.concept, "filled": g.filled, "filled_by": list(g.filled_by)}
                    for g in result.gaps
                ],
                "unmet": list(result.unmet),
            },
            detail=(
                "Nothing in this wardrobe covers: " + ", ".join(result.unmet)
                if result.unmet
                else ""
            ),
        )
        self.jobs.save(job)
        return recommendations + added

    def _generate(self, job: Job, query: UserQuery, metrics: BodyMetrics, *, enabled: bool) -> None:
        job.start_stage(StageName.GENERATE)
        if not enabled or not self.generator.available:
            job.finish_stage(
                StageName.GENERATE,
                state=StageState.SKIPPED,
                detail=(
                    "Generation was not requested."
                    if not enabled
                    else "No generation backend is configured; showing real celebrity "
                    "outfits instead of a generated preview."
                ),
            )
            self.jobs.save(job)
            return

        try:
            result = ImageRagGenerator(
                self.vision, self.embedder, self.store, self.generator
            ).generate(query, metrics)
        except Exception as exc:
            log.warning("generation failed", exc_info=True)
            job.finish_stage(StageName.GENERATE, state=StageState.FAILED, detail=str(exc)[:200])
            self.jobs.save(job)
            return

        job.finish_stage(
            StageName.GENERATE,
            state=StageState.DONE if result.image else StageState.SKIPPED,
            result={
                "has_image": result.image is not None,
                "converged": result.converged,
                "rounds": len(result.rounds),
                "references": list(result.references),
                "gaps": [g.concept for r in result.rounds for g in r.gaps],
            },
            detail=result.unavailable_reason or "",
        )
        self.jobs.save(job)

    def _tryon(
        self,
        job: Job,
        photo: bytes,
        recommendations: list[dict[str, object]],
        *,
        enabled: bool,
    ) -> None:
        job.start_stage(StageName.TRYON)
        if not enabled or not self.tryon.available or not recommendations:
            job.finish_stage(
                StageName.TRYON,
                state=StageState.SKIPPED,
                detail=(
                    "Try-on was not requested."
                    if not enabled
                    else "No try-on backend is configured; showing a side-by-side "
                    "comparison instead."
                    if not self.tryon.available
                    else "Nothing to try on."
                ),
            )
            self.jobs.save(job)
            return

        garment_path = Path(str(recommendations[0].get("image_path", "")))
        if not garment_path.exists():
            job.finish_stage(
                StageName.TRYON,
                state=StageState.SKIPPED,
                detail="The recommended outfit image is unavailable.",
            )
            self.jobs.save(job)
            return

        try:
            rendered = self.tryon.try_on(photo, garment_path.read_bytes())
        except Exception as exc:
            log.warning("try-on failed", exc_info=True)
            job.finish_stage(StageName.TRYON, state=StageState.FAILED, detail=str(exc)[:200])
            self.jobs.save(job)
            return

        job.finish_stage(
            StageName.TRYON,
            state=StageState.DONE if rendered else StageState.SKIPPED,
            result={
                "has_image": rendered is not None,
                "outfit_id": recommendations[0].get("outfit_id"),
            },
            detail="" if rendered else "Try-on produced no image; showing side-by-side.",
        )
        self.jobs.save(job)
