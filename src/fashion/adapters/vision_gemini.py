"""Gemini vision adapter.

Uses the Google AI Studio free tier (~1,500 requests/day, no card). That quota is the
scarcest resource in the project -- labelling a 2,000-image corpus consumes most of a
day's allowance -- so two things are non-negotiable here:

* **Caching.** Every response is cached on a hash of (image, task, model). Re-running a
  labelling pass over an already-processed corpus then costs nothing, which makes the
  pass safely resumable and makes tests against recorded responses free.
* **Schema-constrained output.** Responses are requested as JSON against an explicit
  schema and validated. A free-text response that has to be re-prompted costs quota
  twice.

Body *measurement* is asked of the model; body *classification* is not. The model
estimates proportions, and `fashion.core.body.classify` names the shape. Models are
inconsistent at naming shapes and consistent enough at comparing widths, and keeping the
naming deterministic makes it testable and identical for every user.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from fashion.core.body import Proportions, classify, confidence_from_ratios
from fashion.core.models import (
    BodyMetrics,
    Build,
    HeightBand,
    MissingConcept,
    OutfitItem,
)

log = logging.getLogger(__name__)

GAP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "missing": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "retrieval_caption": {"type": "string"},
                },
                "required": ["concept", "retrieval_caption"],
            },
        }
    },
    "required": ["missing"],
}

GAP_PROMPT = """Compare the image against this requested description:

{prompt}

List concepts that the description requires but the image does not show. For each, also
write a detailed retrieval_caption: a rich visual description of that concept alone, as
it should appear, suitable for finding a reference photograph of it. The caption must be
substantially more detailed than the concept name.

If the image already satisfies the description, return an empty list.
"""


class GeminiVisionModel:
    """VisionModel backed by Gemini, with on-disk response caching."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-2.5-flash",
        cache_dir: Path | None = None,
        client: Any = None,
    ) -> None:
        if not api_key and client is None:
            raise ValueError(
                "Gemini API key is required. Get a free one at "
                "https://aistudio.google.com/apikey and set FASHION_GEMINI_API_KEY."
            )
        self._model = model
        self._cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = client if client is not None else self._build_client(api_key)

    @staticmethod
    def _build_client(api_key: str) -> Any:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "google-genai is not installed. Run `uv sync --extra vision`."
            ) from exc
        return genai.Client(api_key=api_key)

    # -- caching -----------------------------------------------------------------

    def _cache_key(self, task: str, payload: bytes, extra: str = "") -> str:
        digest = hashlib.sha256()
        digest.update(task.encode())
        digest.update(self._model.encode())
        digest.update(extra.encode())
        digest.update(payload)
        return digest.hexdigest()

    def _cached(self, key: str) -> dict[str, Any] | None:
        if self._cache_dir is None:
            return None
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            return loaded
        except (json.JSONDecodeError, OSError):
            return None

    def _store(self, key: str, value: dict[str, Any]) -> None:
        if self._cache_dir is None:
            return
        try:
            (self._cache_dir / f"{key}.json").write_text(json.dumps(value), encoding="utf-8")
        except OSError:
            log.warning("could not write cache entry %s", key, exc_info=True)

    # -- model call --------------------------------------------------------------

    def _call(
        self,
        task: str,
        prompt: str,
        image: bytes | None,
        schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        key = self._cache_key(task, image or b"", prompt)
        cached = self._cached(key)
        if cached is not None:
            return cached

        from google.genai import types

        parts: list[Any] = [prompt]
        if image is not None:
            parts.append(types.Part.from_bytes(data=image, mime_type="image/jpeg"))

        config: dict[str, Any] = {}
        if schema is not None:
            config = {"response_mime_type": "application/json", "response_schema": schema}

        response = self._client.models.generate_content(
            model=self._model, contents=parts, config=config
        )
        text = (response.text or "").strip()
        try:
            parsed = json.loads(text) if schema is not None else {"text": text}
        except json.JSONDecodeError:
            log.warning("non-JSON response for task %s: %.120s", task, text)
            parsed = {"text": text}

        self._store(key, parsed)
        return parsed

    # -- VisionModel -------------------------------------------------------------

    def _analyze_all(self, image: bytes) -> dict[str, Any]:
        """One call returning body measurements, outfit tags and a caption together.

        The three public methods are all backed by this. Because responses are cached on
        the image hash, calling all three during ingest costs a single request rather
        than three -- which is the difference between roughly one day and four of free
        quota over a 2,000-image corpus.
        """
        from fashion.adapters.gemini_schemas import COMBINED_PROMPT, COMBINED_SCHEMA

        return self._call("analyze_all", COMBINED_PROMPT, image, COMBINED_SCHEMA)

    def analyze_body(self, image: bytes) -> BodyMetrics:
        data = self._analyze_all(image).get("body", {}) or {}

        proportions = Proportions(
            shoulder=float(data.get("shoulder_width") or 1.0),
            waist=float(data.get("waist_width") or 1.0),
            hip=float(data.get("hip_width") or 1.0),
        )
        confidence = confidence_from_ratios(proportions)
        if not data.get("full_body_visible", True):
            # Proportions read off a cropped photo are guesswork; halving confidence
            # pushes the result below the trust threshold so the UI asks the user.
            confidence *= 0.5

        return BodyMetrics(
            shape=classify(proportions),
            build=Build(data.get("build", Build.ATHLETIC.value)),
            height_band=HeightBand(data.get("height_band", HeightBand.AVERAGE.value)),
            shoulder_waist_ratio=round(proportions.shoulder / proportions.waist, 3),
            waist_hip_ratio=round(proportions.waist / proportions.hip, 3),
            confidence=round(confidence, 3),
        )

    def tag_outfit(self, image: bytes) -> dict[str, object]:
        outfit: dict[str, object] = self._analyze_all(image).get("outfit", {}) or {}
        return outfit

    def caption_outfit(self, image: bytes) -> str:
        return str(self._analyze_all(image).get("caption", "")).strip()

    def find_gaps(self, generated: bytes, prompt: str) -> tuple[MissingConcept, ...]:
        data = self._call("find_gaps", GAP_PROMPT.format(prompt=prompt), generated, GAP_SCHEMA)
        out: list[MissingConcept] = []
        for entry in data.get("missing", []) or []:
            concept = str(entry.get("concept", "")).strip()
            caption = str(entry.get("retrieval_caption", "")).strip()
            if not concept or not caption:
                continue
            if len(caption) < len(concept):
                # The whole ImageRAG result depends on the caption being richer than
                # the concept name; fall back rather than retrieve on a bare word.
                caption = f"{concept}: {caption}".strip()
            out.append(MissingConcept(concept=concept, retrieval_caption=caption))
        return tuple(out)

    def write_rationale(self, outfit: OutfitItem, metrics: BodyMetrics) -> str:
        from fashion.core.crosscultural import adjustments, explain

        # Grounded: the structural claim comes from the deterministic rules, and the
        # model only rephrases it. This keeps the model from inventing garment details
        # that are not recorded for this item.
        grounded = explain(outfit, metrics.shape)
        fixes = adjustments(outfit, metrics.shape)
        prompt = (
            "Rewrite this styling note as two warm, plain sentences for the wearer. "
            "Do not add any garment detail that is not already stated.\n\n"
            f"Note: {grounded}\n"
            f"Suggested adjustments: {'; '.join(fixes) or 'none'}"
        )
        result = self._call("write_rationale", prompt, None, None)
        return str(result.get("text", "")).strip() or grounded

    def expand_query(self, text: str, n: int = 3) -> tuple[str, ...]:
        """Generate query variations for RAG-Fusion.

        Cached like every other call, so repeating a popular search costs no quota. The
        original query is always returned first: the expansions are there to widen
        recall, not to replace what the user actually asked for.
        """
        query = text.strip()
        if not query:
            return ()

        schema = {
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["queries"],
        }
        prompt = (
            f"Rewrite this outfit search as {n} differently-phrased searches that would "
            f"retrieve overlapping but not identical results. Vary the vocabulary, the "
            f"level of detail, and whether garment names are specific or general. Keep "
            f"every rewrite faithful to the original intent.{chr(10)}{chr(10)}"
            f"Search: {query}"
        )
        data = self._call("expand_query", prompt, None, schema)
        variants = [
            str(q).strip()
            for q in (data.get("queries") or [])
            if str(q).strip() and str(q).strip().casefold() != query.casefold()
        ]
        return (query, *variants[:n])

    def find_missing_concepts(
        self, request: str, found: tuple[str, ...]
    ) -> tuple[MissingConcept, ...]:
        """Text-only gap analysis over a result set.

        Cheap compared with the image path -- no picture is uploaded -- and cached like
        every other call, so a repeated search costs no quota.
        """
        if not request.strip():
            return ()

        schema = {
            "type": "object",
            "properties": {
                "missing": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "concept": {"type": "string"},
                            "retrieval_caption": {"type": "string"},
                        },
                        "required": ["concept", "retrieval_caption"],
                    },
                }
            },
            "required": ["missing"],
        }
        listing = chr(10).join(f"- {item}" for item in found) or "- (nothing retrieved)"
        prompt = (
            "A user asked for an outfit. These are the outfits found for them.{nl}{nl}"
            "Request: {request}{nl}{nl}"
            "Found:{nl}{listing}{nl}{nl}"
            "List the concepts the request calls for that none of the found outfits "
            "provide. For each, write a retrieval_caption: a rich visual description of "
            "that concept alone, as it should appear, suitable for finding a photograph "
            "of it. The caption must be substantially more detailed than the concept "
            "name. Return an empty list if the results already satisfy the request."
        ).format(nl=chr(10), request=request, listing=listing)

        data = self._call("find_missing_concepts", prompt, None, schema)
        out: list[MissingConcept] = []
        for entry in data.get("missing", []) or []:
            concept = str(entry.get("concept", "")).strip()
            caption = str(entry.get("retrieval_caption", "")).strip()
            if not concept or not caption:
                continue
            if len(caption) < len(concept):
                caption = f"{concept}: {caption}".strip()
            out.append(MissingConcept(concept=concept, retrieval_caption=caption))
        return tuple(out)
