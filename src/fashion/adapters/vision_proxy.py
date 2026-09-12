"""Vision model over an Anthropic-compatible proxy.

Targets antigravity-claude-proxy, which exposes Gemini and Claude behind the Anthropic
Messages API on localhost. That removes the API-key problem entirely: the proxy holds
the Google OAuth session, and this client sends a placeholder token.

Two consequences worth stating, because they shape the code:

* **The wire format is Anthropic, not Google.** Images go as base64 `image` content
  blocks, and the response is a `content` array of blocks rather than a single `text`.
* **There is no `response_schema`.** Gemini's native API can constrain output to a JSON
  schema; the Anthropic format cannot. So the schema is described in the prompt and the
  response is parsed defensively -- models wrap JSON in prose or code fences often
  enough that treating the whole body as JSON would fail regularly.

Responses are cached on disk exactly as the direct Gemini adapter does. The proxy is
free, but it is not fast, and a repeated labelling pass should not re-pay that cost.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import urllib.error
import urllib.request
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

DEFAULT_URL = "http://localhost:8080"
DEFAULT_MODEL = "gemini-3.6-flash-high"
DEFAULT_TIMEOUT = 180.0
MAX_TOKENS = 4096

# The proxy authenticates through its own Google session; the token is a placeholder
# the Anthropic wire format requires but the proxy ignores.
PLACEHOLDER_TOKEN = "test"

JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Without schema-constrained decoding the model may wrap JSON in prose or a fenced
    code block. Taking the outermost brace-delimited span recovers it in the cases that
    actually occur, and an unparseable response degrades to an empty dict rather than
    raising into a user's request.
    """
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    match = JSON_BLOCK.search(body)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
    log.warning("could not parse JSON from response: %.160s", body)
    return {}


class ProxyVisionModel:
    """VisionModel backed by an Anthropic-compatible local proxy."""

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        *,
        model: str = DEFAULT_MODEL,
        cache_dir: Path | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    # -- transport ---------------------------------------------------------------

    def available(self) -> bool:
        try:
            request = urllib.request.Request(f"{self._url}/", method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                return bool(200 <= int(response.status) < 500)
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def _cache_key(self, task: str, prompt: str, images: tuple[bytes, ...]) -> str:
        digest = hashlib.sha256()
        digest.update(task.encode())
        digest.update(self._model.encode())
        digest.update(prompt.encode())
        for image in images:
            digest.update(hashlib.sha256(image).digest())
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
            log.debug("could not cache %s", key, exc_info=True)

    def call(
        self,
        task: str,
        prompt: str,
        images: tuple[bytes, ...] = (),
        *,
        system: str = "",
    ) -> dict[str, Any]:
        """Send one message and return the parsed JSON body."""
        key = self._cache_key(task, prompt + system, images)
        cached = self._cached(key)
        if cached is not None:
            return cached

        content: list[dict[str, Any]] = []
        for image in images:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(image).decode(),
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            payload["system"] = system

        request = urllib.request.Request(
            f"{self._url}/v1/messages",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": PLACEHOLDER_TOKEN,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            log.warning("proxy call %s failed: %s", task, exc)
            return {}

        # Anthropic returns a list of content blocks; concatenate the text ones.
        blocks = body.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
        parsed = extract_json(text)
        stop = body.get("stop_reason")
        if stop == "max_tokens":
            # The outer JSON object is unclosed, so extract_json recovers only a nested
            # fragment. Caching that would bake a half-record into the corpus.
            log.warning("response for %s hit the token limit and was discarded", task)
            return {}
        parsed.setdefault("_raw", text[:2000])
        self._store(key, parsed)
        return parsed

    # -- VisionModel -------------------------------------------------------------

    def _analyze_all(self, image: bytes) -> dict[str, Any]:
        from fashion.adapters.prompts import ANALYSIS_PROMPT, ANALYSIS_SYSTEM

        return self.call("analyze_all", ANALYSIS_PROMPT, (image,), system=ANALYSIS_SYSTEM)

    def analyze_body(self, image: bytes) -> BodyMetrics:
        data = self._analyze_all(image).get("body", {}) or {}
        proportions = Proportions(
            shoulder=float(data.get("shoulder_width") or 1.0),
            waist=float(data.get("waist_width") or 1.0),
            hip=float(data.get("hip_width") or 1.0),
        )
        confidence = confidence_from_ratios(proportions)
        if not data.get("full_body_visible", True):
            confidence *= 0.5
        return BodyMetrics(
            shape=classify(proportions),
            build=Build(str(data.get("build", Build.ATHLETIC.value))),
            height_band=HeightBand(str(data.get("height_band", HeightBand.AVERAGE.value))),
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
        from fashion.adapters.prompts import GAP_PROMPT

        data = self.call("find_gaps", GAP_PROMPT.format(request=prompt), (generated,))
        return _to_concepts(data)

    def find_missing_concepts(
        self, request: str, found: tuple[str, ...]
    ) -> tuple[MissingConcept, ...]:
        from fashion.adapters.prompts import MISSING_PROMPT

        listing = "\n".join(f"- {f}" for f in found) or "- (nothing retrieved)"
        data = self.call("find_missing", MISSING_PROMPT.format(request=request, listing=listing))
        return _to_concepts(data)

    def expand_query(self, text: str, n: int = 3) -> tuple[str, ...]:
        from fashion.adapters.prompts import EXPAND_PROMPT

        query = text.strip()
        if not query:
            return ()
        data = self.call("expand", EXPAND_PROMPT.format(n=n, query=query))
        variants = [
            str(q).strip()
            for q in (data.get("queries") or [])
            if str(q).strip() and str(q).strip().casefold() != query.casefold()
        ]
        return (query, *variants[:n])

    def write_rationale(self, outfit: OutfitItem, metrics: BodyMetrics) -> str:
        from fashion.core.crosscultural import adjustments, explain

        grounded = explain(outfit, metrics.shape)
        fixes = adjustments(outfit, metrics.shape)
        data = self.call(
            "rationale",
            "Rewrite this styling note as two warm, plain sentences for the wearer. "
            "Add no garment detail that is not already stated. "
            'Return JSON: {"text": "..."}\n\n'
            f"Note: {grounded}\nAdjustments: {'; '.join(fixes) or 'none'}",
        )
        return str(data.get("text", "")).strip() or grounded


def _to_concepts(data: dict[str, Any]) -> tuple[MissingConcept, ...]:
    out: list[MissingConcept] = []
    for entry in data.get("missing", []) or []:
        if not isinstance(entry, dict):
            continue
        concept = str(entry.get("concept", "")).strip()
        caption = str(entry.get("retrieval_caption", "")).strip()
        if not concept or not caption:
            continue
        if len(caption) < len(concept):
            caption = f"{concept}: {caption}"
        out.append(MissingConcept(concept=concept, retrieval_caption=caption))
    return tuple(out)
