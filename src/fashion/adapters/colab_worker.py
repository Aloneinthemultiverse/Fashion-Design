"""Colab worker client: SDXL + IP-Adapter generation and IDM-VTON try-on.

The CPU-only, zero-cost constraint leaves one free source of GPU: a Google Colab
notebook on the free T4 tier, exposed over a tunnel. `colab/worker.ipynb` runs the models
behind a small HTTP API; this is the client.

Colab's free tier disconnects on idle and caps sessions, so the worker is treated as
*optional infrastructure that is usually absent*. Every failure path returns None rather
than raising, matching the null providers, so the application degrades to showing
retrieved reference outfits instead of breaking. `available` probes with a short timeout
and caches the answer briefly, because checking liveness per request would add a round
trip to every call.

Images cross the wire base64-encoded inside JSON. That is roughly a third larger than raw
bytes, but it survives tunnel proxies that mangle multipart uploads, and generation
latency dwarfs the transfer either way.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

HEALTH_CACHE_SECONDS = 30.0
PROBE_TIMEOUT = 5.0

# A free Cloudflare Quick Tunnel terminates any single request that runs past roughly
# 100 seconds, and SDXL on a T4 takes longer. Doing the work inside the request returns
# HTTP 524 while the worker is perfectly healthy, which is exactly what happened. So the
# worker submits and this polls: each individual request finishes in milliseconds and
# only the polling loop is long.
SUBMIT_TIMEOUT = 30.0
POLL_INTERVAL = 4.0
POLL_TIMEOUT = 420.0


@dataclass
class ColabWorkerClient:
    """Shared transport for the generation and try-on providers."""

    base_url: str
    timeout: float = SUBMIT_TIMEOUT
    _healthy_until: float = field(default=0.0, init=False)
    _healthy: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    def is_available(self) -> bool:
        if not self.base_url:
            return False
        now = time.monotonic()
        if now < self._healthy_until:
            return self._healthy
        self._healthy = self._probe()
        self._healthy_until = now + HEALTH_CACHE_SECONDS
        return self._healthy

    def _probe(self) -> bool:
        try:
            request = urllib.request.Request(f"{self.base_url}/health")
            with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT) as response:
                return bool(200 <= response.status < 300)
        except (urllib.error.URLError, TimeoutError, OSError):
            log.info("colab worker at %s is not reachable", self.base_url)
            return False

    def get(self, path: str) -> dict[str, object] | None:
        request = urllib.request.Request(f"{self.base_url}{path}")
        try:
            with urllib.request.urlopen(request, timeout=SUBMIT_TIMEOUT) as response:
                parsed: dict[str, object] = json.load(response)
                return parsed
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return None

    def submit(self, path: str, payload: dict[str, object]) -> bytes | None:
        """Submit a job and wait for its result."""
        accepted = self.post(path, payload)
        job_id = str((accepted or {}).get("job_id", ""))
        if not job_id:
            # An older worker that still answers inline. Accept its image so a
            # not-yet-updated notebook keeps working.
            return _decode(accepted)
        return _poll(self, job_id)

    def post(self, path: str, payload: dict[str, object]) -> dict[str, object] | None:
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                parsed: dict[str, object] = json.load(response)
                return parsed
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            log.warning("colab worker call to %s failed", path, exc_info=True)
            # A failed call means the session probably died; force the next probe.
            self._healthy_until = 0.0
            return None


def _poll(client: ColabWorkerClient, job_id: str) -> bytes | None:
    """Wait for a submitted job, returning its image.

    Returns None on failure or timeout rather than raising: an unavailable backend is a
    normal outcome here and callers already degrade gracefully.
    """
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL)
        result = client.get(f"/result/{job_id}")
        if result is None:
            continue
        state = str(result.get("state", ""))
        if state == "done":
            return _decode(result)
        if state == "failed":
            log.warning("colab job failed: %s", str(result.get("error"))[:200])
            return None
    log.warning("colab job %s did not finish within %.0fs", job_id, POLL_TIMEOUT)
    return None


def _decode(payload: dict[str, object] | None, key: str = "image") -> bytes | None:
    if payload is None:
        return None
    encoded = payload.get(key)
    if not isinstance(encoded, str) or not encoded:
        return None
    try:
        return base64.b64decode(encoded)
    except (ValueError, TypeError):
        log.warning("colab worker returned undecodable image data")
        return None


class ColabGenerationProvider:
    """SDXL with IP-Adapter image conditioning, running on the Colab worker."""

    def __init__(self, base_url: str, *, timeout: float = SUBMIT_TIMEOUT) -> None:
        self._client = ColabWorkerClient(base_url, timeout=timeout)

    @property
    def available(self) -> bool:
        return self._client.is_available()

    @property
    def supports_references(self) -> bool:
        # IP-Adapter is loaded by the worker notebook, so references are genuinely
        # consumed. This is what makes the ImageRAG loop meaningful rather than a
        # plain text prompt.
        return True

    def generate(
        self,
        prompt: str,
        *,
        references: tuple[bytes, ...] = (),
        seed: int | None = None,
    ) -> bytes | None:
        if not self.available:
            return None
        payload: dict[str, object] = {
            "prompt": prompt,
            "references": [base64.b64encode(r).decode() for r in references],
        }
        if seed is not None:
            payload["seed"] = seed
        return self._client.submit("/generate", payload)


class ColabTryOnProvider:
    """IDM-VTON try-on, running on the Colab worker."""

    def __init__(self, base_url: str, *, timeout: float = SUBMIT_TIMEOUT) -> None:
        self._client = ColabWorkerClient(base_url, timeout=timeout)

    @property
    def available(self) -> bool:
        return self._client.is_available()

    def try_on(self, person: bytes, garment: bytes) -> bytes | None:
        if not self.available:
            return None
        return self._client.submit(
            "/tryon",
            {
                "person": base64.b64encode(person).decode(),
                "garment": base64.b64encode(garment).decode(),
            },
        )
