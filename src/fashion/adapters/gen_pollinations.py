"""Hosted text-to-image generation, free and keyless.

Closes the one part of the problem statement a CPU-only deployment otherwise cannot
reach: Stable Diffusion output. Generation happens on someone else's hardware, so it
works on this laptop, needs no account, and costs nothing.

The important limitation, stated plainly because the ImageRAG loop depends on it:
**this backend cannot be conditioned on reference images.** `supports_references` is
therefore False, and the refinement loop stops and reports its gaps rather than
silently dropping the references and passing off a plain text prompt as
reference-guided generation. For true ImageRAG generation the Colab worker, with
IP-Adapter, is still the only option.

Only a text prompt leaves this machine -- never the user's photo. That matters because
the privacy note promises photos are processed in memory and not sent anywhere they do
not have to go, and generation does not have to.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger(__name__)

ENDPOINT = "https://image.pollinations.ai/prompt/"
USER_AGENT = "FashionRecommender/0.1 (research prototype)"

DEFAULT_TIMEOUT = 90.0
WIDTH, HEIGHT = 768, 1024  # Portrait: full-body outfit shots are taller than wide.

# Steers away from the failure modes that make generated fashion images unusable.
STYLE_SUFFIX = (
    "full body fashion photograph, single person, complete outfit visible, "
    "studio lighting, sharp focus, plain background"
)


class PollinationsGenerationProvider:
    """Text-to-image over HTTP. No key, no GPU, no account."""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        width: int = WIDTH,
        height: int = HEIGHT,
    ) -> None:
        self._timeout = timeout
        self._width = width
        self._height = height

    @property
    def available(self) -> bool:
        # Reported available without probing: the endpoint is stateless and a liveness
        # check would add a round trip to every request for a service whose failure is
        # already handled by returning None.
        return True

    @property
    def supports_references(self) -> bool:
        """False, and the ImageRAG loop relies on this being honest.

        Claiming reference support here would make the loop hand over retrieved
        outfits that this backend cannot use, and present the result as
        reference-guided when it is a plain prompt.
        """
        return False

    def generate(
        self,
        prompt: str,
        *,
        references: tuple[bytes, ...] = (),
        seed: int | None = None,
    ) -> bytes | None:
        if references:
            # Not silently ignored: the caller checks supports_references first, so
            # arriving here with references means a caller bug worth surfacing.
            log.warning(
                "%d reference image(s) discarded: this backend cannot condition on "
                "images. Use the Colab worker for reference-guided generation.",
                len(references),
            )

        full_prompt = f"{prompt.strip()}, {STYLE_SUFFIX}"
        params = {
            "width": str(self._width),
            "height": str(self._height),
            "nologo": "true",
        }
        if seed is not None:
            params["seed"] = str(seed)

        url = (
            ENDPOINT
            + urllib.parse.quote(full_prompt, safe="")
            + "?"
            + urllib.parse.urlencode(params)
        )
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                data: bytes = response.read()
        except (urllib.error.URLError, TimeoutError, OSError):
            log.warning("image generation failed", exc_info=True)
            return None

        # A short body is an error page, not an image. Passing it on would surface a
        # broken picture rather than the graceful fallback the UI already handles.
        if len(data) < 1024 or not data.startswith((b"\xff\xd8", b"\x89PNG")):
            log.warning("generation returned %d bytes that are not an image", len(data))
            return None
        return data
