"""Hosted image *editing*: reference-guided generation and try-on without a GPU.

The keyless text-to-image endpoint (gen_pollinations) cannot see images, which left the
ImageRAG loop reporting its gaps instead of closing them, and left try-on off entirely
whenever the Colab worker was down -- which is most of the time. The authenticated
edits endpoint accepts images, so the same account closes both:

- generation: the retrieved celebrity outfits are uploaded as references, which is what
  the ImageRAG loop needs `supports_references` to be honest about;
- try-on: the user's photo and the garment go up together and the model dresses the
  person, keeping face, pose and background.

Unlike gen_pollinations, the user's photo does leave this machine on the try-on path.
That is inherent to hosted try-on and is why this provider is opt-in via config rather
than the default.

Each call spends account balance. An empty balance is an HTTP 402, and like every
other failure it returns None so the pipeline reports the stage as skipped.
"""

from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
import uuid

log = logging.getLogger(__name__)

ENDPOINT = "https://gen.pollinations.ai/v1/images/edits"
DEFAULT_MODEL = "openai/gpt-image-1.5"
DEFAULT_TIMEOUT = 400.0

# Reference-guided generation. The references are retrieved celebrity outfits; the
# model is told to borrow their garments, not their wearers.
GENERATE_PREFIX = (
    "The attached photos are reference outfits worn by celebrities with the same body "
    "shape as the client. Borrow their garment cuts, colours and embroidery, not the "
    "people. Create: "
)
GENERATE_SUFFIX = (
    ". Show the complete outfit from shoulders to feet on a single model, plain studio "
    "background, photorealistic fashion catalogue photo."
)

TRYON_PROMPT = (
    "Image 1 is a real person. Image 2 shows an outfit. Dress the person in image 1 in "
    "exactly the outfit from image 2: same garments, colours, embroidery, buttons, "
    "trousers and footwear. Keep their face, hair, skin tone, pose, hands, body "
    "proportions and background from image 1 exactly the same. Only change the "
    "clothing. Photorealistic, with fabric folds that suit the pose."
)


class PollinationsEditClient:
    """Multipart upload to the edits endpoint; returns image bytes or None."""

    def __init__(
        self, token: str, *, model: str = DEFAULT_MODEL, timeout: float = DEFAULT_TIMEOUT
    ) -> None:
        self._token = token
        self._model = model
        self._timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self._token)

    def edit(self, prompt: str, images: tuple[bytes, ...]) -> bytes | None:
        if not self._token or not images:
            return None
        boundary = uuid.uuid4().hex
        parts: list[bytes] = []
        for name, value in (("model", self._model), ("prompt", prompt)):
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
                f"\r\n\r\n{value}\r\n".encode()
            )
        for index, data in enumerate(images):
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="image[]"; '
                f'filename="image{index}.png"\r\nContent-Type: image/png\r\n\r\n'.encode()
                + data
                + b"\r\n"
            )
        parts.append(f"--{boundary}--\r\n".encode())

        request = urllib.request.Request(
            ENDPOINT,
            data=b"".join(parts),
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            # 402 is an empty balance: worth saying plainly, since it looks like an
            # outage otherwise.
            body = exc.read().decode("utf-8", "replace")[:200]
            log.warning("pollinations edit failed: HTTP %s %s", exc.code, body)
            return None
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            log.warning("pollinations edit failed", exc_info=True)
            return None

        try:
            return base64.b64decode(payload["data"][0]["b64_json"])
        except (KeyError, IndexError, TypeError, ValueError):
            log.warning("pollinations edit returned no image")
            return None


class PollinationsEditGenerationProvider:
    """Generation that genuinely consumes the ImageRAG references."""

    def __init__(self, client: PollinationsEditClient, fallback: object | None = None) -> None:
        self._client = client
        # Plain text-to-image for the unconditioned first pass; the edits endpoint
        # needs at least one input image.
        self._fallback = fallback

    @property
    def available(self) -> bool:
        return self._client.configured

    @property
    def supports_references(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        *,
        references: tuple[bytes, ...] = (),
        seed: int | None = None,
    ) -> bytes | None:
        if references:
            return self._client.edit(GENERATE_PREFIX + prompt + GENERATE_SUFFIX, references)
        if self._fallback is not None:
            return self._fallback.generate(prompt, seed=seed)  # type: ignore[attr-defined, no-any-return]
        return None


class PollinationsTryOnProvider:
    """Dress the user's own photo in the garment."""

    def __init__(self, client: PollinationsEditClient) -> None:
        self._client = client

    @property
    def available(self) -> bool:
        return self._client.configured

    def try_on(self, person: bytes, garment: bytes) -> bytes | None:
        return self._client.edit(TRYON_PROMPT, (person, garment))
