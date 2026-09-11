"""OpenCLIP embedder.

Runs on CPU. ViT-B-32 encodes an image in roughly 50ms on a laptop, which makes the
whole retrieval layer free and local -- the one heavy component that does *not* need a
GPU or an API.

Both encoders project into a shared space, which is what makes cross-modal retrieval
work: a text embedding can be searched against the stored image vectors directly.
Vectors are L2-normalised on the way out so cosine similarity reduces to a dot product
and scores stay comparable across the two towers.
"""

from __future__ import annotations

import io
import logging
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


@lru_cache(maxsize=2)
def _load(model_name: str, pretrained: str) -> tuple[Any, Any, Any]:
    """Load and cache the model.

    Cached because construction downloads weights and takes seconds; the labelling and
    index-building scripts would otherwise pay that cost per invocation.
    """
    try:
        import open_clip
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "open-clip-torch is not installed. Run `uv sync --extra embed`."
        ) from exc

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    return model, preprocess, tokenizer


class OpenClipEmbedder:
    """Embedder backed by OpenCLIP, CPU-only."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
    ) -> None:
        self._model, self._preprocess, self._tokenizer = _load(model_name, pretrained)
        self._dim = int(self._model.text_projection.shape[1])

    @property
    def dim(self) -> int:
        return self._dim

    def _normalise(self, tensor: Any) -> list[float]:
        import torch

        with torch.no_grad():
            normalised = tensor / tensor.norm(dim=-1, keepdim=True)
        return [float(v) for v in normalised.squeeze(0)]

    def embed_text(self, text: str) -> list[float]:
        import torch

        # CLIP's text encoder truncates at 77 tokens. Captions are written to front-load
        # garment attributes, so truncation loses the least important tail.
        tokens = self._tokenizer([text])
        with torch.no_grad():
            features = self._model.encode_text(tokens)
        return self._normalise(features)

    def embed_image(self, image: bytes) -> list[float]:
        import torch
        from PIL import Image

        with Image.open(io.BytesIO(image)) as img:
            # Convert before preprocessing: Commons serves greyscale, palettised and
            # RGBA files, and the transform expects three channels.
            tensor = self._preprocess(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
        return self._normalise(features)
