"""Provider wiring.

One place where configuration becomes objects. Every builder falls back to the safe
free default rather than raising when a real provider cannot be constructed: a missing
API key or a dead Colab session should degrade the feature, not take down the process.

The exception is the vector store, where silently falling back to an empty in-memory
store would make a misconfigured deployment look like an empty corpus. That fails loudly.
"""

from __future__ import annotations

import logging

from fashion.adapters.embed_fake import FakeEmbedder
from fashion.adapters.gen_null import NullGenerationProvider, NullTryOnProvider
from fashion.adapters.store_memory import InMemoryVectorStore
from fashion.adapters.vision_fake import FakeVisionModel
from fashion.config import Settings, load_settings
from fashion.ports.embedder import Embedder
from fashion.ports.generation import GenerationProvider
from fashion.ports.tryon import TryOnProvider
from fashion.ports.vectorstore import VectorStore
from fashion.ports.vision import VisionModel

log = logging.getLogger(__name__)


def build_vision(settings: Settings | None = None) -> VisionModel:
    settings = settings or load_settings()
    if settings.vision_provider == "fake":
        return FakeVisionModel()

    if settings.vision_provider == "gemini":
        if not settings.gemini_api_key:
            log.warning(
                "vision_provider=gemini but FASHION_GEMINI_API_KEY is unset; "
                "falling back to the fake model. Labels produced now will be meaningless."
            )
            return FakeVisionModel()
        from fashion.adapters.vision_gemini import GeminiVisionModel

        return GeminiVisionModel(
            settings.gemini_api_key,
            model=settings.gemini_model,
            cache_dir=settings.cache_dir / "vlm",
        )

    raise ValueError(f"unsupported vision provider {settings.vision_provider!r}")


def build_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or load_settings()
    if settings.embed_provider == "fake":
        return FakeEmbedder()

    from fashion.adapters.embed_openclip import OpenClipEmbedder

    return OpenClipEmbedder(settings.clip_model, settings.clip_pretrained)


def build_store(settings: Settings | None = None) -> VectorStore:
    settings = settings or load_settings()
    if settings.store_provider == "memory":
        return InMemoryVectorStore()

    # No fallback here on purpose: an empty in-memory store is indistinguishable from
    # an empty corpus, so a misconfigured Qdrant would look like "no results" rather
    # than an outage.
    from fashion.adapters.store_qdrant import QdrantVectorStore

    return QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        collection=settings.qdrant_collection,
    )


def build_generation(settings: Settings | None = None) -> GenerationProvider:
    settings = settings or load_settings()
    if settings.generation_provider == "colab" and settings.colab_worker_url:
        from fashion.adapters.colab_worker import ColabGenerationProvider

        return ColabGenerationProvider(settings.colab_worker_url)
    if settings.generation_provider == "colab":
        log.warning("generation_provider=colab but FASHION_COLAB_WORKER_URL is unset")
    return NullGenerationProvider()


def build_tryon(settings: Settings | None = None) -> TryOnProvider:
    settings = settings or load_settings()
    if settings.tryon_provider == "colab" and settings.colab_worker_url:
        from fashion.adapters.colab_worker import ColabTryOnProvider

        return ColabTryOnProvider(settings.colab_worker_url)
    if settings.tryon_provider == "colab":
        log.warning("tryon_provider=colab but FASHION_COLAB_WORKER_URL is unset")
    return NullTryOnProvider()


def load_corpus(embedder: Embedder, store: VectorStore, settings: Settings | None = None) -> int:
    """Populate an in-memory store from the labelled corpus on disk.

    The in-memory store is process-local and starts empty, so every entry point that
    uses it has to seed it. Qdrant persists across restarts and is left alone.

    Returns the number of indexed outfits.
    """
    settings = settings or load_settings()
    if settings.store_provider != "memory":
        return store.count()

    from fashion.core.dataset import CelebrityRepository, OutfitRepository
    from fashion.pipeline.index import IndexBuilder

    items, report = OutfitRepository(settings.labels_path).load()
    if not report.ok:
        log.warning("skipped %d malformed outfit rows while loading", report.skipped)
    profiles = CelebrityRepository(settings.data_dir / "celebrity_profiles.jsonl").index()
    stats = IndexBuilder(embedder, store).build(items, profiles)
    log.info("indexed %d outfits into the in-memory store", stats.indexed)
    return stats.indexed
