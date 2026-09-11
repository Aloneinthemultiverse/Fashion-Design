"""Qdrant vector store.

Same port as the in-memory store, so it is a configuration swap. Qdrant is used for two
capabilities the brute-force store cannot provide at corpus scale: native **named
vectors**, which is how one point carries both `img_vec` and `cap_vec`, and **filtered
search**, which applies the body-shape constraint inside the index rather than scanning
and discarding afterwards.

Point ids are deterministic UUIDs derived from the outfit id. Qdrant accepts only
unsigned integers or UUIDs as ids, while outfit ids are strings like `Q12345-0`; hashing
them keeps upserts idempotent, so re-running the index builder updates rows instead of
duplicating them. The original id stays in the payload.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fashion.ports.vectorstore import (
    CAPTION_VECTOR,
    IMAGE_VECTOR,
    Filter,
    OutfitVectors,
    SearchHit,
)

log = logging.getLogger(__name__)

# Fixed namespace so the same outfit id always maps to the same point id across runs
# and across machines.
NAMESPACE = uuid.UUID("6f9d3e1a-6c5e-4a2f-9a3b-1d7c8e5f2a40")


def point_id(outfit_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, outfit_id))


class QdrantVectorStore:
    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection: str = "outfits",
        client: Any = None,
    ) -> None:
        self._collection = collection
        self._client = client if client is not None else self._build_client(url, api_key)

    @staticmethod
    def _build_client(url: str, api_key: str | None) -> Any:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "qdrant-client is not installed. Run `uv sync --extra store`."
            ) from exc
        return QdrantClient(url=url, api_key=api_key)

    def ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self._client.collection_exists(self._collection):
            return
        # Cosine because both towers emit L2-normalised vectors; this keeps scores
        # directly comparable with the in-memory store's cosine implementation.
        params = VectorParams(size=dim, distance=Distance.COSINE)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={IMAGE_VECTOR: params, CAPTION_VECTOR: params},
        )
        # Index the fields used as hard filters; without payload indexes Qdrant scans
        # them, which negates the point of filtering in the engine.
        from qdrant_client.models import PayloadSchemaType

        for field in ("body_shape", "culture", "occasion", "celebrity_name", "region"):
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:  # pragma: no cover - index may already exist
                log.debug("payload index for %s not created", field, exc_info=True)

    def upsert(self, item_id: str, vectors: OutfitVectors, payload: dict[str, object]) -> None:
        from qdrant_client.models import PointStruct

        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=point_id(item_id),
                    vector={
                        IMAGE_VECTOR: vectors.img_vec,
                        CAPTION_VECTOR: vectors.cap_vec,
                    },
                    payload={**payload, "outfit_id": item_id},
                )
            ],
        )

    def _to_qdrant_filter(self, where: Filter | None) -> Any:
        if where is None or (not where.must_equal and not where.must_be_in):
            return None
        from qdrant_client.models import FieldCondition, MatchAny, MatchValue
        from qdrant_client.models import Filter as QFilter

        conditions: list[Any] = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in where.must_equal.items()
        ]
        conditions.extend(
            FieldCondition(key=key, match=MatchAny(any=list(allowed)))
            for key, allowed in where.must_be_in.items()
        )
        return QFilter(must=conditions)

    def search(
        self,
        vector: list[float],
        *,
        using: str = IMAGE_VECTOR,
        limit: int = 10,
        where: Filter | None = None,
    ) -> list[SearchHit]:
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            using=using,
            limit=limit,
            query_filter=self._to_qdrant_filter(where),
            with_payload=True,
        )
        return [
            SearchHit(
                id=str((point.payload or {}).get("outfit_id", point.id)),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in response.points
        ]

    def get(self, item_id: str) -> dict[str, object] | None:
        points = self._client.retrieve(
            collection_name=self._collection, ids=[point_id(item_id)], with_payload=True
        )
        if not points:
            return None
        return dict(points[0].payload or {})

    def count(self) -> int:
        return int(self._client.count(collection_name=self._collection).count)
