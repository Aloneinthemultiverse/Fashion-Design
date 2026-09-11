"""Reciprocal Rank Fusion.

The retrieval engine queries several channels (image vector, caption vector, structured
tag match) whose scores live in incomparable spaces — cosine similarity from two
different CLIP towers plus a boolean overlap count. Normalising those into a shared
scale requires calibration data we do not have, so we fuse by *rank* instead, which
needs none.

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and individual
rank learning methods" (SIGIR 2009).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

RRF_K = 60  # Standard damping constant from the paper; blunts the top-rank advantage.


def reciprocal_rank_fusion(
    channels: Mapping[str, Sequence[str]],
    *,
    weights: Mapping[str, float] | None = None,
    k: int = RRF_K,
) -> list[tuple[str, float, tuple[str, ...]]]:
    """Fuse ranked ID lists into one ranking.

    Args:
        channels: channel name -> IDs in descending relevance order.
        weights: optional per-channel multiplier. Missing channels default to 1.0.
        k: damping constant.

    Returns:
        (id, fused_score, channels_that_matched) sorted by score descending. Ties break
        on id so the output is deterministic — important for golden-set tests.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    weights = weights or {}
    scores: dict[str, float] = {}
    matched: dict[str, list[str]] = {}

    for channel, ids in channels.items():
        weight = weights.get(channel, 1.0)
        for rank, item_id in enumerate(ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)
            matched.setdefault(item_id, []).append(channel)

    return sorted(
        ((i, round(s, 6), tuple(matched[i])) for i, s in scores.items()),
        key=lambda row: (-row[1], row[0]),
    )


def take(fused: Iterable[tuple[str, float, tuple[str, ...]]], n: int) -> list[str]:
    return [row[0] for row in list(fused)[:n]]
