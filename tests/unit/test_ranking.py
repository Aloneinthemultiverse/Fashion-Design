"""Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from fashion.core.ranking import reciprocal_rank_fusion, take


def test_item_ranked_well_by_several_channels_wins() -> None:
    """The point of fusion: broad agreement beats a single channel's top pick."""
    fused = reciprocal_rank_fusion(
        {
            "img": ["a", "b", "c"],
            "cap": ["b", "a", "c"],
            "tags": ["b", "c", "a"],
        }
    )
    assert take(fused, 1) == ["b"]


def test_fusion_ignores_raw_score_scale() -> None:
    """Only rank order is consumed, so incomparable score scales cannot skew the result."""
    fused = reciprocal_rank_fusion({"img": ["x", "y"], "cap": ["x", "y"]})
    assert take(fused, 2) == ["x", "y"]


def test_weights_shift_the_outcome() -> None:
    channels = {"img": ["a", "b"], "cap": ["b", "a"]}
    assert take(reciprocal_rank_fusion(channels, weights={"cap": 5.0}), 1) == ["b"]
    assert take(reciprocal_rank_fusion(channels, weights={"img": 5.0}), 1) == ["a"]


def test_reports_which_channels_matched() -> None:
    """The UI explains why an item surfaced, so provenance has to survive fusion."""
    fused = reciprocal_rank_fusion({"img": ["a"], "cap": ["a", "b"]})
    by_id = {row[0]: row[2] for row in fused}
    assert set(by_id["a"]) == {"img", "cap"}
    assert set(by_id["b"]) == {"cap"}


def test_ties_break_deterministically() -> None:
    """Golden-set tests depend on a stable order for equally-scored items."""
    channels = {"img": ["b", "a"], "cap": ["a", "b"]}
    first = reciprocal_rank_fusion(channels)
    assert first == reciprocal_rank_fusion(channels)
    assert take(first, 2) == ["a", "b"]


def test_empty_input_is_not_an_error() -> None:
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"img": []}) == []


def test_rejects_invalid_damping() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion({"img": ["a"]}, k=0)
