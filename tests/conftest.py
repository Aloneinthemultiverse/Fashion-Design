"""Shared fixtures.

Everything here is offline and deterministic -- no network, no Docker, no API key.
"""

from __future__ import annotations

import pytest

from fashion.core.models import (
    Culture,
    Neckline,
    Occasion,
    OutfitItem,
    Silhouette,
    WaistEmphasis,
)


def make_outfit(
    item_id: str = "o1",
    *,
    celebrity_id: str = "c1",
    garment_type: str = "anarkali",
    silhouette: Silhouette = Silhouette.A_LINE,
    neckline: Neckline = Neckline.V_NECK,
    waist_emphasis: WaistEmphasis = WaistEmphasis.HIGH,
    culture: Culture = Culture.ETHNIC,
    occasion: Occasion = Occasion.FESTIVE,
    caption: str = "A festive embroidered anarkali in crimson silk.",
) -> OutfitItem:
    """Build an OutfitItem with sane defaults; override only what a test cares about."""
    return OutfitItem(
        id=item_id,
        celebrity_id=celebrity_id,
        image_path=f"images/{item_id}.jpg",
        garment_type=garment_type,
        silhouette=silhouette,
        neckline=neckline,
        waist_emphasis=waist_emphasis,
        culture=culture,
        occasion=occasion,
        colors=("crimson", "gold"),
        caption=caption,
        source="test-fixture",
        license="CC0",
    )


@pytest.fixture
def outfit() -> OutfitItem:
    return make_outfit()
