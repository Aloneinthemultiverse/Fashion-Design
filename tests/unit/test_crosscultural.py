"""Cross-cultural styling rules.

The central claim of the product is that matching happens on geometry, not ethnicity.
These tests pin that claim down so a future change cannot quietly reintroduce an
ethnicity-based rule.
"""

from __future__ import annotations

import pytest

from fashion.core.crosscultural import (
    GUIDANCE,
    NATURAL_FITS,
    adjustments,
    explain,
    fit_score,
)
from fashion.core.models import BodyShape, Culture, Neckline, Silhouette, WaistEmphasis
from tests.conftest import make_outfit


def test_every_shape_has_guidance_and_natural_fits() -> None:
    """A missing entry would make recommendations silently fail for real users."""
    for shape in BodyShape:
        assert shape in GUIDANCE
        assert shape in NATURAL_FITS
        assert NATURAL_FITS[shape]


def test_matching_outfit_scores_perfectly() -> None:
    g = GUIDANCE[BodyShape.PEAR]
    outfit = make_outfit(
        silhouette=next(iter(g.silhouettes)),
        neckline=next(iter(g.necklines)),
        waist_emphasis=next(iter(g.waist)),
    )
    assert fit_score(outfit, BodyShape.PEAR) == 1.0
    assert adjustments(outfit, BodyShape.PEAR) == ()


def test_mismatched_outfit_scores_zero_and_reports_every_fix() -> None:
    outfit = make_outfit(
        silhouette=Silhouette.BODYCON,
        neckline=Neckline.COLLARED,
        waist_emphasis=WaistEmphasis.DROPPED,
    )
    assert fit_score(outfit, BodyShape.PEAR) == 0.0
    assert len(adjustments(outfit, BodyShape.PEAR)) == 3


def test_score_is_independent_of_culture() -> None:
    """The same cut must score identically whether it is labelled ethnic or western.

    This is the geometry-not-ethnicity premise, asserted directly.
    """
    base = {
        "silhouette": Silhouette.A_LINE,
        "neckline": Neckline.V_NECK,
        "waist_emphasis": WaistEmphasis.HIGH,
    }
    ethnic = make_outfit("e", garment_type="anarkali", culture=Culture.ETHNIC, **base)
    western = make_outfit("w", garment_type="sundress", culture=Culture.WESTERN, **base)
    for shape in BodyShape:
        assert fit_score(ethnic, shape) == fit_score(western, shape)


@pytest.mark.parametrize("shape", list(BodyShape))
def test_rationale_only_mentions_recorded_attributes(shape: BodyShape) -> None:
    """Rationale is built from the payload, so it cannot invent garment details."""
    outfit = make_outfit(garment_type="lehenga")
    text = explain(outfit, shape)
    assert "lehenga" in text
    assert outfit.neckline.value.replace("_", "-") in text
    assert shape.value.replace("_", " ") in text


def test_adjustments_are_actionable_sentences() -> None:
    outfit = make_outfit(silhouette=Silhouette.BODYCON)
    fixes = adjustments(outfit, BodyShape.PEAR)
    assert fixes
    assert all(f.endswith(".") and len(f) > 20 for f in fixes)
