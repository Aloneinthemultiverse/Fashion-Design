"""Body-shape classification.

These assertions encode the styling definitions themselves, so a change in thresholds
that silently reclassifies people will fail here rather than in production.
"""

from __future__ import annotations

import pytest

from fashion.core.body import Proportions, classify, confidence_from_ratios
from fashion.core.models import BodyShape


@pytest.mark.parametrize(
    ("shoulder", "waist", "hip", "expected"),
    [
        # Balanced shoulder/hip with a defined waist -> hourglass.
        (90.0, 65.0, 90.0, BodyShape.HOURGLASS),
        # Balanced but undefined waist -> rectangle.
        (90.0, 85.0, 90.0, BodyShape.RECTANGLE),
        # Hips dominate -> pear, regardless of waist definition.
        (85.0, 68.0, 105.0, BodyShape.PEAR),
        (85.0, 95.0, 105.0, BodyShape.PEAR),
        # Shoulders dominate with a defined waist -> inverted triangle.
        (110.0, 70.0, 88.0, BodyShape.INVERTED_TRIANGLE),
        # Shoulders dominate with an undefined waist -> apple.
        (110.0, 105.0, 88.0, BodyShape.APPLE),
    ],
)
def test_classify(shoulder: float, waist: float, hip: float, expected: BodyShape) -> None:
    assert classify(Proportions(shoulder, waist, hip)) is expected


def test_classification_is_scale_invariant() -> None:
    """Only ratios matter, so unit choice and camera distance must not change the label."""
    small = Proportions(90.0, 65.0, 90.0)
    large = Proportions(180.0, 130.0, 180.0)
    assert classify(small) is classify(large)


def test_rejects_nonpositive_measurements() -> None:
    with pytest.raises(ValueError, match="waist must be positive"):
        Proportions(90.0, 0.0, 90.0)


def test_borderline_measurements_score_lower_confidence() -> None:
    """A body sitting on the waist threshold could flip label with a small angle change.

    That is exactly the case the UI must surface for user confirmation, so it has to
    score below a clearly-defined body.
    """
    borderline = Proportions(90.0, 90.0 * 0.75, 90.0)
    decisive = Proportions(90.0, 55.0, 90.0)
    assert confidence_from_ratios(borderline) < confidence_from_ratios(decisive)


def test_confidence_stays_in_range() -> None:
    for p in (Proportions(90, 65, 90), Proportions(1, 1, 1), Proportions(200, 50, 60)):
        assert 0.0 <= confidence_from_ratios(p) <= 1.0
