"""Body-shape classification from proportion ratios.

Deterministic rules, deliberately kept out of the VLM. The VLM is good at *measuring*
proportions from a photo; it is inconsistent at *naming* the resulting shape. Splitting
the two means the naming step is testable, auditable and identical for every user.

Thresholds follow conventional styling practice:
  shoulder/hip within ~5%  -> balanced top and bottom
  waist at least ~25% smaller than the larger of shoulder/hip -> defined waist
"""

from __future__ import annotations

from dataclasses import dataclass

from fashion.core.models import BodyShape

BALANCED_TOLERANCE = 0.05
DEFINED_WAIST_RATIO = 0.75


@dataclass(frozen=True, slots=True)
class Proportions:
    """Raw measurements in any consistent unit — only their ratios matter."""

    shoulder: float
    waist: float
    hip: float

    def __post_init__(self) -> None:
        for name in ("shoulder", "waist", "hip"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


def classify(p: Proportions) -> BodyShape:
    """Map proportions to a body shape.

    Order matters: the waist test runs first because a defined waist distinguishes
    hourglass from rectangle even when shoulder and hip are balanced.
    """
    widest = max(p.shoulder, p.hip)
    defined_waist = p.waist <= widest * DEFINED_WAIST_RATIO
    balanced = abs(p.shoulder - p.hip) <= widest * BALANCED_TOLERANCE

    if balanced:
        return BodyShape.HOURGLASS if defined_waist else BodyShape.RECTANGLE
    if p.hip > p.shoulder:
        return BodyShape.PEAR
    # Shoulders dominate. An undefined waist on a broad-shouldered frame reads as apple;
    # a defined one reads as inverted triangle.
    return BodyShape.INVERTED_TRIANGLE if defined_waist else BodyShape.APPLE


def confidence_from_ratios(p: Proportions) -> float:
    """How decisive the classification is.

    Measurements sitting right on a threshold produce a shape label that would flip with
    a small change in camera angle, so they score low and trigger the user-confirmation
    prompt in the UI.
    """
    widest = max(p.shoulder, p.hip)
    waist_margin = abs(p.waist - widest * DEFINED_WAIST_RATIO) / widest
    balance_margin = abs(abs(p.shoulder - p.hip) / widest - BALANCED_TOLERANCE)
    # Both margins are distances from a decision boundary; the smaller one dominates.
    decisiveness = min(waist_margin, balance_margin)
    return round(min(1.0, 0.5 + decisiveness * 4), 3)
