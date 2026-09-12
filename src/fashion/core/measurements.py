"""Numeric body measurements and range-based matching.

Replaces "which of five buckets is this person" with "what are their actual
proportions, and whose are close enough". Two reasons this is better:

* **It is measurable.** Widths come off a silhouette, so they can be computed locally
  with no API and no model guessing at a category. A category is an opinion; a ratio is
  an observation.
* **Buckets lose the information that matters.** Two people can both be "pear" while one
  has a 0.72 waist-hip ratio and the other 0.85 -- they need different cuts. And a person
  sitting between two buckets gets an arbitrary label that flips with camera angle.

Matching is a distance within a tolerance band rather than equality. Everything is
expressed as ratios so the result is invariant to camera distance, image resolution and
how much of the frame the person fills -- only proportion survives, which is the only
thing that transfers between two different photographs anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fashion.core.body import Proportions, classify
from fashion.core.models import BodyShape

# How far apart two ratios can be and still count as the same proportion. Derived from
# how much the same person's measured ratios move between photographs: pose and camera
# angle alone shift them by a few percent, so a tolerance below that would reject
# genuine matches, and one far above it would accept everyone.
DEFAULT_TOLERANCE = 0.08

# Relative importance when scoring a match. Waist-hip carries the most weight because it
# is what most styling advice actually turns on; height band the least, because it
# changes hem length rather than which silhouette suits the frame.
WEIGHTS = {
    "shoulder_hip": 1.0,
    "waist_hip": 1.2,
    "shoulder_waist": 1.0,
    "height_ratio": 0.5,
}


@dataclass(frozen=True, slots=True)
class BodyRatios:
    """Scale-invariant proportions.

    Stored as ratios rather than raw widths on purpose: raw pixel widths depend on how
    close the camera was, which carries no information about the body.
    """

    shoulder_hip: float
    waist_hip: float
    shoulder_waist: float
    # Torso length over shoulder width. Separates a long-torsoed frame from a compact
    # one at identical widths, which changes where a waistline should sit.
    height_ratio: float = 0.0

    def __post_init__(self) -> None:
        for name in ("shoulder_hip", "waist_hip", "shoulder_waist"):
            value = getattr(self, name)
            if not (0.0 < value < 10.0) or math.isnan(value):
                raise ValueError(f"{name} must be a positive finite ratio, got {value!r}")

    @classmethod
    def from_widths(
        cls, shoulder: float, waist: float, hip: float, height: float = 0.0
    ) -> BodyRatios:
        if min(shoulder, waist, hip) <= 0:
            raise ValueError("widths must be positive")
        return cls(
            shoulder_hip=round(shoulder / hip, 4),
            waist_hip=round(waist / hip, 4),
            shoulder_waist=round(shoulder / waist, 4),
            height_ratio=round(height / shoulder, 4) if height > 0 else 0.0,
        )

    @property
    def shape(self) -> BodyShape:
        """The bucket these ratios fall into.

        Kept as a derived view, not the source of truth: useful for explaining a match
        in words, never for deciding one.
        """
        return classify(Proportions(shoulder=self.shoulder_hip, waist=self.waist_hip, hip=1.0))

    def distance(self, other: BodyRatios) -> float:
        """Weighted relative difference. 0.0 is identical.

        Relative rather than absolute, so a 0.05 gap counts the same whether the ratios
        are near 0.7 or near 1.4.
        """
        total = 0.0
        weight_sum = 0.0
        for field, weight in WEIGHTS.items():
            a, b = getattr(self, field), getattr(other, field)
            if a <= 0 or b <= 0:
                # height_ratio is optional; skip rather than treat a missing value as a
                # mismatch, which would penalise every profile that lacks it.
                continue
            total += weight * abs(a - b) / max(a, b)
            weight_sum += weight
        return round(total / weight_sum, 4) if weight_sum else 1.0

    def matches(self, other: BodyRatios, tolerance: float = DEFAULT_TOLERANCE) -> bool:
        return self.distance(other) <= tolerance

    def similarity(self, other: BodyRatios) -> float:
        """Distance mapped to [0, 1], for ranking."""
        return round(max(0.0, 1.0 - self.distance(other)), 4)

    def band(self, tolerance: float = DEFAULT_TOLERANCE) -> dict[str, tuple[float, float]]:
        """The acceptable range around each ratio.

        Used to pre-filter candidates in the vector store, which can compare numbers but
        cannot evaluate `distance`. The band is deliberately wider than the true
        tolerance region -- it is a cheap prefilter, and the exact distance check still
        runs afterwards. Being slightly permissive here only costs a few extra
        candidates; being too tight would silently drop real matches.
        """
        out: dict[str, tuple[float, float]] = {}
        for field in WEIGHTS:
            value = getattr(self, field)
            if value <= 0:
                continue
            margin = value * tolerance * 1.5
            out[field] = (round(value - margin, 4), round(value + margin, 4))
        return out

    def describe(self) -> str:
        """Plain-language summary, for explaining a match to the user."""
        parts = [
            f"shoulders {self.shoulder_hip:.2f}x hips",
            f"waist {self.waist_hip:.2f}x hips",
        ]
        if self.height_ratio:
            parts.append(f"torso {self.height_ratio:.1f}x shoulder width")
        return ", ".join(parts)


def rank_by_similarity(
    target: BodyRatios,
    candidates: dict[str, BodyRatios],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    limit: int | None = None,
) -> list[tuple[str, float]]:
    """Candidates within tolerance, closest first.

    Returns only those inside the band. Falling back to "nearest anyway" would quietly
    hand someone a body that is not theirs, which is the failure this whole module
    exists to avoid -- an empty result is the honest answer.
    """
    scored = [
        (key, target.similarity(ratios))
        for key, ratios in candidates.items()
        if target.matches(ratios, tolerance)
    ]
    scored.sort(key=lambda row: (-row[1], row[0]))
    return scored[:limit] if limit else scored
