"""Cross-cultural styling rules: Western body geometry -> Indian ethnic wear.

The premise of the whole system is that a garment flatters a *shape*, not an ethnicity.
An Anarkali that works on an hourglass frame works on an hourglass frame regardless of
whose it is. So matching is done on geometry and these rules encode what each shape
wants from a garment's cut.

Kept as explicit data rather than delegated to the VLM for three reasons: the rules are
stable styling knowledge that does not need re-deriving per request, they cost nothing
to evaluate, and they are auditable — a stylist can read and correct this table, which
matters when the output is advice about someone's appearance.
"""

from __future__ import annotations

from dataclasses import dataclass

from fashion.core.models import (
    BodyShape,
    Neckline,
    OutfitItem,
    Silhouette,
    WaistEmphasis,
)


@dataclass(frozen=True, slots=True)
class ShapeGuidance:
    goal: str
    silhouettes: frozenset[Silhouette]
    necklines: frozenset[Neckline]
    waist: frozenset[WaistEmphasis]


GUIDANCE: dict[BodyShape, ShapeGuidance] = {
    BodyShape.HOURGLASS: ShapeGuidance(
        goal="preserve the natural waist definition rather than hiding it",
        silhouettes=frozenset({Silhouette.FIT_AND_FLARE, Silhouette.BODYCON, Silhouette.MERMAID}),
        necklines=frozenset({Neckline.V_NECK, Neckline.SWEETHEART, Neckline.ROUND}),
        waist=frozenset({WaistEmphasis.NATURAL, WaistEmphasis.HIGH}),
    ),
    BodyShape.PEAR: ShapeGuidance(
        goal="draw attention upward and skim the hip rather than cling to it",
        silhouettes=frozenset({Silhouette.A_LINE, Silhouette.FIT_AND_FLARE, Silhouette.EMPIRE}),
        necklines=frozenset({Neckline.BOAT, Neckline.OFF_SHOULDER, Neckline.SWEETHEART}),
        waist=frozenset({WaistEmphasis.HIGH, WaistEmphasis.NATURAL}),
    ),
    BodyShape.RECTANGLE: ShapeGuidance(
        goal="create waist definition that the frame does not supply on its own",
        silhouettes=frozenset({Silhouette.FIT_AND_FLARE, Silhouette.A_LINE, Silhouette.MERMAID}),
        necklines=frozenset({Neckline.SWEETHEART, Neckline.HALTER, Neckline.V_NECK}),
        waist=frozenset({WaistEmphasis.NATURAL, WaistEmphasis.HIGH}),
    ),
    BodyShape.APPLE: ShapeGuidance(
        goal="lengthen the torso and move structure away from the midsection",
        silhouettes=frozenset({Silhouette.EMPIRE, Silhouette.A_LINE, Silhouette.STRAIGHT}),
        necklines=frozenset({Neckline.V_NECK, Neckline.COLLARED}),
        waist=frozenset({WaistEmphasis.HIGH, WaistEmphasis.NONE}),
    ),
    BodyShape.INVERTED_TRIANGLE: ShapeGuidance(
        goal="add volume below the waist to balance a broader shoulder line",
        silhouettes=frozenset({Silhouette.A_LINE, Silhouette.FIT_AND_FLARE}),
        necklines=frozenset({Neckline.V_NECK, Neckline.ROUND}),
        waist=frozenset({WaistEmphasis.NATURAL, WaistEmphasis.DROPPED}),
    ),
}

# Indian ethnic garments whose traditional cut already satisfies a shape's guidance.
# Used to bias retrieval toward ethnic wear that needs no adaptation.
NATURAL_FITS: dict[BodyShape, frozenset[str]] = {
    BodyShape.HOURGLASS: frozenset({"saree", "lehenga", "churidar kurta"}),
    BodyShape.PEAR: frozenset({"anarkali", "a-line kurta", "cape lehenga"}),
    BodyShape.RECTANGLE: frozenset({"saree", "sharara", "peplum kurta"}),
    BodyShape.APPLE: frozenset({"anarkali", "empire-line kurta", "kaftan"}),
    BodyShape.INVERTED_TRIANGLE: frozenset({"lehenga", "sharara", "palazzo suit"}),
}


def fit_score(outfit: OutfitItem, shape: BodyShape) -> float:
    """How well an outfit's cut suits a shape, in [0, 1].

    Three equally-weighted axes — silhouette, neckline, waist placement — because no one
    of them dominates in practice; a perfect silhouette with the wrong waist placement is
    about as wrong as the reverse.
    """
    g = GUIDANCE[shape]
    hits = (
        (outfit.silhouette in g.silhouettes),
        (outfit.neckline in g.necklines),
        (outfit.waist_emphasis in g.waist),
    )
    return round(sum(hits) / len(hits), 3)


def adjustments(outfit: OutfitItem, shape: BodyShape) -> tuple[str, ...]:
    """Concrete changes that would make a near-miss outfit work for this shape.

    Only reports axes that actually miss, so an empty tuple means "wear as-is".
    """
    g = GUIDANCE[shape]
    out: list[str] = []
    if outfit.silhouette not in g.silhouettes:
        want = ", ".join(sorted(s.value.replace("_", "-") for s in g.silhouettes))
        out.append(f"Prefer a {want} cut over {outfit.silhouette.value.replace('_', '-')}.")
    if outfit.neckline not in g.necklines:
        want = ", ".join(sorted(n.value.replace("_", "-") for n in g.necklines))
        out.append(f"Swap the {outfit.neckline.value.replace('_', '-')} neckline for {want}.")
    if outfit.waist_emphasis not in g.waist:
        want = ", ".join(sorted(w.value for w in g.waist))
        out.append(f"Move the waist to {want} placement.")
    return tuple(out)


def explain(outfit: OutfitItem, shape: BodyShape) -> str:
    """A grounded one-line rationale.

    Built from the outfit's structured attributes rather than free VLM prose so it cannot
    describe a garment detail that is not actually recorded for this item.
    """
    g = GUIDANCE[shape]
    score = fit_score(outfit, shape)
    lead = {3: "works as-is", 2: "mostly works", 1: "needs adjusting", 0: "is a poor match"}[
        round(score * 3)
    ]
    return (
        f"This {outfit.garment_type} {lead} for a {shape.value.replace('_', ' ')} frame: "
        f"the aim is to {g.goal}, and it has a "
        f"{outfit.silhouette.value.replace('_', '-')} silhouette with a "
        f"{outfit.neckline.value.replace('_', '-')} neckline at "
        f"{outfit.waist_emphasis.value} waist."
    )
