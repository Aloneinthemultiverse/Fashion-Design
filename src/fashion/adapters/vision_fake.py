"""Deterministic fake VLM.

Derives its answers from a hash of the input bytes, so the same image always yields the
same result and different images yield different ones. That is enough to exercise every
downstream stage -- ranking, filtering, the ImageRAG loop -- without a network call, and
it keeps CI free and offline.
"""

from __future__ import annotations

import hashlib

from fashion.core.models import (
    BodyMetrics,
    BodyShape,
    Build,
    Culture,
    HeightBand,
    MissingConcept,
    Neckline,
    Occasion,
    OutfitItem,
    Silhouette,
    WaistEmphasis,
)


def _seed(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def _pick[T](data: bytes, options: list[T], salt: int = 0) -> T:
    return options[(_seed(data) + salt) % len(options)]


class FakeVisionModel:
    """Implements the VisionModel protocol without any I/O."""

    def analyze_body(self, image: bytes) -> BodyMetrics:
        s = _seed(image)
        return BodyMetrics(
            shape=list(BodyShape)[s % len(BodyShape)],
            build=list(Build)[(s >> 3) % len(Build)],
            height_band=list(HeightBand)[(s >> 6) % len(HeightBand)],
            shoulder_waist_ratio=round(1.1 + (s % 40) / 100, 3),
            waist_hip_ratio=round(0.65 + (s % 30) / 100, 3),
            confidence=round(0.5 + (s % 50) / 100, 3),
        )

    def tag_outfit(self, image: bytes) -> dict[str, object]:
        return {
            "garment_type": _pick(image, ["saree", "anarkali", "lehenga", "gown", "kurta"]),
            "silhouette": _pick(image, list(Silhouette), 1),
            "neckline": _pick(image, list(Neckline), 2),
            "waist_emphasis": _pick(image, list(WaistEmphasis), 3),
            "culture": _pick(image, list(Culture), 4),
            "occasion": _pick(image, list(Occasion), 5),
            "colors": ("crimson", "gold"),
            "patterns": ("embroidered",),
            "fabric": "silk",
            "style_tags": ("traditional", "glam"),
        }

    def caption_outfit(self, image: bytes) -> str:
        garment = _pick(image, ["saree", "anarkali", "lehenga", "gown", "kurta"])
        silhouette = _pick(image, list(Silhouette), 1)
        neckline = _pick(image, list(Neckline), 2)
        occasion = _pick(image, list(Occasion), 5)
        return (
            f"A silk {garment} in crimson and gold with a {silhouette.value} silhouette "
            f"and {neckline.value} neckline, suited to {occasion.value} wear."
        )

    def find_gaps(self, generated: bytes, prompt: str) -> tuple[MissingConcept, ...]:
        # Report a gap on odd-hashed inputs and none on even, so the ImageRAG loop's
        # iterate-then-converge behaviour is exercised by tests.
        if _seed(generated) % 2 == 0:
            return ()
        return (
            MissingConcept(
                concept="dupatta",
                retrieval_caption=(
                    "A sheer embroidered dupatta draped over one shoulder, "
                    "falling to knee length, in a contrasting colour."
                ),
            ),
        )

    def write_rationale(self, outfit: OutfitItem, metrics: BodyMetrics) -> str:
        return (
            f"The {outfit.garment_type} suits a {metrics.shape.value} frame "
            f"because its {outfit.silhouette.value} cut balances your proportions."
        )

    def expand_query(self, text: str, n: int = 3) -> tuple[str, ...]:
        """Deterministic rephrasings.

        The original always comes first so a caller that takes only the head still gets
        the user's actual words.
        """
        base = text.strip()
        if not base:
            return ()
        templates = (
            "{q}",
            "an outfit in the style of {q}",
            "{q}, flattering silhouette and neckline",
            "traditional interpretation of {q}",
        )
        return tuple(t.format(q=base) for t in templates[: max(1, n)])

    def find_missing_concepts(
        self, request: str, found: tuple[str, ...]
    ) -> tuple[MissingConcept, ...]:
        """Report a gap for any notable word in the request absent from the results.

        Crude but deterministic and genuinely input-dependent, so tests exercise both
        the gap-found and no-gap paths without a network call.
        """
        haystack = " ".join(found).casefold()
        gaps: list[MissingConcept] = []
        for word in ("dupatta", "saree", "lehenga", "anarkali", "embroidered", "pastel"):
            if word in request.casefold() and word not in haystack:
                gaps.append(
                    MissingConcept(
                        concept=word,
                        retrieval_caption=(
                            f"A garment prominently featuring {word}, shown full length "
                            f"in clear studio lighting against a plain background."
                        ),
                    )
                )
        return tuple(gaps)
