"""Domain models.

Pure data with validation. No I/O, no framework imports — everything here must be
constructible in a test without touching the network or disk.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BodyShape(StrEnum):
    HOURGLASS = "hourglass"
    PEAR = "pear"
    RECTANGLE = "rectangle"
    APPLE = "apple"
    INVERTED_TRIANGLE = "inverted_triangle"


class Build(StrEnum):
    SLIM = "slim"
    ATHLETIC = "athletic"
    CURVY = "curvy"
    MUSCULAR = "muscular"


class HeightBand(StrEnum):
    PETITE = "petite"
    AVERAGE = "average"
    TALL = "tall"


class Culture(StrEnum):
    ETHNIC = "ethnic"
    WESTERN = "western"
    FUSION = "fusion"


class Occasion(StrEnum):
    CASUAL = "casual"
    FORMAL = "formal"
    PARTY = "party"
    WEDDING = "wedding"
    FESTIVE = "festive"


class Silhouette(StrEnum):
    A_LINE = "a_line"
    STRAIGHT = "straight"
    FIT_AND_FLARE = "fit_and_flare"
    BODYCON = "bodycon"
    EMPIRE = "empire"
    MERMAID = "mermaid"
    OVERSIZED = "oversized"


class Neckline(StrEnum):
    V_NECK = "v_neck"
    ROUND = "round"
    BOAT = "boat"
    SWEETHEART = "sweetheart"
    HALTER = "halter"
    COLLARED = "collared"
    OFF_SHOULDER = "off_shoulder"


class WaistEmphasis(StrEnum):
    HIGH = "high"
    NATURAL = "natural"
    DROPPED = "dropped"
    NONE = "none"


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)


class BodyMetrics(Frozen):
    """Body geometry inferred from a photo.

    `shape` is a *prior*, not a verdict — single-photo inference is corrupted by pose,
    camera angle and clothing, so the user is given the chance to correct it. `confidence`
    and `user_confirmed` exist so downstream stages can tell an guess from a fact.
    """

    shape: BodyShape
    build: Build
    height_band: HeightBand
    shoulder_waist_ratio: float | None = Field(default=None, gt=0, lt=5)
    waist_hip_ratio: float | None = Field(default=None, gt=0, lt=5)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    user_confirmed: bool = False

    @property
    def is_trustworthy(self) -> bool:
        return self.user_confirmed or self.confidence >= 0.75


class CelebrityProfile(Frozen):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    region: str = Field(description="indian | western")
    shape: BodyShape
    build: Build
    height_band: HeightBand
    style_tags: tuple[str, ...] = ()
    updated_at: datetime | None = None


class OutfitItem(Frozen):
    """One labelled outfit worn by one celebrity.

    `source` and `license` are mandatory and recorded at ingest time. Retrofitting
    provenance onto an existing image corpus is effectively impossible, and without it
    the dataset can never be published or shipped.
    """

    id: str = Field(min_length=1)
    celebrity_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)

    garment_type: str = Field(min_length=1)
    silhouette: Silhouette
    neckline: Neckline
    waist_emphasis: WaistEmphasis
    culture: Culture
    occasion: Occasion
    colors: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    fabric: str | None = None
    style_tags: tuple[str, ...] = ()

    caption: str = Field(default="", description="Dense VLM caption; embedded into cap_vec.")
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    captured_at: datetime | None = None


class UserQuery(Frozen):
    """What the user asked for. Covers all three modes from the architecture doc."""

    text: str = ""
    occasion: Occasion | None = None
    culture: Culture | None = None
    celebrity_name: str | None = None
    style_tags: tuple[str, ...] = ()
    top_k: int = Field(default=10, ge=1, le=100)

    @property
    def mode(self) -> str:
        return "celebrity" if self.celebrity_name else "body_match"


class Recommendation(Frozen):
    outfit: OutfitItem
    score: float
    rationale: str = ""
    adjustments: tuple[str, ...] = ()
    matched_on: tuple[str, ...] = Field(
        default=(), description="Which retrieval channels surfaced this item."
    )


class MissingConcept(Frozen):
    """One gap found by the VLM, plus the dense caption used to retrieve a fix.

    Per ImageRAG (arXiv 2502.09411), retrieving on a dense caption outperforms retrieving
    on the bare concept name or the original prompt — so both fields are carried.
    """

    concept: str = Field(min_length=1)
    retrieval_caption: str = Field(min_length=1)

    @model_validator(mode="after")
    def _caption_is_denser_than_concept(self) -> MissingConcept:
        if len(self.retrieval_caption) < len(self.concept):
            raise ValueError("retrieval_caption must be at least as detailed as concept")
        return self
