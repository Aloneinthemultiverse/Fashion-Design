"""Response schemas for outfit tagging.

Kept separate from the adapter because the enum lists are generated from the domain
models: if someone adds a silhouette to `core.models`, the schema the VLM is constrained
to must widen with it, and having that derivation in one place makes the coupling
obvious rather than a pair of lists that silently drift apart.
"""

from __future__ import annotations

from typing import Any

from fashion.core.models import (
    Culture,
    Neckline,
    Occasion,
    Silhouette,
    WaistEmphasis,
)

OUTFIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "garment_type": {"type": "string"},
        "silhouette": {"type": "string", "enum": [s.value for s in Silhouette]},
        "neckline": {"type": "string", "enum": [n.value for n in Neckline]},
        "waist_emphasis": {"type": "string", "enum": [w.value for w in WaistEmphasis]},
        "culture": {"type": "string", "enum": [c.value for c in Culture]},
        "occasion": {"type": "string", "enum": [o.value for o in Occasion]},
        "colors": {"type": "array", "items": {"type": "string"}},
        "patterns": {"type": "array", "items": {"type": "string"}},
        "fabric": {"type": "string"},
        "style_tags": {"type": "array", "items": {"type": "string"}},
        "full_body_visible": {"type": "boolean"},
        "outfit_clearly_visible": {"type": "boolean"},
    },
    "required": [
        "garment_type",
        "silhouette",
        "neckline",
        "waist_emphasis",
        "culture",
        "occasion",
        "colors",
        "full_body_visible",
        "outfit_clearly_visible",
    ],
}

BODY_FIELDS: dict[str, Any] = {
    "shoulder_width": {"type": "number"},
    "waist_width": {"type": "number"},
    "hip_width": {"type": "number"},
    "build": {"type": "string", "enum": ["slim", "athletic", "curvy", "muscular"]},
    "height_band": {"type": "string", "enum": ["petite", "average", "tall"]},
    "full_body_visible": {"type": "boolean"},
}

# One schema covering body measurement, outfit tagging and captioning. Requesting all
# three in a single response is what keeps a 2,000-image labelling pass inside roughly
# one day of free quota instead of four.
COMBINED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "body": {
            "type": "object",
            "properties": BODY_FIELDS,
            "required": list(BODY_FIELDS),
        },
        "outfit": OUTFIT_SCHEMA,
        "caption": {"type": "string"},
    },
    "required": ["body", "outfit", "caption"],
}

COMBINED_PROMPT = """Analyse this photograph of a person and return three things.

1. body -- measure widths in consistent arbitrary units as seen in the image:
   shoulder_width (widest across shoulders), waist_width (narrowest of the torso),
   hip_width (widest across hips). Use one scale for all three; only ratios are used.
   Do NOT classify the body shape; report measurements only.
   Also give build, height_band, and whether the full body is in frame.

2. outfit -- describe the clothing using the schema. garment_type should be specific
   where a specific name exists (saree, lehenga, anarkali, sharara, kurta, gown), not a
   generic word like "outfit". culture is "ethnic" for Indian traditional wear,
   "western" for Western garments, "fusion" for deliberate blends. Set
   outfit_clearly_visible to false if the clothing is obscured, heavily cropped, or the
   shot is a face close-up.

3. caption -- one dense factual paragraph describing only the clothing: garment type,
   silhouette, neckline, waist placement, colours, patterns, fabric and likely occasion.

Describe clothing and proportions only. Do not identify or describe the person's face.
"""

OUTFIT_PROMPT = """Describe the clothing worn in this image using the given schema.

garment_type should be the specific name where one exists -- saree, lehenga, anarkali,
sharara, kurta, gown, blazer -- not a generic word like "outfit".

culture: "ethnic" for Indian traditional wear, "western" for Western garments, "fusion"
for deliberate blends such as a crop top with a lehenga skirt.

Set outfit_clearly_visible to false if the clothing is obscured, heavily cropped, or the
image is a close-up of the face. Set full_body_visible to false if the legs or hem are
not in frame.

Describe only clothing. Do not describe or identify the person.
"""
