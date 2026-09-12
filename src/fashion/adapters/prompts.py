"""Prompts for the vision model.

Written for a proxy that cannot constrain output to a JSON schema, so the schema has to
be carried by the prompt itself and the model has to be told, firmly and more than once,
to return nothing else.

Three things the analysis prompt is built around:

* **Measure, do not classify.** The model is asked for widths, never for a body-shape
  name. Models are inconsistent at naming shapes and far steadier at comparing widths in
  a single image, and `fashion.core.body.classify` turns widths into a name
  deterministically. That keeps the naming testable and identical for every user.
* **Say when the photo cannot answer.** A cropped or angled shot yields confident
  nonsense unless the model is given an explicit way to report that, so
  `full_body_visible` and `outfit_clearly_visible` exist and the pipeline halves
  confidence when they are false.
* **Clothing, not people.** The system prompt forbids describing faces, identity or
  attractiveness. The product needs garment geometry; anything else is intrusion the
  user did not ask for.
"""

from __future__ import annotations

ANALYSIS_SYSTEM = """\
You are a garment-measurement instrument for a clothing recommendation system.

Rules you must follow exactly:

1. Return ONLY a single JSON object. No prose before or after it, no code fences, no
   explanation. The response is parsed by a program, not read by a person.
2. Measure proportions; do not name a body shape. Never output words like "pear",
   "hourglass", "apple", "rectangle" or "inverted triangle". A separate deterministic
   step assigns those names from your measurements.
3. Describe clothing and proportions only. Do not describe, identify, name or judge the
   person: no face, no ethnicity, no attractiveness, no age.
4. If the photograph cannot support a measurement, say so through the visibility flags
   rather than guessing. A flagged uncertainty is useful; a confident guess is harmful.
"""

ANALYSIS_PROMPT = """\
Analyse this photograph and return exactly this JSON object:

{
  "body": {
    "shoulder_width": <number>,
    "waist_width": <number>,
    "hip_width": <number>,
    "torso_length": <number>,
    "build": "slim" | "athletic" | "curvy" | "muscular",
    "height_band": "petite" | "average" | "tall",
    "wardrobe_suggestion": "menswear" | "womenswear" | "unisex",
    "full_body_visible": true | false
  },
  "outfit": {
    "garment_type": "<specific name>",
    "silhouette": "a_line" | "straight" | "fit_and_flare" | "bodycon"
                | "empire" | "mermaid" | "oversized",
    "neckline": "v_neck" | "round" | "boat" | "sweetheart" | "halter" | "collared" | "off_shoulder",
    "waist_emphasis": "high" | "natural" | "dropped" | "none",
    "culture": "ethnic" | "western" | "fusion",
    "wardrobe": "menswear" | "womenswear" | "unisex",
    "occasion": "casual" | "formal" | "party" | "wedding" | "festive",
    "colors": ["<colour>", ...],
    "patterns": ["<pattern>", ...],
    "fabric": "<fabric>",
    "style_tags": ["<tag>", ...],
    "outfit_clearly_visible": true | false,
    "full_body_visible": true | false
  },
  "caption": "<one dense paragraph describing only the clothing>"
}

Measurement guidance:
- shoulder_width: widest point across the shoulders.
- waist_width: narrowest point of the torso.
- hip_width: widest point across the hips.
- torso_length: shoulder line down to the hip line.
- wardrobe_suggestion: which wardrobe the clothing they are already wearing comes from.
  This is a starting point the user can change, not a judgement about them.
- Use one consistent arbitrary unit for all four. Only their ratios are used, so the
  scale does not matter, but they must be measured against each other in this image.

Field guidance:
- garment_type: the specific name where one exists -- saree, lehenga, anarkali, sharara,
  churidar, kurta, gown, blazer, sherwani -- never a generic word like "outfit".
- wardrobe: which garment tradition the clothing belongs to. A sherwani, kurta-pyjama
  or bandhgala is menswear; a saree, lehenga or anarkali is womenswear; a shirt, tee,
  blazer or jeans is unisex unless clearly cut for one. Judge the *garment*, not the
  wearer -- if the item is genuinely worn by anyone, say unisex.
- culture: "ethnic" for Indian traditional wear, "western" for Western garments,
  "fusion" for deliberate indo-western blends such as a crop top with a lehenga skirt,
  a saree worn with a shirt, or a dhoti paired with a jacket.
- outfit_clearly_visible: false if the clothing is obscured, heavily cropped, or the
  shot is a head-and-shoulders close-up.
- caption: front-load garment type, silhouette, neckline, waist and colour. It is
  truncated at 77 tokens downstream, so the least important detail goes last.

Return the JSON object and nothing else.
"""

GAP_PROMPT = """\
Compare this image against the request below.

Request: {request}

List every concept the request calls for that the image does not show. For each one,
write a "retrieval_caption": a rich visual description of that concept alone, as it
should appear, detailed enough to find a photograph of it. The caption must be
substantially longer and more specific than the concept name -- a bare noun retrieves
poorly, a described garment retrieves well.

Return ONLY this JSON, with an empty list if the image already satisfies the request:

{{"missing": [{{"concept": "<short name>", "retrieval_caption": "<rich description>"}}]}}
"""

MISSING_PROMPT = """\
A user asked for an outfit. These are the outfits found for them.

Request: {request}

Found:
{listing}

List the concepts the request calls for that none of the found outfits provide. For each,
write a "retrieval_caption": a rich visual description of that concept alone, detailed
enough to find a photograph of it, and substantially more specific than the concept name.

Judge only against what is listed. If the found outfits already satisfy the request,
return an empty list.

Return ONLY this JSON:

{{"missing": [{{"concept": "<short name>", "retrieval_caption": "<rich description>"}}]}}
"""

EXPAND_PROMPT = """\
Rewrite this outfit search as {n} differently-phrased searches that would retrieve
overlapping but not identical results. Vary the vocabulary, the level of detail, and
whether garment names are specific or general. Every rewrite must stay faithful to the
original intent.

Search: {query}

Return ONLY this JSON:

{{"queries": ["<rewrite>", "<rewrite>", "<rewrite>"]}}
"""

# Used when a Western body is matched to an Indian wardrobe: the model is asked how the
# garment should change, not merely whether it fits.
ADAPT_PROMPT = """\
A person with these proportions wants to wear the garment described below, adapted into
indo-western style.

Proportions: {proportions}
Garment: {garment}

Describe the adapted garment: what to keep from the original, what to change so it suits
these proportions, and how to blend Indian and Western elements. Be concrete about
silhouette, neckline and waist placement.

Return ONLY this JSON:

{{"adapted_description": "<a single vivid paragraph suitable as an image-generation prompt>",
  "kept": ["<element>", ...],
  "changed": ["<element and why>", ...]}}
"""
