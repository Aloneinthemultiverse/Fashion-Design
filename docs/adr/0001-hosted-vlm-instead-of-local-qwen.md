# ADR 0001 — Hosted VLM instead of local Qwen2-VL

**Status:** Accepted · 2026-09-11

## Context

The architecture document specifies Qwen2-VL-2B running locally, justified as "small,
fast, free". That reasoning assumes a GPU. Development and initial deployment target a
CPU-only Windows laptop, where a 2B vision model takes roughly 30–60 seconds per image.

The pipeline calls the VLM several times per user request (body analysis, gap analysis,
rationale) and once per image during dataset labelling. At CPU speeds the labelling pass
alone would take hours and a single user request would exceed any reasonable timeout.

## Decision

Use **Gemini 2.5 Flash** via the Google AI Studio free tier as the default `VisionModel`
adapter. Keep a local Qwen2-VL adapter behind the same port as a fallback.

The project's own problem statement already names Gemini, so this aligns the
implementation with the stated design rather than departing from it.

## Consequences

- Body analysis drops from ~45s to ~1–2s. The pipeline becomes interactive.
- Introduces a network dependency and a quota (~1,500 requests/day, no credit card).
  Mitigated by content-hash caching of every VLM response, so re-labelling and repeated
  queries cost nothing.
- Images are sent to a third party. User photos are processed in memory and not
  persisted by default; this must be stated in the privacy notice before any public use.
- The local Qwen2-VL adapter remains the escape hatch if the quota, the terms, or the
  privacy position become unacceptable. Because it sits behind `VisionModel`, switching
  is a configuration change.
