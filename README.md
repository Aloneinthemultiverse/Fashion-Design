# Fashion — Cross-Cultural Outfit Recommendation

Takes a photo, infers body geometry, and recommends outfits from a labelled celebrity
wardrobe worn by people with the same geometry — then explains why each one works.

Matching is done on **body shape, not ethnicity**. An A-line anarkali flatters a pear
frame for the same structural reasons an A-line sundress does; that equivalence is what
lets the system bridge Western and Indian wardrobes.

## Quick start

```bash
uv sync --extra dev
uv run pytest
```

The full test suite runs with **no API key, no Docker and no network** — the default
providers are a deterministic fake VLM, a fake embedder and an in-memory vector store.
Real providers are opt-in via `.env` (copy `.env.example`).

## Architecture

Ports and adapters. Every expensive, GPU-bound or legally-constrained capability sits
behind a `Protocol` in `src/fashion/ports/`, so the free implementation and the real one
are interchangeable:

| Port | Free default | Real option |
|---|---|---|
| `VisionModel` | `vision_fake` | Gemini 2.5 Flash (free tier) |
| `Embedder` | `embed_fake` | OpenCLIP ViT-B-32 (local CPU) |
| `VectorStore` | `store_memory` | Qdrant |
| `GenerationProvider` | `gen_null` | SDXL + IP-Adapter on a Colab T4 |
| `TryOnProvider` | `gen_null` | IDM-VTON on the same worker |
| `ImageSource` | Wikimedia Commons | Instagram (see below) |

`src/fashion/core/` is pure domain logic — no I/O, no framework imports. Body-shape
classification lives there rather than in the VLM prompt: the VLM is good at *measuring*
proportions and inconsistent at *naming* the resulting shape, so the naming step is kept
deterministic, testable and identical for every user.

### Retrieval

One collection, two named vectors per outfit, plus a structured payload:

- `img_vec` — CLIP **image** tower. Answers image-to-image queries and, since CLIP shares
  one space across towers, cross-modal text-to-image queries.
- `cap_vec` — CLIP **text** tower over a dense VLM caption. Answers text-to-text queries,
  which resolve fine attributes (neckline, silhouette, fabric) more sharply.
- `payload` — body shape, build, height band, culture, occasion, colours, tags. Applied
  as a **hard pre-filter**: body match is a constraint, not a score term.

Channels are combined with Reciprocal Rank Fusion, which needs no score calibration
across the two incomparable vector spaces.

### ImageRAG

Implements [arXiv 2502.09411](https://arxiv.org/abs/2502.09411). The load-bearing detail
is that the VLM writes a **dense caption per missing concept** and retrieval runs on that
caption — retrieving on the bare concept name or the original prompt measurably
underperforms.

## Data sourcing and licensing

Outfit records carry mandatory `source` and `license` fields, recorded at ingest.
Provenance cannot be retrofitted onto an image corpus, and without it the dataset can
never be published.

Two `ImageSource` adapters ship:

- **Wikimedia Commons** — real celebrity photographs with explicit CC/public-domain
  licences and Wikidata metadata. Safe to redistribute. This is the default.
- **Instagram** — richer and more current, but scraping it violates Instagram's Terms of
  Service, risks rate-limiting or banning the account and IP used, and yields
  copyrighted images that cannot be redistributed. It requires your own authenticated
  session and is **not run by default**. Use it only where you have the rights to.

## Status

| Phase | State |
|---|---|
| P0 — foundation: ports, domain logic, fakes, CI | done |
| P1 — image sources, 1,995-celebrity roster, resumable ingest | done |
| P2 — dual-vector index + hybrid retrieval (+ Qdrant) | done |
| P3 — ImageRAG generation loop | done |
| P2 tail — async API, worker, Streamlit UI | next |
| P4 — virtual try-on | pending |
| P5 — production hardening | pending |

132 tests, ruff and mypy --strict clean.

### What is real today

`data/seed/celebrities.jsonl` holds 1,995 real celebrities (1,000 Indian, 598
American, 397 British) fetched from Wikidata, each with a Commons-hosted image.
32 of them are labelled end-to-end and indexed, and retrieval returns ranked,
explained recommendations over that corpus.

### What still needs you

1. **A Gemini API key.** The 32 labelled items were produced by the *fake* VLM, so
   their garment tags are meaningless. Get a free key at
   [aistudio.google.com/apikey](https://aistudio.google.com/apikey), put it in `.env`,
   then re-run ingest against a clean `labels.jsonl` for real labels.
2. **Time, for the full corpus.** ~1,400 images/day fits the free tier, so labelling
   all 1,995 takes about two days of wall-clock. Ingest is resumable, so run it in
   chunks.
3. **A decision on Instagram.** The adapter is written and tested but never runs by
   default — see `docs/adr/0003`.
4. **A decision on generation.** Everything through ImageRAG works against the null
   provider; producing actual images needs the Colab worker (P3 hardware, not yet
   written).
