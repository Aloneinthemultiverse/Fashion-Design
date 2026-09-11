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

Run the app:

```bash
uv sync --extra dev --extra sources --extra store --extra api --extra ui
uv run python scripts/fetch_roster.py --indian 1000 --american 600 --british 400
uv run python scripts/ingest.py --limit 40 --vision fake
uv run streamlit run src/fashion/ui/app.py
```

Or the API:

```bash
uv run uvicorn fashion.api.app:app --port 8000
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

All phases implemented.

| Phase | State |
|---|---|
| P0 — foundation: ports, domain logic, fakes, CI | done |
| P1 — image sources, 1,995-celebrity roster, resumable ingest | done |
| P2 — dual-vector index, hybrid retrieval, Qdrant | done |
| P2 tail — async API, RQ worker, Streamlit UI | done |
| P3 — ImageRAG loop + Colab GPU worker | done |
| P4 — virtual try-on stage | done |
| P5 — rate limiting, caching, feedback, metrics | done |

**183 tests**, ruff and mypy --strict clean.

## API

```
POST /recommendations      -> 202 { job_id, poll_url }
GET  /recommendations/{id} -> stage-by-stage progress and partial results
POST /feedback             -> 204
GET  /health               -> which backends are live
GET  /metrics              -> cache hit rate, quota burn, satisfaction
```

Submit-and-poll rather than a single synchronous call: the pipeline takes 45–90
seconds and no production HTTP request survives that (ADR 0002). Stages publish
results as they land, so body analysis appears at ~2s and recommendations at
~10s.

`SKIPPED` and `FAILED` are distinct stage states. A missing GPU is a
configuration choice the UI explains; a crash is not. Collapsing them would make
an outage look deliberate.

## What is real today

`data/seed/celebrities.jsonl` holds **1,995 real celebrities** (1,000 Indian, 598
American, 397 British) from Wikidata, each with a Commons-hosted image and a
recorded licence. 32 are labelled and indexed end-to-end; retrieval, the API and
the UI all run against them.

## What still needs you

1. **A Gemini API key.** The 32 labelled items came from the *fake* VLM, so their
   garment tags are meaningless — it labelled a man in a tuxedo as wearing a
   saree. Get a free key at
   [aistudio.google.com/apikey](https://aistudio.google.com/apikey), set
   `FASHION_VISION_PROVIDER=gemini`, delete `data/seed/labels.jsonl`, and re-run
   ingest. The UI shows a prominent warning until you do.
2. **Time, for the full corpus.** ~1,200 images/day fits inside the budget, so
   labelling all 1,995 takes about two days. Ingest is resumable.
3. **A Colab session,** if you want generated previews or try-on. Run
   `colab/worker.ipynb` and paste the printed URL into `.env`. Everything works
   without it — you just get real photographed outfits instead.
4. **A decision on Instagram.** The adapter is written and tested but never runs
   by default. See `docs/adr/0003`.

## Measuring retrieval quality

```bash
uv run python scripts/evaluate.py --embed openclip --probes
```

Unit tests cannot tell a semantically meaningful index from a meaningless one — the
fake embedder satisfies every structural property while encoding nothing. This
script separates the two concerns:

- **self-retrieval** — a wiring check any working index passes;
- **caption-to-image** — querying image vectors with text embeddings, which needs
  both a real dual encoder *and* truthful captions;
- **`--probes`** — hand-written queries verified against the actual photographs,
  which isolate the encoder from the captions.

Current readings on the 32-item seed corpus:

| | self-retrieval | caption-to-image (chance = 16.5) |
|---|---|---|
| `fake` | 32/32 | median rank 17.5 |
| `openclip` | 32/32 | median rank 15.5 |

Both sit at chance on caption-to-image, but for different reasons. The probes show
OpenCLIP is working correctly — "a man wearing a formal black tuxedo and bow tie"
returns the tuxedo photo, and "a person on a red carpet at a film festival" returns
three festival photographs it identified from pixels alone. **The captions are the
bottleneck**: they came from the fake VLM and describe garments the photos do not
contain, so a correct encoder has nothing to match. This number should move once
the corpus is relabelled with a real VLM, and is the cheapest way to confirm the
relabelling worked.

## Known limits

- Rate limiters and the result cache are in-process: correct for one node, wrong
  for several, where the effective limit would multiply per node.
- Single-photo body inference is the weakest link in the system. It is treated as
  a prior the user can correct, not a verdict, and `/metrics` reports
  satisfaction separately for confirmed readings so the two failure modes stay
  distinguishable.
- The try-on path uses SDXL inpainting with IP-Adapter conditioning rather than
  IDM-VTON proper, whose weights and repo layout are unstable.
