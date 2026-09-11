# ADR 0003 — Image sourcing and licensing

**Status:** Accepted · 2026-09-11

## Context

The system needs celebrity outfit images with body-type and garment metadata. No public
dataset provides both: DeepFashion and Fashionpedia have garment labels but no celebrity
identity or body-shape labels, and neither covers Indian ethnic wear well.

The architecture document specifies Instaloader against Instagram. That carries three
distinct problems, which are worth separating because they have different severities:

1. **Terms of service.** Automated scraping violates Instagram's ToS. The practical risk
   is rate-limiting or banning the account and IP used. As of 2026 most profile browsing
   also requires an authenticated session, so this cannot be done anonymously.
2. **Copyright.** Celebrity photographs are owned by the photographer or agency. They can
   generally be viewed, but not redistributed or served from our own infrastructure —
   which is exactly what a recommendation gallery does. This is the constraint that would
   block a public launch, and it does not go away with rate limiting or proxies.
3. **Fragility.** Scrapers break whenever the target changes its markup or API.

Problem 2 is the binding one and is independent of how the images are obtained.

## Decision

Put image acquisition behind an `ImageSource` port and ship two adapters:

- **Wikimedia Commons (default).** Real celebrity photographs carrying explicit
  CC-BY / CC-BY-SA / public-domain licences, with structured metadata available from
  Wikidata. Redistributable with attribution. This populates the seed index.
- **Instagram (opt-in, not run by default).** Implemented as requested, but requires the
  operator's own authenticated session and an explicit flag. Rate-limited and cached.

`OutfitItem.source` and `OutfitItem.license` are **mandatory** fields, recorded at ingest
time for every item regardless of adapter.

## Consequences

- The seed dataset is smaller and less current than Instagram would give, and Commons
  photographs are often red-carpet or event shots rather than curated outfit posts.
  Accepted: a smaller lawful corpus is more valuable than a larger unusable one, because
  the retrieval architecture is identical either way and the corpus can grow.
- Mandatory provenance means every item can be traced and, if necessary, removed. This is
  not retrofittable — an image corpus without per-item licence records cannot be audited
  after the fact, which is why the fields are required from the first commit.
- Whether this ships publicly, internally, or as an academic project remains an open
  decision, but the data layer no longer forecloses the choice.
