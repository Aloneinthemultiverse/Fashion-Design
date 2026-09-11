# ADR 0002 — Asynchronous job model

**Status:** Accepted · 2026-09-11

## Context

The architecture document's end-to-end flow budgets 45–90 seconds: body analysis,
retrieval, an ImageRAG generation cycle and virtual try-on. It describes this as a
single user request.

No production HTTP request survives 90 seconds intact. Load balancers and reverse
proxies default to 30–60 second idle timeouts, browsers abandon requests, and a client
retry on timeout duplicates the entire expensive pipeline — including the GPU work and
the metered VLM calls.

Retrofitting asynchrony is expensive: it changes every route signature, the client
contract, and the error model. Doing it later would mean rewriting work already done.

## Decision

Submit-and-poll from the first endpoint:

```
POST /recommendations      -> 202 Accepted, { job_id }
GET  /recommendations/{id} -> { status, stages[], partial_results }
```

Each pipeline stage is a separate worker task that publishes its result as soon as it
completes. The UI renders stages as they land.

## Consequences

- The user sees body analysis at ~2s and recommendations at ~10s rather than a 90-second
  spinner, even though total wall-clock time is unchanged. Perceived latency improves
  substantially.
- Partial failure becomes expressible: try-on can fail while recommendations succeed,
  which is precisely the graceful degradation the architecture calls for.
- Costs a Redis dependency and a job-state model up front.
- Stage-level results are individually cacheable, which matters given the VLM quota.
