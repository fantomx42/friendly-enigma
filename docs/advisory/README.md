# Advisories

Recommendation and decision papers for Wheeler Memory — "here is what we should do
and why." Each advisory weighs options against the project's actual constraints and
records a recommendation, so future work doesn't have to re-derive the reasoning.

**Advisories vs. reports.** Advisories are *recommendations*. For *empirical*
writeups — audits, probes, "here is what we measured" — see [`../reports/`](../reports/).

## Naming convention

`<topic>-YYYY-MM-DD.md` — matches `docs/reports/`. The date is when the advice was
given; an advisory reflects what was true on that date and may be superseded by a
later paper rather than edited in place.

## Suggested structure

1. `# Title` + a one-line dated provenance note (as in `docs/reports/`).
2. **TL;DR** — the verdict in one or two sentences.
3. The analysis / trade-offs.
4. A **Recommendation** and, where relevant, a sequenced path.
5. **Status** — e.g. `advice only, not yet acted on`.

## Index

| Advisory | Date | Status | Summary |
|----------|------|--------|---------|
| [SQLite storage evaluation](sqlite-storage-evaluation-2026-05-30.md) | 2026-05-30 | Advice only | Whether SQLite (and Turso/D1/LiteFS/Litestream/DuckDB) helps Wheeler's storage layer. Verdict: SQLite fixes the JSON metadata/edge churn, not the recall scan; recommended stack is stock SQLite + `sqlite-vec`. |
