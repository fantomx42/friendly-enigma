# Would SQLite Help the Storage Layer? — Advisory

Assessment of whether SQLite (and its variants/cousins) would help Wheeler Memory's
persistence layer, grounded in `wheeler_memory/storage.py` + `warming.py` and the
live runtime data under `~/.wheeler_memory/` as of 2026-05-30.

**Status:** Advice only — no migration performed. This paper records the
recommendation for a future decision.

---

## TL;DR

**Yes for the metadata + edge layer. No for the recall similarity scan.** There are
two separate cost centers and SQLite only fixes one. Don't let the "move it all to a
database" instinct merge them. None of the fancier SQLite variants
(Turso/D1/LiteFS/Litestream/DuckDB) is a better *base* for a single-user, local
agent — the one idea worth borrowing is vector search, best obtained locally via the
`sqlite-vec` extension on stock SQLite.

## The two cost centers

### 1. Metadata / graph layer → SQLite is the right fix

All hot state lives in monolithic JSON files read-whole and rewritten-whole:

- `chunks/science/index.json` — **16.4 MB, 10,438 entries**, and
  `storage.py:_bump_recalled_memories` **rewrites it on every recall** just to bump
  `hit_count`/`last_accessed`. ~16 MB write amplification per counter bump.
- `associations.json` — edge graph + warmth; loaded whole, mutated, rewritten on
  every store and every warmth spread (`warming.py`).
- `cross_chunk_edges.json` — **640 KB and growing**, gets a **full linear scan** per
  multi-chunk recall (`warming.py:propagate_warmth_cross_chunk`).

SQLite replaces each with indexed access:

- `SELECT ... WHERE hex_key = ?` instead of parsing 16 MB of JSON.
- `UPDATE memories SET hit_count = hit_count + 1 WHERE uuid = ?` instead of a
  full-file rewrite.
- Indexed edge joins instead of full-file scans.

The data model maps onto ~3 tables almost mechanically: `memories` (the `index.json`
fields), `edges` (`associations.json` adjacency + `source`/`weight`/`decay_count`,
absorbing the cross-chunk file too), `warmth`. Concurrency/locking comes for free,
replacing the hand-rolled `fcntl` `.lock` files.

### 2. Recall similarity scan → SQLite does NOT help

`storage.py:_score_item` does `np.load(attractor_path, mmap_mode="r")` **once per
stored attractor in scope** + Pearson correlation — up to 10,438 file opens for a
single `science` query, and recall searches *all* chunks, not just the routed one
(`storage.py:270-274`). That's an O(N) similarity scan. SQLite won't touch it. The
right tool there is a **vector index** — see the recommended stack below.

## You already started this

`~/.wheeler_memory/wheeler.db` is a real SQLite file with a `memories` table
(`uuid, key, blob_path, hit_count, stability, confidence, created_at,
last_accessed`). It holds **one stale row** and **nothing in the codebase references
it** — an abandoned prototype from before the JSON+npy layout. A past iteration
already chose this direction and didn't finish.

## SQLite variants evaluated

Wheeler is single-user, single-machine, local Python over `~/.wheeler_memory/`
(incl. a 3 GB local `.npz`). It is NOT a SaaS/edge/multi-node app, so most of these
solve problems Wheeler doesn't have.

| Option | Verdict | Why |
|--------|---------|-----|
| **Turso / libSQL** | ⚠️ Only for vector search | Native vector search hits Wheeler's real bottleneck, but its headline value (edge replication, multi-tenancy, cloud sync) is dead weight here. `sqlite-vec` gives the same capability on stock SQLite. |
| **Cloudflare D1** | ❌ Wrong shape | Runs in Cloudflare Workers, remote, size-capped, per-query network latency. Wheeler is a local process touching a 3 GB local file. |
| **LiteFS** | ❌ Irrelevant | Cross-node replication; Wheeler is one node. |
| **Litestream** | ✅ Optional add-on | Cheap continuous DB backup to S3, but only backs up the DB; the bulk (`.npy`/`.npz`, `context_ri_vectors.npz`) lives outside it. It's "SQLite + a sidecar," not a replacement. |
| **DuckDB** | ❌ primary / ✅ analytics lens | Columnar OLAP is bad at Wheeler's point lookups + counter updates. Excellent as a secondary read-only tool over the corpus (can query Parquet/Arrow/`.npy` directly). |

## Recommendation

**Stock SQLite remains the right base.** None of the variants is a better
replacement — they answer questions Wheeler isn't asking (distribute / scale / serve
many users).

Recommended stack and sequencing:

1. **Base:** stock SQLite for the catalog + edges. Start narrow — move `index.json`
   → a `memories` table first (biggest, lowest-risk win; revive the schema already
   in `wheeler.db`). Then fold `associations.json` + `cross_chunk_edges.json` into an
   `edges` table with an index on the endpoint keys.
2. **Recall:** add the **`sqlite-vec`** extension (a `vec0` virtual table for the
   attractor vectors) to address the O(N) recall scan — the local-first way to get
   the vector search that Turso markets. Pair it with the existing `att_mean`/
   `att_std` caching.
3. **Keep attractor `.npy` and brick `.npz` as external files** referenced by path
   (the legacy `blob_path` column already does this). Do not store them as BLOBs.
4. **Optional:** Litestream for off-machine DB backups; DuckDB for ad-hoc
   introspection of the corpus. Neither is the primary store.

## Caveats for implementation

- Migration is a one-time, reversible backfill from the JSON files; keep the JSON as
  a fallback until parity is verified.
- The CA attractors, bricks, pickles (`knowledge.pkl`, checkpoints), and
  `context_ri_vectors.npz` (3.19 GB) are **not** database candidates — leave them.
- No code currently flags these bottlenecks (no TODO/FIXME) and there are no tests
  guarding the access patterns — add them before/with a migration.

## Verification (when acted on)

- Benchmark a `science`-chunk recall before/after: wall-time and file-open count.
- Confirm recall result parity (same top_k for a fixed query set) JSON vs SQLite.
- Confirm `hit_count`/warmth updates survive a process restart.
