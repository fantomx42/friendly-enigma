# CLI Tools Reference

## Store a memory

```bash
wheeler-store "fix the python debug error"
# Chunk:    code (auto)
# State:    CONVERGED
# Ticks:    43
# Rotation: 0° (attempt 1)
# Time:     0.003s
# Memory stored successfully.

wheeler-store --chunk hardware "solder the GPIO header"   # explicit chunk
echo "piped input" | wheeler-store -                       # stdin
wheeler-store --embed "fuzzy memory"                       # store with semantic embedding
wheeler-store --encoder hippocampus "native encoding"      # explicit encoder choice
wheeler-store --encoder context "distributional semantics" # context-RI encoder
wheeler-store --salience high "critical insight"           # deep attractor (3000 iters, 1e-6 threshold)
wheeler-store --salience low "background note"             # fast store (200 iters, 5e-4 threshold)
```

## Recall memories

```bash
wheeler-recall "python bug"
# Rank  Similarity  Chunk        State        Ticks  Text
# ----------------------------------------------------------------------------------
# 1        0.0145  code         CONVERGED       43  fix the python debug error
# ...

wheeler-recall --chunk code "debug error"   # search specific chunk
wheeler-recall --top-k 10 "something"       # more results
wheeler-recall --embed "debugging issues"   # fuzzy semantic search
wheeler-recall --encoder blended "mixed approach"  # hippocampus + language wheeler blend
wheeler-recall --salience high "important query"  # more CA patience for query evolution
```

### Salience levels

The `--salience` flag controls how much computational attention a store or recall operation receives:

| Level | max_iters | threshold | Use case |
|-------|-----------|-----------|----------|
| `low` | 200 | 5e-4 | Bulk ingestion, background notes |
| `medium` | 1000 | 1e-4 | Default (omitting `--salience` is the same) |
| `high` | 3000 | 1e-6 | Important memories, deep attractors |

When omitted, salience defaults to `medium` (backwards compatible). For reconstruction, if no explicit salience is given, hot memories automatically get more attention based on their temperature.

### Temperature-boosted recall

```bash
wheeler-recall --temperature-boost 0.1 "python bug"
```

When `--temperature-boost` is nonzero, ranking uses `similarity + boost × temperature` — hotter memories get a slight ranking bonus. Default boost is 0.0 (pure similarity ranking, identical to previous behavior).

## Scrub a brick timeline

```bash
wheeler-scrub --text "fix the python debug error"           # find by text
wheeler-scrub --text "solder header" --chunk hardware       # in specific chunk
wheeler-scrub path/to/brick.npz                              # direct path
```

Opens an interactive matplotlib viewer with a tick slider.

## GPU benchmark

```bash
wheeler-bench-gpu                                # CPU vs GPU comparison
wheeler-bench-gpu --verify-only                  # correctness check only
wheeler-bench-gpu --batch-sizes 100,500,2000     # custom sizes
```

## Inspect temperatures

```bash
wheeler-temps                     # all memories
wheeler-temps --chunk code        # specific chunk
wheeler-temps --tier hot          # filter by tier
wheeler-temps --sort hits         # sort by hit count
```

## Forget / evict memories

```bash
wheeler-forget                      # full sweep (fade + evict + capacity)
wheeler-forget --dry-run            # show what would happen
wheeler-forget --text "some memory" # forget a specific memory
wheeler-forget --hex abc123...      # forget by hex key
wheeler-forget --coldest 10         # diagnostic: show 10 coldest memories
```

The sweep runs three phases:

1. **Fade** — memories below temperature 0.05 have their brick (`.npz` evolution history) deleted. The attractor and index entry remain, so the memory is still recallable but its formation history is lost.
2. **Evict** — memories below temperature 0.01 are fully removed (attractor, index entry, association edges, warmth).
3. **Capacity** — if the total memory count exceeds 10,000, the bottom 10% by temperature are evicted (never warm or hot memories).

Memories younger than 1 day are never evicted regardless of temperature.

## Sleep consolidation

```bash
wheeler-sleep                      # consolidate all eligible memories
wheeler-sleep --dry-run            # show what would be consolidated
wheeler-sleep --chunk code         # consolidate specific chunk
wheeler-sleep --stats              # show per-memory frame counts + potential savings
```

Consolidation prunes redundant intermediate frames *within* each brick, keeping only salient keyframes where the pattern changed significantly. Frame 0 (seed) and the final frame (attractor) are always preserved.

Thresholds are temperature-tiered:

- **Hot (>=0.6)** — skipped entirely (actively used memories)
- **Warm (>=0.3)** — light pruning (~40-60% frames retained)
- **Cold (<0.3)** — aggressive pruning (~15-25% frames retained)

Already-consolidated bricks are skipped (idempotent). This is distinct from eviction — consolidation reduces storage while preserving the formation story at key transition points.

## Corpus crystallization

```bash
wheeler-crystallize corpus.jsonl                              # crystallize with embeddings
wheeler-crystallize corpus.csv --chunk science                # force chunk
wheeler-crystallize corpus.txt --max-items 10000 --verbose    # validation run
wheeler-crystallize corpus.jsonl --batch-size 64              # larger batches
wheeler-crystallize corpus.jsonl --no-resume                  # re-process everything
wheeler-crystallize corpus.jsonl --no-embed                   # SHA-256 instead of embeddings
wheeler-crystallize corpus.jsonl --fmt jsonl                  # explicit format
```

Pre-trains Wheeler by feeding a text corpus through the full encode-evolve-store pipeline at scale. Supported formats: JSONL, CSV, TXT, Parquet.

Resume is on by default — re-running skips already-stored entries.

## LLM agent

```bash
wheeler-agent                      # start the Ollama/qwen3 agent loop
wheeler-agent --interactive        # explicit REPL mode
wheeler-agent --model qwen3:8b     # override model
wheeler-agent --ollama http://host:11434  # custom Ollama URL
```

The agent recalls relevant memories before each response, stores the exchange afterward, and uses temperature to calibrate epistemic confidence language ("I remember..." vs. "I vaguely recall..."). Requires Ollama running locally.

## Wheeler-primary agent

```bash
wheeler-primary "What is quantum entanglement?"     # single query
wheeler-primary --interactive                        # REPL mode
wheeler-primary --show-state "Tell me about Python"  # show Wheeler state before response
wheeler-primary --model qwen2.5:1.5b                 # override decoder model
wheeler-primary --confidence-floor 0.4               # stricter confidence threshold
wheeler-primary --recall-k 10                        # recall more memories per query
wheeler-primary --verbose                            # pipeline diagnostics
```

Wheeler-primary mode: Wheeler Memory is the cognitive system, the small model is a pure language renderer. The model reads Wheeler's attractor state and renders it as natural language — it does not reason or add its own knowledge.

## Encoder types

The `--encoder` flag (available on `wheeler-store`, `wheeler-recall`, and benchmark commands) selects the text-to-frame encoding strategy:

| Encoder | Description |
|---------|-------------|
| `hash` | SHA-256 deterministic encoding (exact match only) |
| `hippocampus` | Native character n-gram random indexing (no pretrained models) |
| `embedding` | MiniLM sentence-transformer (requires `.[embed]`) |
| `language` | Language Wheeler encoding |
| `blended` | Hippocampus (0.7) + Language Wheeler (0.3) — **default** |
| `word` | Word-level random indexing (SVD on PMI matrix) |
| `word-blended` | Hippocampus + word encoder hybrid |
| `context` | Context-window random indexing (distributional semantics, trained on WikiText-103) |

Default encoder is `blended` (configurable via `DEFAULT_ENCODER` in `constants.py`).

## Quality benchmark

```bash
wheeler-bench                                              # run CA quality score
wheeler-bench --commit abc1234 --changed "MAX_PUSH_STRENGTH"  # tag with commit + param
wheeler-bench --notes "testing sharper attractors"          # add notes
```

Outputs a composite quality score: `0.6*avg_corr + 0.2*(1-conv_ratio) + 0.1*(ticks/1000) + 0.1*(1-alive)`. Lower is better. Results appended to `results.tsv`.

## MMLU benchmark

```bash
wheeler-mmlu --subjects high_school_physics --mode cortex  # single subject
wheeler-mmlu --all --mode cortex                           # all 57 subjects
wheeler-mmlu --all --mode learn                            # learn → consolidate → test
wheeler-mmlu --all --mode learn-interference               # learn + experiential + SCM
wheeler-mmlu --mode cortex --classifier-weights cortex_classifier.npz  # L3 classifier
wheeler-mmlu --list-subjects                               # show available subjects
wheeler-mmlu --output results.tsv                          # save results
```

Modes: `cortex` (L3 classifier scoring), `semantic` (Pearson correlation), `recall-text` (reconstruction + text decode), `decode` (small model decoder), `learn` (full cycle), `learn-interference` (learn + experiential).

## SimLex-999 benchmark

```bash
wheeler-simlex --encoder context --mode pearson            # context-RI semantic similarity
wheeler-simlex --encoder hippocampus --mode pearson        # hippocampus baseline
wheeler-simlex --encoder embedding --mode pearson          # MiniLM ceiling
```

Evaluates semantic similarity against the SimLex-999 gold standard. Reports Spearman rho (higher is better).

## SCM inspector

```bash
wheeler-scm                             # show SCM trust topology summary
wheeler-scm --heatmap                   # visualise SCM as heatmap
wheeler-scm --reset                     # reset SCM to zeros (fully permissive)
```

Inspect and manage the Structural Coherence Map (SCM) — the persistent 64×64 trust topology that gates three-grid interference.

## Generative engine

```bash
wheeler-generate                        # IT-from-BIT generative text
wheeler-generate --verbose              # show attractor state
```

Generates text from attractor dynamics without any language model.
