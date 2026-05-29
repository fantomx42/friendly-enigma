---
name: run-wheeler-memory
description: Run, smoke-test, or benchmark wheeler-memory — the pure-Python cellular-automaton associative memory. Use when the user asks to "run wheeler", "start wheeler-memory", "smoke wheeler", "verify wheeler-memory", "run the CA benchmark", "store and recall a memory", "run wheeler-baseline" or "run wheeler-rerank".
---

Wheeler-memory is a pure-Python associative memory system whose runtime surface is a fleet of 18 CLI commands (`wheeler-store`, `wheeler-recall`, `wheeler-bench`, `wheeler-baseline`, `wheeler-rerank`, ...) registered as `[project.scripts]` in `pyproject.toml`. There is no GUI, no long-running server — every "run" is one of those CLIs. This skill's driver is `.claude/skills/run-wheeler-memory/smoke.sh`, a bash script that exercises the four CLIs that prove the install is healthy (`wheeler-info`, `wheeler-store`, `wheeler-recall`, `wheeler-bench`) against an isolated tempdir.

All commands below are run from the repo root: `/home/tristan/projects/wheeler-memory` (a symlink to `/run/media/tristan/projects/wheeler-memory`). Paths in this file are relative to that root.

## Prerequisites

System Python on this host is externally-managed (Arch/CachyOS) — **you cannot `pip install` system-wide**. Always use the project venv. If the venv is missing, bootstrap it:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[embed]"
```

For the re-ranking bench (`wheeler-rerank`), add the `rerank` extras (pulls `ir-datasets` + `ir-measures` and ~3GB MS MARCO corpus on first use):

```bash
.venv/bin/pip install -e ".[embed,rerank]"
```

No `apt-get` packages were needed in this container (numpy/scipy/matplotlib/psutil are wheels; sentence-transformers downloads its weights at first use).

## Run (agent path)

**Primary: the smoke driver.** Verifies install + all four core CLIs + bench target in ~5s warm (~25s cold).

```bash
./.claude/skills/run-wheeler-memory/smoke.sh
```

Expected output (verified this session):

```
==> wheeler-info (system + GPU detection)
  cores=20 mem=30.8GB device=cpu
==> wheeler-store + wheeler-recall (hash encoder, tempdir=/tmp/wheeler-smoke-XXXXXX)
  store/recall round-trip ok
==> wheeler-bench --json --no-save (quality benchmark, target score < 0.25)
  score=0.0107  avg|r|=0.0143  converged=20/20  elapsed=0.20s
==> OK: wheeler-memory smoke passed
```

The smoke uses the `hash` encoder (no model load, ~5s). For semantic recall, add `--encoder embedding` to `wheeler-store`/`wheeler-recall` — first call loads MiniLM (~10s).

### Run individual CLIs

```bash
# Quality benchmark — the canonical health check (composite score, lower better)
.venv/bin/wheeler-bench                          # appends to results.tsv
.venv/bin/wheeler-bench --json --no-save         # one-shot, no persistence

# Store + recall against a tempdir (does NOT touch ~/.wheeler_memory)
D=$(mktemp -d); .venv/bin/wheeler-store --data-dir "$D" "your text here"
.venv/bin/wheeler-recall --data-dir "$D" "your query"

# Semantic store/recall (requires .[embed])
.venv/bin/wheeler-store --data-dir "$D" --encoder embedding "your text here"
.venv/bin/wheeler-recall --data-dir "$D" --encoder embedding "your query"
```

### Run the bench harnesses (CA-vs-encoder delta benchmarks)

These were added in commit `abf2e985` and write to their own TSV at repo root:

```bash
# sciq question -> support retrieval, three arms (raw MiniLM / CA / CA+reconstruction)
.venv/bin/wheeler-baseline --n 50 --seed 42                    # canonical, ~5 min
.venv/bin/wheeler-baseline --n 5  --seed 42 --no-save          # smoke, ~30s

# TREC DL 2019 passage re-ranking via ir_datasets (needs .[rerank])
.venv/bin/wheeler-rerank --queries 3  --top-k 100 --no-save    # smoke
.venv/bin/wheeler-rerank --queries 43 --top-k 100              # canonical, ~15 min
```

First `wheeler-rerank` invocation downloads ~3GB MS MARCO corpus to `~/.ir_datasets/`. Cached thereafter.

### Run the tests

```bash
.venv/bin/pytest                          # full suite
.venv/bin/pytest -m "not slow"            # skip slow
.venv/bin/pytest -m "not embed"           # skip sentence-transformers tests
```

A `PostToolUse` hook auto-runs `pytest` when `wheeler_memory/` or `tests/` files change (60s timeout). Another auto-runs `wheeler-bench` when `constants.py` changes. See `/hooks`.

## Run (human path)

There isn't one. Every entrypoint is a CLI — none of them spawn a GUI or stay running. If a user "runs wheeler-memory," they mean one of the commands above.

## Gotchas

- **System Python rejects `pip install`** on Arch/CachyOS (PEP 668). The first error you'll see is `externally-managed-environment`. Always invoke `.venv/bin/pip`, never bare `pip`.
- **`wheeler-info` emits JSON followed by a colorized ANSI footer.** `json.load(stdin)` chokes on the trailing text — use `json.JSONDecoder().raw_decode(text)` to parse only the leading object. (The smoke does this.)
- **`wheeler-recall` returns ranked results, not exact-match.** With the default `hash` encoder, similarity is essentially random for semantically-related queries (SHA-256 by design destroys semantic similarity — that's what `--encoder embedding` is for). The smoke uses hash to keep startup fast; for real semantic use, pass `--encoder embedding`.
- **Sacred files — do NOT modify** even if a test failure or refactor looks tempting:
  - `wheeler_memory/hashing.py` — deterministic SHA-256 encoding
  - `wheeler_memory/storage.py` — locked storage contract
  - `wheeler_memory/chunking.py` — locked domain routing
  - `wheeler_memory/rotation.py` — locked rotation logic
  - `scripts/bench_quality.py` `TEST_INPUTS` list — bench results must be comparable across experiments
- **`~/.wheeler_memory/`** is the default data dir; running `wheeler-store` without `--data-dir` writes to it (and `auto_evict=True` by default may purge old memories). Always `--data-dir $TMPDIR` for ephemeral testing.
- **GPU dispatch is opaque from Python.** `wheeler-info` reported `optimal_device=cpu` even though a Radeon RX 9070 XT is present and the HIP kernel works for `evolve_batch` (`wheeler-bench` elapsed 0.2s for 20 inputs — that's GPU). The `optimal_device` field reflects PyTorch/sentence-transformers' choice, NOT whether the HIP kernel is dispatching.
- **`results.tsv`, `baseline.tsv`, `rerank.tsv` at repo root are append-only artifacts** — each row is a measurement. Don't truncate; don't sort; new rows are written by `wheeler-bench` / `wheeler-baseline` / `wheeler-rerank`.
- **Autoresearch protocol**: when tuning parameters, edit ONLY `wheeler_memory/constants.py`, then re-run `wheeler-bench --commit <hash7> --changed "<param>"`. See `docs/program.md`.

## Troubleshooting

- **`FAIL: .venv missing`** from smoke.sh → run `python -m venv .venv && .venv/bin/pip install -e '.[embed]'`.
- **`externally-managed-environment`** from any pip command → you're using system Python. Use `.venv/bin/pip`.
- **`json.decoder.JSONDecodeError: Extra data`** when piping `wheeler-info` to a JSON parser → use `raw_decode`, not `load`. The CLI emits trailing human-readable text after the JSON.
- **`Error: ... requires sentence-transformers`** from any encoder-using script → run `.venv/bin/pip install -e ".[embed]"`.
- **`Error: ... requires ir-datasets`** from `wheeler-rerank` → run `.venv/bin/pip install -e ".[embed,rerank]"`.
- **`wheeler-bench` score > 0.25** → indicates a regression in CA dynamics quality. Compare against the last row of `results.tsv` — if `improved=0` from the script, revert the most recent `constants.py` change.
- **`wheeler-rerank` fails with "No candidate source available"** → `ir_datasets` version regression dropped `scoreddocs` for `msmarco-passage/trec-dl-2019/judged`. Fallback: generate `runs/dl2019.bm25.txt` via `pyserini` (the script reads TREC run format).
