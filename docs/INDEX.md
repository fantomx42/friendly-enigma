# Wheeler Memory Documentation

## Canon

| Guide | Description |
|-------|-------------|
| [Canon](../CANON.md) | Architectural source-of-truth — read this first |

## Getting Started

| Guide | Description |
|-------|-------------|
| [Installation](install.md) | Python venv setup, platform-specific GPU notes, Ollama |
| [Interactive Demo](demos/demo.html) | See the CA engine in your browser (no server needed) |

## Core Guides

| Guide | Description |
|-------|-------------|
| [Architecture](architecture.md) | CA dynamics, encoders, three-grid interference, cortex pipeline, chunked storage |
| [Concepts](concepts.md) | Theoretical foundation, reconstructive recall, semantic vs exact search |
| [Design Principles](design.md) | The Darman philosophy — why recall is reconstruction, not retrieval |

## Reference

| Guide | Description |
|-------|-------------|
| [CLI Reference](cli.md) | Every command and flag documented with examples |
| [API Reference](api.md) | Python library usage — store, recall, reconstruct, crystallize, decode |
| [GPU Acceleration](gpu.md) | HIP/ROCm and CUDA setup, benchmarks |

## Project

| Guide | Description |
|-------|-------------|
| [Future / Roadmap](future.md) | Active research and planned features |
| [Autoresearch Protocol](program.md) | Parameter sweep workflow for constants.py |
| [Contributing](../CONTRIBUTING.md) | Development setup, testing, code style |
| [Changelog](../CHANGELOG.md) | Release history |

## Advisories

Recommendation / decision papers (distinct from empirical [reports](reports/)).

| Advisory | Description |
|----------|-------------|
| [Advisories index](advisory/README.md) | All recommendation papers + naming convention |
| [SQLite storage evaluation](advisory/sqlite-storage-evaluation-2026-05-30.md) | Would SQLite (or Turso/D1/LiteFS/Litestream/DuckDB) help the storage layer? Verdict + recommended stock-SQLite + `sqlite-vec` path |

## Suggested Reading Order

1. **Installation** — get running
2. **Concepts** — understand the theory
3. **Architecture** — understand the implementation
4. **CLI Reference** — start using it
5. **API Reference** — integrate into your code
6. **Design Principles** — understand the philosophy
