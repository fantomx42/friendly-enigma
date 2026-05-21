# Contributing to Wheeler Memory

## Development Setup

```bash
git clone https://github.com/fantomx42/wheeler-memory.git
cd wheeler-memory
python -m venv .venv
source .venv/bin/activate
pip install -e ".[embed]"
```

Python 3.11+ required. The system Python on Arch/CachyOS is externally-managed — always use the venv.

## Running Tests

```bash
pytest                           # run all tests (757 tests across 42 modules)
pytest -m "not slow"             # skip slow tests
pytest -m "not embed"            # skip tests requiring sentence-transformers
pytest tests/test_dynamics.py    # run a specific test file
```

Test markers:
- `slow` — tests that take significant time
- `embed` — tests requiring `sentence-transformers` (install with `.[embed]`)

## Code Style

- Python 3.11+ features are fine (type unions `X | Y`, `match`, etc.)
- Use type hints for public function signatures
- Docstrings on public functions (NumPy style with Parameters/Returns sections)
- Formatting: `ruff format` (auto-runs via PostToolUse hook on .py edits)

## Project Structure

- `wheeler_memory/` — core library (36 modules, import as `from wheeler_memory import ...`)
  - `theories/` — theory experiments (basin, resonance, synthesis)
  - `gpu/` — HIP/CUDA kernel sources and compiled `.so`
- `scripts/` — CLI entry points (16 commands registered in `pyproject.toml`)
  - `bench/` — benchmarks & evaluation scripts
  - `exploration/` — standalone exploration scripts
  - `tools/` — data prep, corpus cleanup, HIP build utilities
  - `experiments/` — theory test harnesses
- `tests/` — pytest suite (42 modules, 757 tests)
- `docs/` — documentation (Markdown)
  - `reports/` — generated assessment reports
  - `demos/` — archived HTML demos
- `results/` — benchmark logs and MMLU baselines
- `plans/` — research & implementation plans
- `artifacts/` — experiment artifacts
- `docs/program.md` — autoresearch tuning program

## Making Changes

1. Create a branch from `main`
2. Make your changes
3. Run `pytest` to verify nothing breaks
4. Commit with a clear message (e.g., `feat: add X`, `fix: Y`, `docs: Z`)
5. Open a PR against `main`

## Adding CLI Commands

1. Create the script in `scripts/`
2. Add the entry point in `pyproject.toml` under `[project.scripts]`
3. Run `pip install -e .` to register the new command
4. Document in `docs/cli.md`
