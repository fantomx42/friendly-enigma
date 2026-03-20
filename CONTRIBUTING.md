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
pytest                           # run all tests (~233 tests)
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
- No linter is enforced, but keep code clean and consistent with existing patterns

## Project Structure

- `wheeler_memory/` — core library (import as `from wheeler_memory import ...`)
- `scripts/` — CLI entry points (registered in `pyproject.toml`)
  - `bench/` — benchmarks & evaluation scripts
  - `exploration/` — standalone exploration scripts
  - `tools/` — data prep, corpus cleanup, HIP build utilities
  - `experiments/` — theory test harnesses
- `tests/` — pytest test suite
- `docs/` — documentation (Markdown)
  - `reports/` — generated assessment reports
  - `demos/` — archived HTML demos
- `plans/` — research & implementation plans

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
