# Notes — Research Archive

This directory holds research artifacts kept for reference but **not** part of the
active Wheeler Memory project. Nothing in here runs under pytest, nothing is
imported by `wheeler_memory/`, nothing has a registered CLI entry point.

The archive exists because the ideas are good — but tracking them as live code
invited feature creep. Moving them here is "yes, and later" rather than "no."

## Layout

```
notes/
├── exploration/   Research notebook scripts (CA dynamics surveys, paraphrase
│                  studies, eviction trials). Some still produce diagrams in
│                  docs/assets/reports/ — those are run manually:
│                      python notes/exploration/test_diversity.py --output ...
│
├── experiments/   Per-theory exploratory scripts (test_basin.py,
│                  test_lichtenberg.py, ...). Each script is the hands-on
│                  exercise of the matching theory module.
│
└── theories/      Theoretical modules archived because no production code
│                  depends on them:
│                      lichtenberg.py — topology visualization (PCA, plots)
│                      resonance.py   — CA-dynamics corpus search
│                      structured.py  — Theory-builder using recognize_top_k
│   tests/         The pytest suites that exercised the archived theories.
```

## What stayed live

`wheeler_memory/theories/` retains three modules because they have production
callers:

| Module | Used by |
|---|---|
| `basin.py` | imported by `metrics.py` and `synthesis.py` (transitive) |
| `metrics.py` | `wheeler_memory/agent.py`, `wheeler_memory/decoder.py`, `scripts/wheeler_mmlu.py` |
| `synthesis.py` | `scripts/bench/apple_test_semantic.py` |

Their tests stayed under `tests/` (`test_theories_basin.py`, `test_theories_metrics.py`,
`test_theories_synthesis.py`).

## Reviving anything

If a notes/ module is wanted back as live code:

1. Move it back under `wheeler_memory/theories/` (or a new home).
2. Re-add its export line to `wheeler_memory/theories/__init__.py`.
3. Move its tests back under `tests/`.
4. Verify by running `pytest -m "not slow"`.

Imports inside the archived modules use the same `from ..dynamics import ...`
relative-import shape they used when live — they just need to be one folder up
from `notes/` to resolve, which is true if you move them back.
