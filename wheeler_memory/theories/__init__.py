"""Wheeler Theories — production-supporting CA topology helpers.

Three modules remain in the package because production code depends on them:
    basin     — measure_basin_width / find_basin_gaps (used by metrics + synthesis)
    metrics   — classify_output / energy / hallucination_score (used by agent, decoder, wheeler_mmlu)
    synthesis — apple_test (used by scripts/bench/apple_test_semantic.py)

The exploratory siblings — lichtenberg, resonance, structured — were archived to
`notes/theories/` to fight feature creep. See `notes/README.md`.
"""

from .basin import find_basin_gaps, map_all_basins, measure_basin_width
from .metrics import (
    basin_width,
    context_weight,
    energy,
    hallucination_score,
    topology_consistency,
)
from .synthesis import apple_test, run_apple_battery, synthesize_from_gap
