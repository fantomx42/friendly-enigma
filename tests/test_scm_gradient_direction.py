"""Gradient-direction sanity check for SCMGrid.update_from_recall.

If this test fails, the gradient rule is miscalibrated and the closed-loop
A/B (Pearson vs. interference-frozen-SCM vs. interference-learning-SCM) is
invalid. Must pass before that A/B runs.

Construction notes
------------------
The rule pulls cells toward 0 on positive advantage and clamps |M| at
SCM_HARDENING_FLOOR (sign-preservation), so the rule's actual fixed point
under positive advantage is S_good = ε_floor · sign(S_perturbed).

S_perturbed has |M| ≈ 0.5 so openness < SCM_OPEN_FRACTION_CEIL — without
this, the homeostasis guard at scm_grid.py:168 short-circuits the update.
"""

import numpy as np

from wheeler_memory.constants import (
    SCM_HARDENING_FLOOR,
    SCM_OPEN_FRACTION_CEIL,
)
from wheeler_memory.scm_grid import SCMGrid


def test_gradient_step_moves_toward_attractor(tmp_path):
    rng = np.random.default_rng(seed=42)

    base = np.sign(rng.standard_normal((64, 64))).astype(np.float32)
    noise = (0.1 * rng.standard_normal((64, 64))).astype(np.float32)
    s_perturbed = (0.5 * base + noise).astype(np.float32)
    s_good = (SCM_HARDENING_FLOOR * np.sign(s_perturbed)).astype(np.float32)

    scm = SCMGrid.load_or_create(tmp_path)
    scm.grid[:] = s_perturbed
    scm.kappa_base = 0.0

    corpus_att = np.full((64, 64), 0.8, dtype=np.float32)
    experiential_att = np.full((64, 64), 0.8, dtype=np.float32)

    assert scm.openness() < SCM_OPEN_FRACTION_CEIL, (
        "test setup invalid: homeostasis ceiling would skip the gradient"
    )

    n_updated = scm.update_from_recall(corpus_att, experiential_att, kappa=0.9)
    assert n_updated > 0, "gradient must touch at least one cell"

    err_before = float(np.linalg.norm(s_perturbed - s_good))
    err_after = float(np.linalg.norm(scm.grid - s_good))

    assert err_after < err_before, (
        f"Gradient should reduce ||S - S_good||. "
        f"before={err_before:.6f}, after={err_after:.6f}"
    )
