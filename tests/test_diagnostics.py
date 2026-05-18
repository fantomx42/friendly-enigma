"""Parity tests for the read-only CA tick decomposer.

The diagnostic re-implements the same math as ``apply_ca_dynamics`` so it can
report intermediates. If the two ever drift, the diagnostic stops being a
faithful observation of the substrate. These tests catch that drift.
"""

import numpy as np
import pytest

from wheeler_memory.diagnostics import NEIGHBOR_NAMES, decompose_tick
from wheeler_memory.dynamics import apply_ca_dynamics


@pytest.mark.parametrize(
    "regime",
    ["smooth", "ternary", "flat", "spike", "edge_wrap"],
)
def test_decompose_tick_matches_apply_ca_dynamics(regime):
    """Diagnostic ``next_frame`` must equal ``apply_ca_dynamics`` byte-for-byte."""
    rng = np.random.default_rng(0)

    if regime == "smooth":
        frame = rng.uniform(-1.0, 1.0, size=(64, 64)).astype(np.float32)
    elif regime == "ternary":
        frame = rng.choice([-1.0, 0.0, 1.0], size=(64, 64)).astype(np.float32)
    elif regime == "flat":
        frame = np.full((64, 64), 0.4, dtype=np.float32)
    elif regime == "spike":
        frame = np.zeros((64, 64), dtype=np.float32)
        frame[32, 32] = 1.0
    elif regime == "edge_wrap":
        # Ensures the wrapping boundary is exercised — spike at (0, 0)
        frame = np.zeros((64, 64), dtype=np.float32)
        frame[0, 0] = 1.0
        frame[63, 63] = -1.0
    else:
        raise AssertionError("unreachable")

    diag = decompose_tick(frame)
    substrate = apply_ca_dynamics(frame)
    assert np.array_equal(diag["next_frame"], substrate), (
        f"diagnostic drift on regime '{regime}'"
    )


def test_decompose_tick_when_and_why_are_canon_gaps():
    """``when`` and ``why`` are ``None`` because the per-tick rule does not
    consult T or SCM. This is canon-vs-code documentation, not a stub."""
    frame = np.zeros((64, 64), dtype=np.float32)
    diag = decompose_tick(frame)
    assert diag["when"] is None
    assert diag["why"] is None


def test_decompose_tick_who_axis_recovers_neighbor_identity():
    """``who`` is ``argmax`` over the neighbor stack — index into
    ``where_names``. On a single-spike frame, each of the four cells
    immediately neighboring the spike should identify the direction of
    the spike as its max neighbor.

    The substrate computes ``max_neighbor = neighbors.max(axis=0)`` and
    discards which index produced the max. The diagnostic recovers that
    identity without changing the rule.
    """
    frame = np.zeros((64, 64), dtype=np.float32)
    frame[32, 32] = 1.0
    diag = decompose_tick(frame)
    who = diag["who"]
    # Cell above the spike — its 'down' neighbor is the spike
    assert NEIGHBOR_NAMES[int(who[31, 32])] == "down"
    # Cell below the spike — its 'up' neighbor is the spike
    assert NEIGHBOR_NAMES[int(who[33, 32])] == "up"
    # Cell to the left — its 'right' neighbor is the spike
    assert NEIGHBOR_NAMES[int(who[32, 31])] == "right"
    # Cell to the right — its 'left' neighbor is the spike
    assert NEIGHBOR_NAMES[int(who[32, 33])] == "left"


def test_decompose_tick_what_is_input_frame_value():
    """``what`` is the cell's pre-tick value, copy-safe (mutation does not
    leak back to caller)."""
    frame = np.full((4, 4), 0.5, dtype=np.float32)
    diag = decompose_tick(frame)
    assert np.array_equal(diag["what"], frame)
    diag["what"][0, 0] = 999.0
    assert frame[0, 0] == pytest.approx(0.5)


def test_decompose_tick_how_delta_matches_substrate_change():
    """``how_delta`` plus the input should equal the substrate output before
    the ``clip`` — the diagnostic exposes the combination-operator output."""
    rng = np.random.default_rng(7)
    frame = rng.uniform(-0.6, 0.6, size=(32, 32)).astype(np.float32)  # avoid clip
    diag = decompose_tick(frame)
    assert np.allclose(frame + diag["how_delta"], diag["next_frame"], atol=1e-7)
