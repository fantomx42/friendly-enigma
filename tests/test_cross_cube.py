"""Tests for cross-cube interference semantics (CANON §6.4)."""

from __future__ import annotations

import numpy as np

from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.fcas import NUM_FACES, cross_cube_interference
from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.interference import InterferenceResult

_VALID_STATES = set(InterferenceResult.STATE_CODES)


def _parent(text: str) -> np.ndarray:
    return evolve_and_interpret(hash_to_frame(text))["attractor"]


def test_cross_cube_is_deterministic():
    a = cross_cube_interference(_parent("a parent attractor"))
    b = cross_cube_interference(_parent("a parent attractor"))
    assert a["depth_coherence"] == b["depth_coherence"]
    assert a["per_face"][0]["correlation"] == b["per_face"][0]["correlation"]


def test_six_faces_with_valid_states():
    r = cross_cube_interference(_parent("six faces"))
    assert r["n_faces"] == NUM_FACES
    assert set(r["per_face"]) == set(range(NUM_FACES))
    for face in r["per_face"].values():
        assert face["dominant_state"] in _VALID_STATES
        for key in (
            "grounded_fraction",
            "absorbed_fraction",
            "unconsolidated_fraction",
            "contested_fraction",
        ):
            assert 0.0 <= face[key] <= 1.0


def test_contested_is_zero_on_open_depth_axis():
    """Zero SCM is fully open, so the closed-SCM CONTESTED state never fires."""
    r = cross_cube_interference(_parent("no trust gate on depth"))
    assert r["contested_fraction"] == 0.0
    for face in r["per_face"].values():
        assert face["contested_fraction"] == 0.0


def test_children_are_orthogonal_to_parent():
    """Canonical semantics: nested cubes are an orthogonal decomposition.

    The portal hash decorrelates children from the parent, so mean
    |Pearson(parent, child)| must sit near the noise floor for independent
    length-4096 vectors (1/sqrt(4096) ≈ 0.0156), not climb toward 1.
    """
    noise_floor = 1.0 / np.sqrt(4096)
    coherences = [
        cross_cube_interference(_parent(t))["depth_coherence"]
        for t in ("orthogonality one", "orthogonality two", "orthogonality three")
    ]
    assert max(coherences) < 5 * noise_floor, (
        f"parent↔child coherence {coherences} exceeded the orthogonality bound; "
        "children are not independent of the parent"
    )
