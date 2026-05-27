"""Tests for the FCAS core address layer (wheeler_memory/fcas.py)."""

from __future__ import annotations

import numpy as np
import pytest

from wheeler_memory import fcas
from wheeler_memory.fcas import (
    NUM_FACES,
    Address,
    address_of,
    evolve_cube,
    expand_cube,
    portal_hash,
    recognize_address,
    resolve,
    traverse,
)
from wheeler_memory.hashing import hash_to_frame

from conftest import store_test_memory


# --- Portal primitives: determinism ---------------------------------------


def test_portal_hash_is_deterministic_and_layout_invariant():
    att = hash_to_frame("an attractor seed")
    h1 = portal_hash(att)
    h2 = portal_hash(att)
    assert h1 == h2, "portal_hash must be stable across calls"
    assert len(h1) == 64, "SHA-256 hex digest is 64 chars"
    # A non-contiguous view of the same values must hash identically.
    view = np.asfortranarray(att)
    assert portal_hash(view) == h1


def test_portal_hash_differs_for_different_attractors():
    a = hash_to_frame("alpha")
    b = hash_to_frame("beta")
    assert portal_hash(a) != portal_hash(b)


def test_expand_cube_six_distinct_deterministic_seeds():
    portal = portal_hash(hash_to_frame("root"))
    seeds = expand_cube(portal)
    assert set(seeds) == set(range(NUM_FACES))
    assert len(set(seeds.values())) == NUM_FACES, "all 6 face seeds must be distinct"
    assert expand_cube(portal) == seeds, "expand_cube must be deterministic"


# --- Branching ------------------------------------------------------------


def test_evolve_cube_returns_six_valid_results():
    portal = portal_hash(hash_to_frame("root"))
    results = evolve_cube(portal)
    assert set(results) == set(range(NUM_FACES))
    for face, res in results.items():
        assert res["attractor"].shape == (64, 64)
        assert res["state"] in {"CONVERGED", "OSCILLATING", "CHAOTIC", "DEGENERATE"}


# --- Address type ---------------------------------------------------------


def test_address_of_is_depth_zero_coordinate():
    att = hash_to_frame("memory")
    addr = address_of(att)
    assert addr == Address(hex_key=portal_hash(att), depth=0, face_path=())
    assert addr.depth == 0
    assert addr.face_path == ()


# --- Traversal ------------------------------------------------------------


def test_traverse_is_reproducible():
    a = traverse("seed text", [0, 3, 1])
    b = traverse("seed text", [0, 3, 1])
    assert a == b, "same inputs must yield the same chain"


def test_traverse_depth_equals_face_path_length():
    chain = traverse("seed text", [0, 3, 1])
    assert len(chain) == 4  # root + 3 steps
    for i, addr in enumerate(chain):
        assert addr.depth == i
        assert addr.depth == len(addr.face_path)
    assert chain[-1].face_path == (0, 3, 1)


def test_traverse_extends_prefix():
    short = traverse("seed text", [2])
    long = traverse("seed text", [2, 5])
    # A deeper walk shares its prefix with the shorter one.
    assert long[:2] == short
    assert long[-1].depth == 2


def test_traverse_rejects_out_of_range_face():
    with pytest.raises(ValueError):
        traverse("seed text", [NUM_FACES])


# --- Resolution -----------------------------------------------------------


def test_resolve_empty_path_returns_depth_zero_attractor():
    res = resolve("seed text", [])
    root = traverse("seed text", [])[0]
    assert res["address"] == root
    assert res["address"].depth == 0
    assert res["attractor"].shape == (64, 64)


def test_resolve_matches_terminal_of_traverse():
    seq = [1, 4, 2]
    res = resolve("seed text", seq)
    chain = traverse("seed text", seq)
    assert res["address"] == chain[-1]
    assert res["address"].face_path == tuple(seq)


# --- Recall-path bridge ---------------------------------------------------


def test_recognize_address_returns_coordinate_for_known_memory(tmp_data_dir):
    text = "python list comprehensions"
    key = store_test_memory(text, tmp_data_dir)

    seed, addr = recognize_address(text, data_dir=tmp_data_dir)
    assert seed is not None, "expected recognition hit on exact stored text"
    assert addr is not None

    # The address coordinate is the portal hash of the stored attractor.
    stored = np.load(tmp_data_dir / "chunks" / seed.chunk / "attractors" / f"{key}.npy")
    assert addr.hex_key == portal_hash(stored)
    assert addr.depth == 0


def test_recognize_address_miss_returns_none_pair(tmp_data_dir):
    store_test_memory("python list comprehensions", tmp_data_dir)
    seed, addr = recognize_address(
        "totally unrelated quantum chromodynamics gibberish",
        data_dir=tmp_data_dir,
        threshold=0.99,
    )
    assert seed is None
    assert addr is None
