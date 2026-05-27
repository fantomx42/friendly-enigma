"""FCAS — Fractal Cube Address Space (core address layer).

Addresses are ``(hash, depth)`` tuples (CANON.md §6). The SHA-256 of a terminal
attractor plays a triple role: it is the *coordinate* of that attractor, the
*reconstruction seed* for re-instantiating it, and the *origin* of a fresh
nested sub-grid. That collapse of roles is what makes the space fractal — every
attractor is also a coordinate which is also a new origin.

This module composes existing primitives without modifying any sacred/locked
file. It only *reads* ``hash_to_frame`` (hashing.py) and ``evolve_and_interpret``
/ ``evolve_batch`` (dynamics.py), and wraps ``recall_api.recognize`` the same way
recall_api.py wraps storage.py. Nothing here is persisted: addresses are derived
deterministically on demand.

The portal chain (archived spec, plans/archive/fractal_cube_address_space.md):

    text → hash_to_frame → evolve_and_interpret → portal_hash
         → expand_cube[face] → hash_to_frame → evolve_and_interpret → ...

The branching factor of 6 is not a design choice: the CA's ternary state space
(-1, 0, +1) maps onto cube geometry as 3 axes × 2 directions = 6 faces.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dynamics import evolve_and_interpret, evolve_batch
from .hashing import hash_to_frame

NUM_FACES = 6  # 3 axes × 2 directions — falls out of the ternary geometry.


# --- Portal primitives ----------------------------------------------------


def portal_hash(attractor: np.ndarray) -> str:
    """SHA-256 of attractor bytes — the portal to the next depth level.

    ``ascontiguousarray`` guarantees a canonical byte layout, so a memory-mapped
    or transposed view of the same values yields the same hash (determinism).
    """
    canonical = np.ascontiguousarray(attractor, dtype=np.float32)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def expand_cube(portal: str) -> dict[int, str]:
    """Deterministic per-face child seeds from a portal hash.

    Returns ``{face_index: child_seed_hex}`` for the 6 faces. The children are
    independent by construction — they share no state and can be evolved in
    parallel.
    """
    return {
        face: hashlib.sha256(portal.encode("utf-8") + bytes([face])).hexdigest()
        for face in range(NUM_FACES)
    }


def evolve_cube(portal: str) -> dict[int, dict]:
    """Evolve all 6 face-children of a node.

    Returns ``{face_index: evolve_and_interpret_result}``. Uses ``evolve_batch``
    so the 6 frames dispatch together (GPU batch when available, serial CPU
    otherwise).
    """
    face_seeds = expand_cube(portal)
    frames = [hash_to_frame(seed) for seed in face_seeds.values()]
    results = evolve_batch(frames)
    return dict(zip(face_seeds.keys(), results))


# --- Address type ---------------------------------------------------------


@dataclass(frozen=True)
class Address:
    """A coordinate in the fractal cube address space.

    ``hex_key`` is the portal hash of the attractor at this node (the *hash*
    half of the ``(hash, depth)`` tuple). ``depth`` equals ``len(face_path)`` —
    the address length, which the spec reads as "depth of understanding."
    """

    hex_key: str
    depth: int
    face_path: tuple[int, ...] = field(default=())


def address_of(attractor: np.ndarray) -> Address:
    """Depth-0 coordinate of a basin: its own portal hash, no faces taken."""
    return Address(hex_key=portal_hash(attractor), depth=0, face_path=())


# --- Traversal & resolution ----------------------------------------------


def _evolve_seed(seed_text: str) -> dict:
    """Evolve a seed string to its attractor (always has an 'attractor' key)."""
    return evolve_and_interpret(hash_to_frame(seed_text))


def traverse(seed_text: str, face_sequence: list[int]) -> list[Address]:
    """Walk the address chain, returning one Address per node visited.

    The list has ``len(face_sequence) + 1`` entries: index 0 is the depth-0
    root reached by evolving ``seed_text``, and index i is the node reached
    after taking ``face_sequence[:i]``. Pure function of its inputs — the same
    arguments always produce the same chain.
    """
    for face in face_sequence:
        if not 0 <= face < NUM_FACES:
            raise ValueError(f"face index {face} out of range [0, {NUM_FACES})")

    current_seed = seed_text
    chain: list[Address] = []
    for depth in range(len(face_sequence) + 1):
        result = _evolve_seed(current_seed)
        portal = portal_hash(result["attractor"])
        chain.append(
            Address(
                hex_key=portal,
                depth=depth,
                face_path=tuple(face_sequence[:depth]),
            )
        )
        if depth < len(face_sequence):
            current_seed = expand_cube(portal)[face_sequence[depth]]
    return chain


def resolve(seed_text: str, face_sequence: list[int]) -> dict:
    """Resolve an address to its terminal node.

    Returns ``{"attractor", "state", "convergence_ticks", "address"}`` for the
    node at ``depth == len(face_sequence)``. An empty ``face_sequence`` returns
    the depth-0 attractor (no faces taken).
    """
    for face in face_sequence:
        if not 0 <= face < NUM_FACES:
            raise ValueError(f"face index {face} out of range [0, {NUM_FACES})")

    current_seed = seed_text
    for depth in range(len(face_sequence) + 1):
        result = _evolve_seed(current_seed)
        portal = portal_hash(result["attractor"])
        if depth == len(face_sequence):
            return {
                "attractor": result["attractor"],
                "state": result["state"],
                "convergence_ticks": result["convergence_ticks"],
                "address": Address(
                    hex_key=portal,
                    depth=depth,
                    face_path=tuple(face_sequence),
                ),
            }
        current_seed = expand_cube(portal)[face_sequence[depth]]
    raise AssertionError("unreachable")  # loop always returns at the terminal depth


# --- Recall-path bridge ---------------------------------------------------


def recognize_address(query: str, **recognize_kwargs):
    """Recognize a basin and resolve its FCAS depth-0 address.

    Wraps ``recall_api.recognize`` (CANON.md §9 item #1: wire the (hash, depth)
    tuple keys into the recall path). Returns ``(seed, address)``; both are
    ``None`` on a recognition miss. The address's ``hex_key`` is the portal hash
    of the matched stored attractor — the coordinate at which that memory lives.
    """
    from . import recall_api

    seed = recall_api.recognize(query, **recognize_kwargs)
    if seed is None:
        return None, None

    attractor_path = (
        Path(seed.data_dir)
        / "chunks"
        / seed.chunk
        / "attractors"
        / f"{seed.hex_key}.npy"
    )
    stored = np.load(attractor_path)
    return seed, address_of(stored)
