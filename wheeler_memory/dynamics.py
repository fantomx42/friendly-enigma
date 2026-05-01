"""Core cellular automata engine for Wheeler Memory.

Implements the 3-state CA dynamics: local max cells push toward +1,
local min cells push toward -1, slope cells flow toward their max neighbor.
"""

import logging

import numpy as np

from .constants import (
    ALIVE_THRESHOLD,
    CONVERGENCE_PERCENTILE,
    MAX_PUSH_STRENGTH,
    MIN_ALIVE_FRACTION,
    SALIENCE_MAX_ITERS_MED,
    SALIENCE_THRESHOLD_MED,
    SLOPE_FLOW_STRENGTH,
    STABILITY_WINDOW,
)
from .oscillation import detect_oscillation


def apply_ca_dynamics(frame: np.ndarray) -> np.ndarray:
    """Apply a single CA iteration using 3-state logic.

    Update rules (von Neumann neighborhood, wrapping boundaries):
      - Local max (>= all 4 neighbors): delta = (1 - cell) * 0.35
      - Local min (<= all 4 neighbors): delta = (-1 - cell) * 0.35
      - Slope (neither): delta = (max_neighbor - cell) * 0.20
    """
    n_up = np.roll(frame, 1, axis=0)
    n_down = np.roll(frame, -1, axis=0)
    n_left = np.roll(frame, 1, axis=1)
    n_right = np.roll(frame, -1, axis=1)

    is_max = (
        (frame >= n_up) & (frame >= n_down) & (frame >= n_left) & (frame >= n_right)
    )
    is_min = (
        (frame <= n_up) & (frame <= n_down) & (frame <= n_left) & (frame <= n_right)
    )

    neighbors = np.stack([n_up, n_down, n_left, n_right])
    max_neighbor = np.max(neighbors, axis=0)

    is_flat = is_max & is_min  # all neighbours equal — no gradient, no movement

    delta = np.zeros_like(frame)
    delta = np.where(is_max & ~is_flat, (1 - frame) * MAX_PUSH_STRENGTH, delta)
    delta = np.where(is_min & ~is_flat, (-1 - frame) * MAX_PUSH_STRENGTH, delta)
    delta = np.where(
        ~is_max & ~is_min, (max_neighbor - frame) * SLOPE_FLOW_STRENGTH, delta
    )
    # is_flat cells: delta stays 0 — uniform plateau is a stable fixed point

    return np.clip(frame + delta, -1, 1)


def apply_ca_dynamics_parameterized(
    frame: np.ndarray,
    push_strength: float,
    slope_strength: float,
) -> np.ndarray:
    """Apply a single CA iteration with caller-supplied push/slope strengths.

    Identical logic to apply_ca_dynamics() but accepts per-grid constants,
    enabling corpus (tight) and experiential (loose) evolution regimes
    from the same engine.
    """
    n_up = np.roll(frame, 1, axis=0)
    n_down = np.roll(frame, -1, axis=0)
    n_left = np.roll(frame, 1, axis=1)
    n_right = np.roll(frame, -1, axis=1)

    is_max = (
        (frame >= n_up) & (frame >= n_down) & (frame >= n_left) & (frame >= n_right)
    )
    is_min = (
        (frame <= n_up) & (frame <= n_down) & (frame <= n_left) & (frame <= n_right)
    )

    neighbors = np.stack([n_up, n_down, n_left, n_right])
    max_neighbor = np.max(neighbors, axis=0)

    is_flat = is_max & is_min

    delta = np.zeros_like(frame)
    delta = np.where(is_max & ~is_flat, (1 - frame) * push_strength, delta)
    delta = np.where(is_min & ~is_flat, (-1 - frame) * push_strength, delta)
    delta = np.where(
        ~is_max & ~is_min, (max_neighbor - frame) * slope_strength, delta
    )

    return np.clip(frame + delta, -1, 1)


# GPU dispatch — imported after apply_ca_dynamics is defined.
try:
    from .accel.ca import (
        gpu_available,
        gpu_evolve_batch as _gpu_evolve_batch,
        gpu_evolve_single as _gpu_evolve,
    )

    _GPU_READY = gpu_available()
except ImportError:
    _GPU_READY = False
    _gpu_evolve = None
    _gpu_evolve_batch = None


def _enrich_metadata(result: dict) -> None:
    """Add structural features to an evolution result's metadata in-place.

    Works for both GPU and CPU paths — merges grid features and
    convergence speed into the existing metadata dict.
    """
    md = result.get("metadata", {})
    feats = compute_attractor_features(result["attractor"])
    md.update(feats)
    ticks = result.get("convergence_ticks", 0)
    if result.get("state") == "CONVERGED":
        md["convergence_speed"] = "F" if ticks <= 50 else ("M" if ticks <= 150 else "S")
    else:
        md["convergence_speed"] = "?"
    md.setdefault("final_delta", 0.0)
    result["metadata"] = md


def evolve_and_interpret(
    frame: np.ndarray,
    max_iters: int = SALIENCE_MAX_ITERS_MED,
    stability_threshold: float = SALIENCE_THRESHOLD_MED,
) -> dict:
    """Run CA evolution until convergence, oscillation, or chaos.

    Returns dict with keys:
      - state: 'CONVERGED' | 'OSCILLATING' | 'CHAOTIC'
      - attractor: final frame (for CONVERGED)
      - convergence_ticks: number of iterations
      - history: list of all frame copies (for brick construction)
      - metadata: additional info (cycle_period, etc.)
    """
    if _GPU_READY and _gpu_evolve is not None:
        try:
            result = _gpu_evolve(
                frame,
                max_iters=max_iters,
                stability_threshold=stability_threshold,
            )
            # GPU path doesn't store per-tick history; synthesize a minimal
            # 2-frame history so MemoryBrick.save() (np.stack) doesn't crash.
            if not result["history"]:
                result["history"] = [frame.copy(), result["attractor"].copy()]
            _enrich_metadata(result)
            return result
        except Exception as e:
            logging.warning("GPU evolution failed, falling back to CPU: %s", e)

    history = [frame.copy()]
    stable_count = 0
    delta = 0.0

    for i in range(max_iters):
        frame_old = frame
        frame = apply_ca_dynamics(frame)
        abs_deltas = np.abs(frame - frame_old)
        delta = float(np.percentile(abs_deltas, CONVERGENCE_PERCENTILE))

        history.append(frame.copy())

        if delta < stability_threshold:
            # Gate: reject convergence if the attractor is 0-dominant.
            # On a 4096-cell grid, p99 can be 0 when <1% of cells are active,
            # causing false CONVERGED in STABILITY_WINDOW ticks.
            alive = float(np.mean(np.abs(frame) > ALIVE_THRESHOLD))
            if alive < MIN_ALIVE_FRACTION:
                stable_count = 0
            else:
                stable_count += 1
                if stable_count >= STABILITY_WINDOW:
                    ticks = i + 1
                    feats = compute_attractor_features(frame)
                    spd = "F" if ticks <= 50 else ("M" if ticks <= 150 else "S")
                    return {
                        "state": "CONVERGED",
                        "attractor": frame,
                        "convergence_ticks": ticks,
                        "history": history,
                        "metadata": {
                            **feats,
                            "convergence_speed": spd,
                            "final_delta": round(delta, 6),
                        },
                    }
        else:
            stable_count = 0

        if i > 50 and i % 10 == 0:
            osc_result = detect_oscillation(history)
            if osc_result["oscillating"]:
                ticks = i + 1
                feats = compute_attractor_features(frame)
                return {
                    "state": "OSCILLATING",
                    "attractor": frame,
                    "convergence_ticks": ticks,
                    "history": history,
                    "metadata": {
                        "cycle_period": osc_result["period"],
                        "oscillating_cells": osc_result["oscillating_cells"],
                        "cycle_states": osc_result["cycle_states"],
                        **feats,
                        "convergence_speed": "?",
                        "final_delta": round(delta, 6),
                    },
                }

    feats = compute_attractor_features(frame)
    final_alive = feats.get("alive_fraction", 1.0)
    final_state = "DEGENERATE" if final_alive < MIN_ALIVE_FRACTION else "CHAOTIC"
    return {
        "state": final_state,
        "attractor": frame,
        "convergence_ticks": max_iters,
        "history": history,
        "metadata": {
            **feats,
            "convergence_speed": "?",
            "final_delta": round(delta, 6),
        },
    }


def evolve_batch(
    frames: list[np.ndarray],
    max_iters: int = SALIENCE_MAX_ITERS_MED,
    stability_threshold: float = SALIENCE_THRESHOLD_MED,
) -> list[dict]:
    """Evolve multiple frames, using GPU batch dispatch when available.

    On GPU this is dramatically faster than serial evolve_and_interpret()
    calls (71x at batch=1000 on RX 9070 XT). Falls back to serial CPU
    loop when GPU is not available.

    Returns list of dicts with same keys as evolve_and_interpret().
    """
    if not frames:
        return []

    if _GPU_READY and _gpu_evolve_batch is not None:
        try:
            results = _gpu_evolve_batch(
                frames,
                max_iters=max_iters,
                stability_threshold=stability_threshold,
            )
            for i, result in enumerate(results):
                if not result["history"]:
                    result["history"] = [frames[i].copy(), result["attractor"].copy()]
                _enrich_metadata(result)
            return results
        except Exception as e:
            logging.warning("GPU batch evolution failed, falling back to CPU: %s", e)

    return [
        evolve_and_interpret(f, max_iters=max_iters, stability_threshold=stability_threshold)
        for f in frames
    ]


def evolve_batch_with_params(
    frames: list[np.ndarray],
    push_strength: float,
    slope_strength: float,
    max_iters: int = SALIENCE_MAX_ITERS_MED,
    stability_threshold: float = SALIENCE_THRESHOLD_MED,
) -> list[dict]:
    """Evolve multiple frames with custom push/slope, GPU batch when available."""
    if not frames:
        return []

    if _GPU_READY and _gpu_evolve_batch is not None:
        try:
            results = _gpu_evolve_batch(
                frames,
                max_iters=max_iters,
                stability_threshold=stability_threshold,
                push_strength=push_strength,
                slope_strength=slope_strength,
            )
            for i, result in enumerate(results):
                if not result["history"]:
                    result["history"] = [frames[i].copy(), result["attractor"].copy()]
                _enrich_metadata(result)
            return results
        except Exception as e:
            logging.warning("GPU batch evolution failed, falling back to CPU: %s", e)

    return [
        evolve_with_params(
            f,
            push_strength=push_strength,
            slope_strength=slope_strength,
            max_iters=max_iters,
            stability_threshold=stability_threshold,
        )
        for f in frames
    ]


def evolve_with_params(
    frame: np.ndarray,
    push_strength: float,
    slope_strength: float,
    max_iters: int = SALIENCE_MAX_ITERS_MED,
    stability_threshold: float = SALIENCE_THRESHOLD_MED,
) -> dict:
    """Evolve a frame using custom push/slope strengths.

    Same convergence/oscillation/chaos detection as evolve_and_interpret(),
    but uses apply_ca_dynamics_parameterized() with the given strengths.
    Enables corpus vs experiential evolution from a single code path.

    For GPU: ca_evolve_single_v2 already accepts push/slope as runtime params.
    """
    if _GPU_READY and _gpu_evolve is not None:
        try:
            result = _gpu_evolve(
                frame,
                max_iters=max_iters,
                stability_threshold=stability_threshold,
                push_strength=push_strength,
                slope_strength=slope_strength,
            )
            if not result["history"]:
                result["history"] = [frame.copy(), result["attractor"].copy()]
            _enrich_metadata(result)
            return result
        except Exception as e:
            logging.warning("GPU evolution failed, falling back to CPU: %s", e)

    history = [frame.copy()]
    stable_count = 0
    delta = 0.0

    for i in range(max_iters):
        frame_old = frame
        frame = apply_ca_dynamics_parameterized(frame, push_strength, slope_strength)
        abs_deltas = np.abs(frame - frame_old)
        delta = float(np.percentile(abs_deltas, CONVERGENCE_PERCENTILE))

        history.append(frame.copy())

        if delta < stability_threshold:
            alive = float(np.mean(np.abs(frame) > ALIVE_THRESHOLD))
            if alive < MIN_ALIVE_FRACTION:
                stable_count = 0
            else:
                stable_count += 1
                if stable_count >= STABILITY_WINDOW:
                    ticks = i + 1
                    feats = compute_attractor_features(frame)
                    spd = "F" if ticks <= 50 else ("M" if ticks <= 150 else "S")
                    return {
                        "state": "CONVERGED",
                        "attractor": frame,
                        "convergence_ticks": ticks,
                        "history": history,
                        "metadata": {
                            **feats,
                            "convergence_speed": spd,
                            "final_delta": round(delta, 6),
                        },
                    }
        else:
            stable_count = 0

        if i > 50 and i % 10 == 0:
            osc_result = detect_oscillation(history)
            if osc_result["oscillating"]:
                ticks = i + 1
                feats = compute_attractor_features(frame)
                return {
                    "state": "OSCILLATING",
                    "attractor": frame,
                    "convergence_ticks": ticks,
                    "history": history,
                    "metadata": {
                        "cycle_period": osc_result["period"],
                        "oscillating_cells": osc_result["oscillating_cells"],
                        "cycle_states": osc_result["cycle_states"],
                        **feats,
                        "convergence_speed": "?",
                        "final_delta": round(delta, 6),
                    },
                }

    feats = compute_attractor_features(frame)
    final_alive = feats.get("alive_fraction", 1.0)
    final_state = "DEGENERATE" if final_alive < MIN_ALIVE_FRACTION else "CHAOTIC"
    return {
        "state": final_state,
        "attractor": frame,
        "convergence_ticks": max_iters,
        "history": history,
        "metadata": {
            **feats,
            "convergence_speed": "?",
            "final_delta": round(delta, 6),
        },
    }


def snap_to_ternary(frame: np.ndarray, mode: str | None = None) -> np.ndarray:
    """Snap a continuous [-1, 1] attractor frame to {-1, 0, +1}.

    Parameters
    ----------
    frame : (64, 64) float array — a converged attractor.
    mode  : "topological" uses neighborhood-based cell roles (local max → +1,
            local min → -1, slope → 0).  "threshold" buckets by value magnitude.
            Defaults to constants.TERNARY_SNAP_MODE.

    Returns
    -------
    (64, 64) int8 array with values in {-1, 0, +1}.
    """
    from .constants import (
        TERNARY_SNAP_MODE,
        TERNARY_THRESHOLD_POS,
        TERNARY_THRESHOLD_NEG,
    )

    if mode is None:
        mode = TERNARY_SNAP_MODE

    if mode == "topological":
        from .oscillation import get_cell_roles

        return get_cell_roles(frame)
    elif mode == "threshold":
        out = np.zeros(frame.shape, dtype=np.int8)
        out[frame > TERNARY_THRESHOLD_POS] = 1
        out[frame < TERNARY_THRESHOLD_NEG] = -1
        return out
    else:
        raise ValueError(f"Unknown ternary snap mode: {mode!r}")


def _count_clusters(mask: np.ndarray) -> int:
    """Count connected components in a boolean mask (von Neumann neighbourhood)."""
    visited = np.zeros_like(mask)
    rows, cols = mask.shape
    clusters = 0
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] and not visited[r, c]:
                clusters += 1
                queue = [(r, c)]
                while queue:
                    cr, cc = queue.pop()
                    if visited[cr, cc]:
                        continue
                    visited[cr, cc] = True
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and not visited[nr, nc]
                            and mask[nr, nc]
                        ):
                            queue.append((nr, nc))
    return clusters


def compute_attractor_features(grid: np.ndarray) -> dict:
    """Structural features of a converged attractor grid.

    Returns
    -------
    dict with:
        grid_entropy      : Shannon entropy (bits) over the 3-state distribution
        cluster_count     : connected components of positive cells (von Neumann)
        neg_cluster_count : connected components of negative cells
        alive_fraction    : fraction of non-neutral cells
        boundary_length   : positive cells adjacent to negative cells (4-neighbour)
        energy            : mean |delta| after one CA step (basin depth)
    """
    # Discretise continuous [-1, 1] values into three buckets for statistics
    pos = grid > 0.33  # "positive" state
    neg = grid < -0.33  # "negative" state
    neutral = ~pos & ~neg

    n = grid.size
    counts = np.array([neg.sum(), neutral.sum(), pos.sum()], dtype=np.float64)
    probs = counts / n
    entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))

    # Alive fraction: cells not in the neutral bucket
    alive = float((pos.sum() + neg.sum()) / n)

    # Connected components via BFS (von Neumann neighbourhood)
    clusters = _count_clusters(pos)
    neg_clusters = _count_clusters(neg)

    # Boundary length: positive cells with at least one negative 4-neighbour
    rows, cols = grid.shape
    boundary = 0
    for r in range(rows):
        for c in range(cols):
            if pos[r, c]:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = (r + dr) % rows, (c + dc) % cols
                    if neg[nr, nc]:
                        boundary += 1
                        break

    # Energy: mean absolute change after one CA step (low = deep basin)
    next_state = apply_ca_dynamics(grid)
    e = float(np.abs(next_state - grid).mean())

    return {
        "grid_entropy": round(entropy, 3),
        "cluster_count": clusters,
        "neg_cluster_count": neg_clusters,
        "alive_fraction": round(alive, 3),
        "boundary_length": boundary,
        "energy": round(e, 6),
    }


def extract_spatial_features(grid: np.ndarray) -> np.ndarray:
    """Extract fixed-length spatial topology feature vector from a 64x64 grid.

    Captures cluster count, size distribution, centroids, and boundary length
    for both +1 and -1 islands.  Returns a flat float32 vector suitable for
    cosine similarity comparison.

    Feature layout (51 elements):
      [0:1]   pos_count/50, neg_count/50
      [2:10]  top-8 positive cluster sizes / 4096
      [10:18] top-8 negative cluster sizes / 4096
      [18:34] top-8 positive cluster centroids (y,x) / 64
      [34:50] top-8 negative cluster centroids (y,x) / 64
      [50]    boundary_length / 1000
    """
    ternary = snap_to_ternary(grid, mode="threshold")
    rows, cols = ternary.shape
    K = 8  # max clusters to track

    def _cluster_features(mask):
        visited = np.zeros((rows, cols), dtype=np.bool_)
        clusters = []
        for r in range(rows):
            for c in range(cols):
                if mask[r, c] and not visited[r, c]:
                    queue = [(r, c)]
                    cells_r, cells_c = [], []
                    while queue:
                        cr, cc = queue.pop()
                        if visited[cr, cc]:
                            continue
                        visited[cr, cc] = True
                        cells_r.append(cr)
                        cells_c.append(cc)
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = (cr + dr) % rows, (cc + dc) % cols
                            if not visited[nr, nc] and mask[nr, nc]:
                                queue.append((nr, nc))
                    clusters.append((
                        len(cells_r),
                        np.mean(cells_r),
                        np.mean(cells_c),
                    ))
        clusters.sort(key=lambda x: -x[0])
        return clusters

    pos_clusters = _cluster_features(ternary == 1)
    neg_clusters = _cluster_features(ternary == -1)

    # Boundary: count +1 cells adjacent to -1 cells (wrapped)
    t = ternary
    pos_mask = t == 1
    boundary = 0
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.roll(np.roll(t, dr, axis=0), dc, axis=1)
        boundary += int(np.sum(pos_mask & (shifted == -1)))

    # Build fixed-length vector
    fv = np.zeros(51, dtype=np.float32)
    fv[0] = len(pos_clusters) / 50.0
    fv[1] = len(neg_clusters) / 50.0

    for i, cl in enumerate(pos_clusters[:K]):
        fv[2 + i] = cl[0] / 4096.0
        fv[18 + 2 * i] = cl[1] / 64.0
        fv[19 + 2 * i] = cl[2] / 64.0

    for i, cl in enumerate(neg_clusters[:K]):
        fv[10 + i] = cl[0] / 4096.0
        fv[34 + 2 * i] = cl[1] / 64.0
        fv[35 + 2 * i] = cl[2] / 64.0

    fv[50] = boundary / 1000.0
    return fv
