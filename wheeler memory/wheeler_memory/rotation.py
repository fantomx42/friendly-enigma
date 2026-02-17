"""Rotation retry logic for escaping bad attractor basins.

When a CA evolution fails to converge, rotating the initial frame
by 90/180/270 degrees changes the neighbor topology and can lead
to convergence on a different dynamical trajectory.
"""

import json
import time
from pathlib import Path

import numpy as np

from .brick import MemoryBrick
from .dynamics import evolve_and_interpret
from .hashing import hash_to_frame
from .storage import DEFAULT_DATA_DIR, store_memory


def _load_rotation_stats(data_dir: Path) -> dict:
    path = data_dir / "rotation_stats.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"0": 0, "90": 0, "180": 0, "270": 0}


def _save_rotation_stats(data_dir: Path, stats: dict) -> None:
    path = data_dir / "rotation_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))


def update_rotation_stats(angle: int, success: bool, data_dir: str | Path | None = None) -> None:
    """Track per-angle success counts."""
    d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    stats = _load_rotation_stats(d)
    if success:
        stats[str(angle)] = stats.get(str(angle), 0) + 1
    _save_rotation_stats(d, stats)


def store_with_rotation_retry(
    text: str,
    max_rotations: int = 4,
    save: bool = True,
    data_dir: str | Path | None = None,
) -> dict:
    """Try 0/90/180/270 degree rotations, return first converged result.

    Returns dict with:
      - state, attractor, convergence_ticks, history, metadata
      - metadata includes rotation_used, attempts, wall_time_seconds
    """
    d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    base_frame = hash_to_frame(text)
    angles = [0, 90, 180, 270][:max_rotations]
    last_result = None

    for i, angle in enumerate(angles):
        k = angle // 90
        frame = np.rot90(base_frame, k=k) if k > 0 else base_frame.copy()

        start = time.time()
        result = evolve_and_interpret(frame)
        wall_time = time.time() - start

        result["metadata"]["rotation_used"] = angle
        result["metadata"]["attempts"] = i + 1
        result["metadata"]["wall_time_seconds"] = wall_time

        if result["state"] == "CONVERGED":
            update_rotation_stats(angle, True, d)
            if save:
                brick = MemoryBrick.from_evolution_result(result, {"input_text": text})
                store_memory(text, result, brick, d)
            return result

        update_rotation_stats(angle, False, d)
        last_result = result

    # All rotations failed -- return best attempt
    last_result["state"] = "FAILED_ALL_ROTATIONS"
    return last_result
