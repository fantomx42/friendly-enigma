"""Structural Coherence Map (SCM) — persistent spatial trust topology.

The SCM grid is the invisible waveguide between the corpus and experiential
grids.  It stores WHERE interference is permitted — not content, but permission.

Properties:
  - 64x64 float32 grid, values in [-1, 1].  0 = fully permissive (untested).
  - Hardening: each cell tracks its update count.  Well-tested regions resist
    future change proportional to their update count.
  - Only the self-consistency feedback loop writes to the SCM.
  - The SCM does NOT evolve autonomously (no CA dynamics).
  - Persists across power cycles (numpy files on disk).

Answer equation: Answer(i,j) = Corpus(i,j) * Experiential(i,j) * (1 - |SCM(i,j)|)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import (
    SCM_ANNEAL_RATE,
    SCM_GAP_THRESHOLD,
    SCM_HARDENING_FLOOR,
    SCM_LEARNING_RATE,
)

GRID_SIZE = 64


class SCMGrid:
    """Persistent spatial trust topology for the three-grid interference system."""

    def __init__(
        self,
        grid: np.ndarray,
        hardening: np.ndarray,
        data_dir: Path,
    ) -> None:
        self.grid = grid
        self.hardening = hardening
        self._data_dir = data_dir
        self._grid_path = data_dir / "scm_grid.npy"
        self._hardening_path = data_dir / "scm_hardening.npy"

    @classmethod
    def load_or_create(cls, data_dir: str | Path | None = None) -> SCMGrid:
        """Load an existing SCM grid from disk, or initialize to zeros (fully permissive)."""
        if data_dir is None:
            data_dir = Path.home() / ".wheeler_memory"
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        grid_path = data_dir / "scm_grid.npy"
        hardening_path = data_dir / "scm_hardening.npy"

        if grid_path.exists() and hardening_path.exists():
            grid = np.load(grid_path).astype(np.float32)
            hardening = np.load(hardening_path).astype(np.uint32)
        else:
            grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            hardening = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint32)

        return cls(grid=grid, hardening=hardening, data_dir=data_dir)

    def save(self) -> None:
        """Atomic save — write to tmp files, then rename."""
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # np.save appends .npy if missing, so use a .tmp prefix to avoid that
        tmp_grid = self._grid_path.with_name("_scm_grid_tmp.npy")
        tmp_hard = self._hardening_path.with_name("_scm_hardening_tmp.npy")

        np.save(tmp_grid, self.grid)
        np.save(tmp_hard, self.hardening)

        tmp_grid.replace(self._grid_path)
        tmp_hard.replace(self._hardening_path)

    def update(self, mask: np.ndarray, direction: np.ndarray) -> None:
        """Apply self-consistency feedback to the SCM.

        Parameters
        ----------
        mask : (64, 64) boolean — cells to update (where content peaks aligned).
        direction : float scalar or (64, 64) array.
            +1 = close gap (inconsistent output), -1 = open gap (consistent output).

        Update rule: grid[mask] += direction * LR / (1 + hardening[mask])
        Then hardening[mask] += 1.  Clip grid to [-1, 1].

        The denominator (1 + hardening) is the hardening mechanism — cells with
        many prior updates resist further change.  The learning rate floor
        ensures updates never reach absolute zero.
        """
        if not mask.any():
            return

        direction = np.broadcast_to(
            np.asarray(direction, dtype=np.float32),
            self.grid.shape,
        )

        effective_lr = np.maximum(
            SCM_LEARNING_RATE / (1.0 + self.hardening[mask].astype(np.float32)),
            SCM_HARDENING_FLOOR,
        )

        self.grid[mask] += direction[mask] * effective_lr
        self.grid = np.clip(self.grid, -1, 1)
        self.hardening[mask] += 1

    def gap_mask(self, threshold: float = SCM_GAP_THRESHOLD) -> np.ndarray:
        """Boolean mask where |SCM| < threshold — open gaps permitting interference."""
        return np.abs(self.grid) < threshold

    def openness(self, threshold: float = SCM_GAP_THRESHOLD) -> float:
        """Fraction of cells below the gap threshold (diagnostic)."""
        return float(self.gap_mask(threshold).mean())

    def anneal(self, rate: float = SCM_ANNEAL_RATE) -> int:
        """Soften hardened regions by decaying update counts.

        Called during sleep cycles.  Reduces hardening counts by the given
        fraction so that trust regions gradually soften if not reinforced.

        Returns the number of cells whose hardening was reduced.
        """
        active = self.hardening > 0
        if not active.any():
            return 0
        decay = np.maximum((self.hardening[active] * rate).astype(np.uint32), 1)
        self.hardening[active] = np.where(
            self.hardening[active] > decay,
            self.hardening[active] - decay,
            0,
        ).astype(np.uint32)
        return int(active.sum())

    def stats(self) -> dict:
        """Diagnostic statistics for the SCM grid."""
        gap = self.gap_mask()
        return {
            "openness": round(float(gap.mean()), 4),
            "mean_abs_scm": round(float(np.abs(self.grid).mean()), 4),
            "max_abs_scm": round(float(np.abs(self.grid).max()), 4),
            "mean_hardening": round(float(self.hardening.mean()), 2),
            "max_hardening": int(self.hardening.max()),
            "hardened_cells": int((self.hardening > 0).sum()),
            "closed_cells": int((~gap).sum()),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"SCMGrid(openness={s['openness']:.1%}, "
            f"hardened={s['hardened_cells']}/{GRID_SIZE**2}, "
            f"max_hardening={s['max_hardening']})"
        )
