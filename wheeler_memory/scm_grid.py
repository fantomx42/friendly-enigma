"""Structural Coherence Map (SCM) — persistent spatial trust topology.

The SCM grid is the invisible waveguide between the corpus and experiential
grids.  It stores WHERE interference is permitted — not content, but permission.

Properties:
  - 64x64 float32 grid, values in [-1, 1].  0 = fully permissive (untested).
  - Hardening: each cell tracks its update count.  Well-tested regions resist
    future change proportional to their update count.
  - Two feedback pathways write to the SCM:
    1. Self-consistency check (coincidence erosion) — opens/closes based on output fidelity.
    2. Recall-driven feedback — adjusts gate magnitude based on recall quality (κ).
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
    SCM_KAPPA_EMA_RATE,
    SCM_OPEN_FRACTION_CEIL,
    SCM_RECALL_LR,
    SCM_RECALL_UPDATE_CAP,
)

GRID_SIZE = 64


class SCMGrid:
    """Persistent spatial trust topology for the three-grid interference system."""

    def __init__(
        self,
        grid: np.ndarray,
        hardening: np.ndarray,
        data_dir: Path,
        kappa_base: float = 0.0,
    ) -> None:
        self.grid = grid
        self.hardening = hardening
        self.kappa_base = kappa_base
        self._data_dir = data_dir
        self._grid_path = data_dir / "scm_grid.npy"
        self._hardening_path = data_dir / "scm_hardening.npy"
        self._kappa_path = data_dir / "scm_kappa_base.npy"

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

        kappa_path = data_dir / "scm_kappa_base.npy"
        kappa_base = float(np.load(kappa_path)) if kappa_path.exists() else 0.0

        return cls(grid=grid, hardening=hardening, data_dir=data_dir, kappa_base=kappa_base)

    def save(self) -> None:
        """Atomic save — write to tmp files, then rename."""
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # np.save appends .npy if missing, so use a .tmp prefix to avoid that
        tmp_grid = self._grid_path.with_name("_scm_grid_tmp.npy")
        tmp_hard = self._hardening_path.with_name("_scm_hardening_tmp.npy")
        tmp_kappa = self._kappa_path.with_name("_scm_kappa_base_tmp.npy")

        np.save(tmp_grid, self.grid)
        np.save(tmp_hard, self.hardening)
        np.save(tmp_kappa, np.float64(self.kappa_base))

        tmp_grid.replace(self._grid_path)
        tmp_hard.replace(self._hardening_path)
        tmp_kappa.replace(self._kappa_path)

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

    def update_from_recall(
        self,
        corpus_att: np.ndarray,
        experiential_att: np.ndarray,
        kappa: float,
    ) -> int:
        """Apply outcome-driven feedback to the SCM based on recall quality.

        ΔM_i = -η · (κ - κ_base) · sign(M_i) · |C_i · X_i|

        When κ > κ_base (good recall), cells where both content grids peaked
        get driven toward ε_floor (open further — reinforce what worked).
        When κ < κ_base (bad recall), those cells get driven toward ±1
        (close — suppress what let noise through).

        Cells where M_i = 0 (untouched by self-consistency) get no update
        because sign(0) = 0.  Only cells with existing opinions are adjusted.

        Parameters
        ----------
        corpus_att : (64, 64) float32 — corpus attractor of the top recalled memory.
        experiential_att : (64, 64) float32 — experiential attractor (zeros if absent).
        kappa : float — recall quality score (top interference score).

        Returns
        -------
        int — number of cells updated.
        """
        # Advantage over rolling baseline
        advantage = kappa - self.kappa_base

        # Update kappa EMA (always, even if we skip the grid update)
        self.kappa_base = (
            (1 - SCM_KAPPA_EMA_RATE) * self.kappa_base
            + SCM_KAPPA_EMA_RATE * kappa
        )

        # Homeostasis: if grid is too open, only allow closing (advantage < 0)
        if self.openness() >= SCM_OPEN_FRACTION_CEIL and advantage > 0:
            return 0

        # Credit assignment: |C_i · X_i| — only cells in the interference path
        credit = np.abs(corpus_att * experiential_att)

        # Direction: sign(M_i) — fresh cells (M=0) get sign=0 → no update
        m_sign = np.sign(self.grid)

        # Mask: cells with existing opinion AND nonzero credit
        mask = (m_sign != 0) & (credit > 0)
        if not mask.any():
            return 0

        # Compute raw deltas
        delta = np.zeros_like(self.grid)
        delta[mask] = -SCM_RECALL_LR * advantage * m_sign[mask] * credit[mask]

        # Per-recall cap: Σ|ΔM| ≤ α · N · cap_frac
        total_delta = np.abs(delta[mask]).sum()
        cap = SCM_LEARNING_RATE * self.grid.size * SCM_RECALL_UPDATE_CAP
        if total_delta > cap:
            delta *= cap / total_delta

        # Apply deltas with sign preservation
        original_signs = m_sign[mask].copy()
        new_vals = self.grid[mask] + delta[mask]

        # Prevent sign flip: if update overshoots zero, clamp at ε_floor
        sign_changed = np.sign(new_vals) != original_signs
        new_vals = np.where(
            sign_changed,
            original_signs * SCM_HARDENING_FLOOR,
            new_vals,
        )

        # Clamp |M| ∈ [ε_floor, 1], preserving sign
        signs = np.sign(new_vals)
        new_vals = signs * np.clip(np.abs(new_vals), SCM_HARDENING_FLOOR, 1.0)

        self.grid[mask] = new_vals.astype(np.float32)

        return int(mask.sum())

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
