"""Trajectory signature cache for vectorized batch similarity search."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .trajectory import TrajectorySignature, load_signature, batch_similarity
from .constants import TRAJECTORY_SIG_DIR


class TrajectoryCache:
    """Pre-loads all stored trajectory signatures into numpy matrices for fast search."""

    def __init__(self, data_dir: Path | None = None):
        """Load all trajectory signatures from chunks/{chunk}/signatures/*.npz.

        Args:
            data_dir: Wheeler Memory data dir (default ~/.wheeler_memory)
        """
        from .storage import DEFAULT_DATA_DIR, list_memories

        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
        d = Path(data_dir)

        t0 = time.time()
        memories = list_memories(data_dir=d)

        self._texts: list[str] = []
        self._hex_keys: list[str] = []
        energy_list = []
        role_list = []
        dist_list = []
        tick_list = []
        corr_list = []

        for m in memories:
            chunk = m.get("chunk", "general")
            hex_key = m["hex_key"]
            sig_path = d / "chunks" / chunk / TRAJECTORY_SIG_DIR / f"{hex_key}.npz"

            if not sig_path.exists():
                continue

            try:
                sig = load_signature(sig_path)
                energy_list.append(sig.energy_curve)
                role_list.append(sig.role_curve)
                dist_list.append(sig.final_role_distribution)
                tick_list.append(sig.convergence_ticks)
                corr_list.append(sig.seed_attractor_corr)
                self._texts.append(m["text"])
                self._hex_keys.append(hex_key)
            except Exception:
                continue

        if energy_list:
            self._energy_curves = np.stack(energy_list).astype(np.float32)  # (N, 10)
            self._role_curves = np.stack(role_list).astype(np.float32)  # (N, 10)
            self._role_dists = np.stack(dist_list).astype(np.float32)  # (N, 3)
            self._ticks = np.array(tick_list, dtype=np.float32)  # (N,)
            self._seed_att_corrs = np.array(corr_list, dtype=np.float32)  # (N,)
        else:
            self._energy_curves = np.zeros((0, 10), dtype=np.float32)
            self._role_curves = np.zeros((0, 10), dtype=np.float32)
            self._role_dists = np.zeros((0, 3), dtype=np.float32)
            self._ticks = np.array([], dtype=np.float32)
            self._seed_att_corrs = np.array([], dtype=np.float32)

        elapsed = time.time() - t0
        print(
            f"TrajectoryCache: loaded {len(self._texts)} signatures in {elapsed:.2f}s"
        )

    def __len__(self) -> int:
        return len(self._texts)

    def search(self, query_sig: TrajectorySignature, top_k: int = 10) -> list[dict]:
        """Find top-K most similar trajectory signatures.

        Args:
            query_sig: Query trajectory signature.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: text, hex_key, traj_similarity.
        """
        if len(self._texts) == 0:
            return []

        sims = batch_similarity(
            query_sig,
            self._energy_curves,
            self._role_curves,
            self._role_dists,
            self._ticks,
            self._seed_att_corrs,
        )

        k = min(top_k, len(sims))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        return [
            {
                "text": self._texts[i],
                "hex_key": self._hex_keys[i],
                "traj_similarity": float(sims[i]),
            }
            for i in top_idx
        ]
