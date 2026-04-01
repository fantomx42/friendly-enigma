"""Tests for wheeler_memory.trajectory_cache — vectorized batch trajectory search."""

import numpy as np
import pytest

from wheeler_memory.trajectory import TrajectorySignature
from wheeler_memory.trajectory_cache import TrajectoryCache


class TestTrajectoryCacheEmpty:
    """TrajectoryCache on empty data dir."""

    def test_empty_cache_length_zero(self, tmp_path):
        """Empty data dir → cache has length 0."""
        # Set up minimal chunk structure
        (tmp_path / "chunks" / "general").mkdir(parents=True)
        cache = TrajectoryCache(data_dir=tmp_path)
        assert len(cache) == 0

    def test_empty_cache_search_returns_empty(self, tmp_path):
        """Search on empty cache → empty list."""
        (tmp_path / "chunks" / "general").mkdir(parents=True)
        cache = TrajectoryCache(data_dir=tmp_path)
        query = TrajectorySignature(
            convergence_ticks=50,
            energy_curve=np.zeros(10, dtype=np.float32),
            role_curve=np.zeros(10, dtype=np.float32),
            final_role_distribution=np.zeros(3, dtype=np.float32),
            state="CONVERGED",
            seed_attractor_corr=0.5,
        )
        results = cache.search(query, top_k=5)
        assert results == []


class TestTrajectoryCacheWithData:
    """TrajectoryCache with pre-seeded signatures."""

    @pytest.fixture
    def seeded_cache(self, tmp_path):
        """Create a cache with two stored signatures via the full pipeline."""
        from wheeler_memory.hashing import hash_to_frame
        from wheeler_memory.dynamics import evolve_and_interpret
        from wheeler_memory.storage import store_memory
        from wheeler_memory.brick import MemoryBrick
        from wheeler_memory.trajectory import compute_signature, save_signature
        from wheeler_memory.chunking import select_chunk
        from wheeler_memory.constants import TRAJECTORY_SIG_DIR

        texts = ["the cat sat on the mat", "quantum physics describes nature"]
        for text in texts:
            frame = hash_to_frame(text)
            result = evolve_and_interpret(frame)
            brick = MemoryBrick.from_evolution_result(result, {"input_text": text})
            hex_key = store_memory(text, result, brick, tmp_path)

            # Save signature to the correct chunk directory
            chunk = select_chunk(text)
            sig = compute_signature(frame)
            chunk_dir = tmp_path / "chunks" / chunk
            sig_dir = chunk_dir / TRAJECTORY_SIG_DIR
            sig_dir.mkdir(parents=True, exist_ok=True)
            save_signature(sig, sig_dir / f"{hex_key}.npz")

        return TrajectoryCache(data_dir=tmp_path)

    def test_cache_loaded_two_sigs(self, seeded_cache):
        """Cache loaded both signatures."""
        assert len(seeded_cache) == 2

    def test_search_returns_results(self, seeded_cache):
        """Search with a query returns matches."""
        query = TrajectorySignature(
            convergence_ticks=30,
            energy_curve=np.random.default_rng(42).random(10).astype(np.float32),
            role_curve=np.random.default_rng(43).random(10).astype(np.float32),
            final_role_distribution=np.array([0.3, 0.5, 0.2], dtype=np.float32),
            state="CONVERGED",
            seed_attractor_corr=0.6,
        )
        results = seeded_cache.search(query, top_k=5)
        assert len(results) > 0
        assert len(results) <= 2  # only 2 stored

    def test_search_result_keys(self, seeded_cache):
        """Each result has text, hex_key, traj_similarity."""
        query = TrajectorySignature(
            convergence_ticks=50,
            energy_curve=np.ones(10, dtype=np.float32),
            role_curve=np.ones(10, dtype=np.float32),
            final_role_distribution=np.array([0.5, 0.3, 0.2], dtype=np.float32),
            state="CONVERGED",
            seed_attractor_corr=0.5,
        )
        results = seeded_cache.search(query, top_k=5)
        for r in results:
            assert "text" in r
            assert "hex_key" in r
            assert "traj_similarity" in r
            assert isinstance(r["traj_similarity"], float)

    def test_search_similarity_sorted_descending(self, seeded_cache):
        """Results are sorted by similarity, highest first."""
        query = TrajectorySignature(
            convergence_ticks=10,
            energy_curve=np.ones(10, dtype=np.float32),
            role_curve=np.zeros(10, dtype=np.float32),
            final_role_distribution=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            state="CONVERGED",
            seed_attractor_corr=0.9,
        )
        results = seeded_cache.search(query, top_k=5)
        if len(results) >= 2:
            assert results[0]["traj_similarity"] >= results[1]["traj_similarity"]

    def test_top_k_limits_results(self, seeded_cache):
        """top_k=1 returns at most 1 result."""
        query = TrajectorySignature(
            convergence_ticks=50,
            energy_curve=np.ones(10, dtype=np.float32),
            role_curve=np.ones(10, dtype=np.float32),
            final_role_distribution=np.array([0.3, 0.3, 0.4], dtype=np.float32),
            state="CONVERGED",
            seed_attractor_corr=0.5,
        )
        results = seeded_cache.search(query, top_k=1)
        assert len(results) == 1
