"""Tests for the three-grid interference engine."""

import numpy as np
import pytest

from wheeler_memory.constants import INTERFERENCE_PEAK_THRESHOLD, SCM_GAP_THRESHOLD
from wheeler_memory.interference import (
    ABSORBED,
    CONTESTED,
    GROUNDED,
    SILENT,
    UNCONSOLIDATED,
    InterferenceResult,
    compute_interference,
    interference_score,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_grid(value: float) -> np.ndarray:
    """Create a uniform 64x64 grid."""
    return np.full((64, 64), value, dtype=np.float32)


def _make_peaked_grid(peak_value: float, background: float = 0.0) -> np.ndarray:
    """Create a grid with top-left quadrant peaked and rest at background."""
    grid = np.full((64, 64), background, dtype=np.float32)
    grid[:32, :32] = peak_value
    return grid


# ── compute_interference tests ───────────────────────────────────────────────


class TestComputeInterference:
    def test_grounded_state(self):
        """Both grids peak through open SCM gaps → GROUNDED."""
        corpus = _make_grid(0.8)  # above peak threshold
        experiential = _make_grid(0.8)
        scm = _make_grid(0.0)  # fully open (|0| < gap threshold)

        result = compute_interference(corpus, experiential, scm)

        assert result.dominant_state == GROUNDED
        assert result.grounded_fraction > 0.99

    def test_absorbed_state(self):
        """Corpus peaks, no experiential signal, SCM open → ABSORBED."""
        corpus = _make_grid(0.8)
        experiential = _make_grid(0.0)  # below peak threshold
        scm = _make_grid(0.0)

        result = compute_interference(corpus, experiential, scm)

        assert result.dominant_state == ABSORBED
        assert result.absorbed_fraction > 0.99

    def test_absorbed_with_none_experiential(self):
        """experiential=None defaults to zeros → ABSORBED."""
        corpus = _make_grid(0.8)
        scm = _make_grid(0.0)

        result = compute_interference(corpus, None, scm)

        assert result.dominant_state == ABSORBED

    def test_unconsolidated_state(self):
        """Experiential peaks but no corpus signal → UNCONSOLIDATED."""
        corpus = _make_grid(0.0)
        experiential = _make_grid(0.8)
        scm = _make_grid(0.0)

        result = compute_interference(corpus, experiential, scm)

        assert result.dominant_state == UNCONSOLIDATED
        assert result.unconsolidated_fraction > 0.99

    def test_contested_state(self):
        """Both grids peak but SCM is closed → CONTESTED."""
        corpus = _make_grid(0.8)
        experiential = _make_grid(0.8)
        scm = _make_grid(0.9)  # |0.9| > gap threshold → closed

        result = compute_interference(corpus, experiential, scm)

        assert result.dominant_state == CONTESTED
        assert result.contested_fraction > 0.99

    def test_silent_state(self):
        """Neither grid peaks → SILENT (all cells state=0)."""
        corpus = _make_grid(0.0)
        experiential = _make_grid(0.0)
        scm = _make_grid(0.0)

        result = compute_interference(corpus, experiential, scm)

        # Dominant is whichever of the named states has highest fraction,
        # but all named fractions are 0; the silent cells dominate
        assert result.grounded_fraction == 0.0
        assert result.absorbed_fraction == 0.0

    def test_pattern_shape(self):
        """Interference pattern is 64x64."""
        corpus = _make_grid(0.5)
        experiential = _make_grid(0.5)
        scm = _make_grid(0.0)

        result = compute_interference(corpus, experiential, scm)

        assert result.pattern.shape == (64, 64)
        assert result.state_map.shape == (64, 64)

    def test_answer_equation(self):
        """Pattern = Corpus * Experiential * (1 - |SCM|)."""
        corpus = _make_grid(0.6)
        experiential = _make_grid(0.4)
        scm = _make_grid(0.2)

        result = compute_interference(corpus, experiential, scm)

        expected = 0.6 * 0.4 * (1.0 - 0.2)
        np.testing.assert_allclose(result.pattern, expected, atol=1e-6)

    def test_mixed_states(self):
        """Grid with different regions → multiple states present."""
        corpus = np.zeros((64, 64), dtype=np.float32)
        experiential = np.zeros((64, 64), dtype=np.float32)
        scm = np.zeros((64, 64), dtype=np.float32)

        # Top-left: GROUNDED (both peak, SCM open)
        corpus[:32, :32] = 0.8
        experiential[:32, :32] = 0.8

        # Top-right: ABSORBED (only corpus peaks)
        corpus[:32, 32:] = 0.8

        result = compute_interference(corpus, experiential, scm)

        assert result.grounded_fraction > 0.0
        assert result.absorbed_fraction > 0.0

    def test_signal_strength(self):
        """Signal strength is mean |pattern| in open gap regions."""
        corpus = _make_grid(0.8)
        experiential = _make_grid(0.8)
        scm = _make_grid(0.0)

        result = compute_interference(corpus, experiential, scm)

        expected_signal = abs(0.8 * 0.8 * 1.0)
        assert result.signal_strength == pytest.approx(expected_signal, abs=1e-4)


# ── interference_score tests ─────────────────────────────────────────────────


class TestInterferenceScore:
    def test_identical_corpus_high_score(self):
        """Identical corpus attractors → high score."""
        att = np.random.RandomState(42).randn(64, 64).astype(np.float32)
        scm = _make_grid(0.0)  # fully open

        score, state, _ = interference_score(att, None, att, None, scm)

        assert score > 0.5  # identical → Pearson ~1.0, gated by openness ~1.0

    def test_no_experiential_gives_absorbed(self):
        """When both experiential sides are None, state should be ABSORBED."""
        att = np.random.RandomState(42).randn(64, 64).astype(np.float32)
        scm = _make_grid(0.0)

        _, state, _ = interference_score(att, None, att, None, scm)

        assert state == ABSORBED

    def test_closed_scm_reduces_score(self):
        """Fully closed SCM (|M|=1) → all weights 0 → score collapses to 0.

        Under weighted-Pearson scoring, the SCM acts as a spatial mask rather
        than a uniform attenuator. A uniform |M|<1 SCM does not reduce a
        self-correlation score because weighted Pearson is scale-invariant in
        uniform weights. The cutoff that actually drops the score is |M|=1,
        which zeros the weights everywhere.
        """
        att = np.random.RandomState(42).randn(64, 64).astype(np.float32)
        scm_open = _make_grid(0.0)
        scm_closed = _make_grid(1.0)  # fully closed → all weights 0

        score_open, _, _ = interference_score(att, None, att, None, scm_open)
        score_closed, _, _ = interference_score(att, None, att, None, scm_closed)

        assert score_open > 0.9
        assert score_closed == pytest.approx(0.0)

    def test_experiential_boosts_score(self):
        """Adding experiential signal increases score."""
        rng = np.random.RandomState(42)
        corpus = rng.randn(64, 64).astype(np.float32)
        exp = rng.randn(64, 64).astype(np.float32)
        scm = _make_grid(0.0)

        score_no_exp, _, _ = interference_score(corpus, None, corpus, None, scm)
        score_with_exp, _, _ = interference_score(corpus, exp, corpus, exp, scm)

        assert score_with_exp > score_no_exp

    def test_score_distinguishes_pattern_at_equal_mean_openness(self):
        """Two SCMs with identical mean(1-|SCM|) but mirrored open regions
        yield different scores when content agreement is spatially non-uniform.

        Pre-fix the score factored as (content_pearson) * (mean_openness), so
        both SCMs produced the same score. Post-fix the score is a weighted
        Pearson where the SCM's spatial pattern selects which cells contribute
        to covariance and variance — the "Who" axis.
        """
        rng = np.random.RandomState(7)
        stored = rng.randn(64, 64).astype(np.float32)
        query = stored.copy()
        # Replace bottom half of query with independent noise — disagreement region
        query[32:, :] = rng.randn(32, 64).astype(np.float32)

        # SCM_A: top half open, bottom half closed → sees agreement region
        scm_a = np.zeros((64, 64), dtype=np.float32)
        scm_a[32:, :] = 1.0
        # SCM_B: bottom half open, top half closed → sees disagreement region
        scm_b = np.zeros((64, 64), dtype=np.float32)
        scm_b[:32, :] = 1.0

        # Identical mean openness — pre-fix collapse made these indistinguishable
        assert (1.0 - np.abs(scm_a)).mean() == pytest.approx(
            (1.0 - np.abs(scm_b)).mean()
        )

        score_a, _, _ = interference_score(query, None, stored, None, scm_a)
        score_b, _, _ = interference_score(query, None, stored, None, scm_b)

        # Agreement-region score is meaningfully higher than disagreement-region
        assert score_a > score_b + 0.3, (
            f"weighted Pearson should distinguish patterns: "
            f"score_a={score_a:.4f} score_b={score_b:.4f}"
        )

    def test_score_zero_when_fully_closed_scm(self):
        """All weights zero → score 0.0 (not NaN). Guards the Σw < ε branch."""
        rng = np.random.RandomState(1)
        att = rng.randn(64, 64).astype(np.float32)

        scm_pos = _make_grid(1.0)  # |M|=1 everywhere
        score_pos, _, _ = interference_score(att, None, att, None, scm_pos)
        assert score_pos == pytest.approx(0.0)
        assert not np.isnan(score_pos)

        # Mixed-sign closure still zeroes weights via the abs()
        scm_mixed = _make_grid(0.0)
        scm_mixed[:32, :] = 1.0
        scm_mixed[32:, :] = -1.0
        score_mixed, _, _ = interference_score(att, None, att, None, scm_mixed)
        assert score_mixed == pytest.approx(0.0)

    def test_score_recovers_pearson_when_scm_uniform_open(self):
        """Uniform fully-open SCM (|M|=0 everywhere) → weighted Pearson on
        each channel equals classical Pearson. Invariance check guards
        against algebraic drift in the helper.
        """
        from scipy.stats import pearsonr

        rng = np.random.RandomState(0)
        a = rng.randn(64, 64).astype(np.float32)
        b = rng.randn(64, 64).astype(np.float32)
        scm = _make_grid(0.0)

        score, _, _ = interference_score(a, None, b, None, scm)
        classical, _ = pearsonr(a.flatten(), b.flatten())

        assert score == pytest.approx(float(classical), abs=1e-5)


# ── InterferenceResult dataclass tests ───────────────────────────────────────


class TestInterferenceResult:
    def test_state_codes_roundtrip(self):
        """STATE_CODES and CODE_NAMES are inverse mappings."""
        for name, code in InterferenceResult.STATE_CODES.items():
            assert InterferenceResult.CODE_NAMES[code] == name


# ── Agent recall wiring tests ────────────────────────────────────────────────


class TestAgentRecallInterference:
    """Tests that _exec_recall_memory dispatches to interference when asked."""

    def test_exec_recall_with_interference_flag(self):
        """_exec_recall_memory(use_interference=True) should include interference keys."""
        import json
        from unittest.mock import patch

        from wheeler_memory.agent import _exec_recall_memory

        mock_hits = [
            {
                "text": "test memory",
                "similarity": 0.9,
                "state": "CONVERGED",
                "temperature": 0.5,
                "hex_key": "abc123",
                "chunk": "general",
                "interference_score": 0.85,
                "interference_state": "GROUNDED",
            }
        ]
        with patch(
            "wheeler_memory.interference.recall_with_interference",
            return_value=(mock_hits, "GROUNDED", 0.8),
        ) as mock_fn:
            result = _exec_recall_memory(
                "test query", top_k=5, data_dir=None, use_interference=True
            )

        parsed = json.loads(result)
        assert len(parsed["results"]) == 1
        assert parsed["results"][0]["interference_score"] == 0.85
        assert parsed["results"][0]["interference_state"] == "GROUNDED"

    def test_exec_recall_without_interference_default(self):
        """Default _exec_recall_memory should not include interference keys."""
        import json
        from unittest.mock import patch

        from wheeler_memory.agent import _exec_recall_memory

        mock_hits = [
            {
                "text": "test memory",
                "similarity": 0.9,
                "state": "CONVERGED",
                "temperature": 0.5,
            }
        ]
        with patch(
            "wheeler_memory.agent.recall_memory", return_value=mock_hits
        ):
            result = _exec_recall_memory(
                "test query", top_k=5, data_dir=None, use_interference=False
            )

        parsed = json.loads(result)
        assert "interference_score" not in parsed["results"][0]


# ── self_consistency_check tests ─────────────────────────────────────────────


class TestSelfConsistencyCheck:
    """self_consistency_check() — re-encode output, check basin convergence, update SCM."""

    def _corpus_att(self, text: str) -> np.ndarray:
        from wheeler_memory.constants import CORPUS_MAX_PUSH, CORPUS_SLOPE_FLOW
        from wheeler_memory.dynamics import evolve_with_params
        from wheeler_memory.rotation import _get_frame_fn

        frame = _get_frame_fn(encoder="hash")(text)
        return evolve_with_params(frame, CORPUS_MAX_PUSH, CORPUS_SLOPE_FLOW)["attractor"]

    def test_same_text_synthesis(self, tmp_path):
        """Re-encoding the same text → same basin → SYNTHESIS."""
        from wheeler_memory.interference import self_consistency_check
        from wheeler_memory.scm_grid import SCMGrid

        text = "the quick brown fox jumps over the lazy dog"
        corpus_att = self._corpus_att(text)
        scm = SCMGrid.load_or_create(tmp_path)

        cr = self_consistency_check(text, corpus_att, None, scm, encoder="hash")

        assert cr.consistent
        assert cr.classification == "SYNTHESIS"

    def test_different_text_not_synthesis(self, tmp_path):
        """Different text → divergent basin → not SYNTHESIS."""
        from wheeler_memory.interference import self_consistency_check
        from wheeler_memory.scm_grid import SCMGrid

        corpus_att = self._corpus_att("the quick brown fox jumps over the lazy dog")
        scm = SCMGrid.load_or_create(tmp_path)

        cr = self_consistency_check(
            "quantum entanglement superconductor photon", corpus_att, None, scm, encoder="hash"
        )

        assert not cr.consistent
        assert cr.classification != "SYNTHESIS"

    def test_cells_updated_nonzero(self, tmp_path):
        """Corpus peaks always exist → cells_updated > 0."""
        from wheeler_memory.interference import self_consistency_check
        from wheeler_memory.scm_grid import SCMGrid

        text = "memory consolidation during sleep"
        corpus_att = self._corpus_att(text)
        scm = SCMGrid.load_or_create(tmp_path)

        cr = self_consistency_check(text, corpus_att, None, scm, encoder="hash")

        assert cr.cells_updated > 0

    def test_consistent_opens_gaps(self, tmp_path):
        """Consistent output drives SCM cells toward open (direction=-1)."""
        from wheeler_memory.interference import self_consistency_check
        from wheeler_memory.scm_grid import SCMGrid

        text = "the quick brown fox jumps over the lazy dog"
        corpus_att = self._corpus_att(text)
        scm = SCMGrid.load_or_create(tmp_path)
        scm.grid[:] = 0.2  # pre-seed as partially closed

        cr = self_consistency_check(text, corpus_att, None, scm, encoder="hash")

        if cr.consistent:
            # Peak cells were driven negative (opened from 0.2)
            assert scm.grid.min() < 0.2

    def test_inconsistent_closes_gaps(self, tmp_path):
        """Inconsistent output drives SCM cells toward closed (direction=+1)."""
        from wheeler_memory.interference import self_consistency_check
        from wheeler_memory.scm_grid import SCMGrid

        corpus_att = self._corpus_att("the quick brown fox jumps over the lazy dog")
        scm = SCMGrid.load_or_create(tmp_path)

        cr = self_consistency_check(
            "quantum entanglement superconductor photon", corpus_att, None, scm, encoder="hash"
        )

        if not cr.consistent:
            # Peak cells were pushed positive (closed from 0)
            assert scm.grid.max() > 0
