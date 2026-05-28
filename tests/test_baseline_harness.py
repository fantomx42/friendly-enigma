"""Smoke tests for scripts/bench/baseline_harness.py.

Covers what the harness itself controls: helpers, probe construction, and
unit-level shape of the per-arm scoring functions. The full main() path
(corpus storage + MiniLM + SCM warmup + JSONL writing) is exercised by an
end-to-end test gated behind the `slow` + `embed` markers, since it needs
arc.jsonl on disk and sentence-transformers installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.bench.baseline_harness import (
    COS_BAND_HI,
    COS_BAND_LO,
    _ArmStats,
    _build_probes,
    _content_tokens,
    _find_rank,
    _paraphrase,
    _q_part,
    _shared_cue,
    main,
)


# ── helper-level tests (cheap, no model load) ────────────────────────────────


def test_q_part_strips_answer():
    assert _q_part("Q: what is X? A: thing.") == "Q: what is X?"
    assert _q_part("no A marker here") == "no A marker here"


def test_paraphrase_drops_stopwords_and_is_deterministic():
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    # Need >=3 content words to exercise the stopword filter; below that the
    # function falls back to the full word list (see _paraphrase docstring).
    src = "the quick brown fox jumps over the lazy dog"
    out1 = _paraphrase(src, rng1)
    out2 = _paraphrase(src, rng2)
    assert out1 == out2, "same seed → identical paraphrase"
    assert "the" not in out1.split(), "stopword should be removed"


def test_content_tokens_normalizes_punctuation():
    toks = _content_tokens("Energy! transfers, between systems?")
    assert toks == ["energy", "transfers", "between", "systems"]


def test_shared_cue_finds_overlap():
    cue = _shared_cue("how does energy transfer", "what makes energy transfer happen")
    assert "energy" in cue.split() and "transfer" in cue.split()


def test_find_rank_returns_1_indexed():
    assert _find_rank(["a", "b", "c"], "b") == 2
    assert _find_rank(["a", "b", "c"], "missing") is None


def test_arm_stats_recall_and_mrr():
    s = _ArmStats("x")
    s.update(1)
    s.update(2)
    s.update(None)
    s.update(5)
    out = s.summary()
    assert out["n"] == 4
    assert out["found"] == 3
    assert out["recall@1"] == 0.25
    assert out["recall@3"] == 0.5
    # MRR = (1/1 + 1/2 + 1/5) / 4 = (1 + 0.5 + 0.2) / 4 = 0.425
    assert abs(out["mrr"] - 0.425) < 1e-6


def test_build_probes_respects_cosine_band():
    """Pairs outside [COS_BAND_LO, COS_BAND_HI] must not yield probes."""
    rng = np.random.default_rng(42)
    passages = [
        "Q: how does energy transfer in food webs? A: producers to consumers.",
        "Q: how does energy transfer in circuits? A: voltage drives current.",
        "Q: what is mitosis? A: cell division producing identical copies.",
    ]
    keys = [f"key_{i}" for i in range(3)]
    # Hand-craft cosines: (0,1) in-band, (0,2) and (1,2) out-of-band low.
    cos = np.array(
        [
            [-1.0, 0.40, 0.05],
            [0.40, -1.0, 0.05],
            [0.05, 0.05, -1.0],
        ],
        dtype=np.float32,
    )
    probes = _build_probes(passages, keys, cos, rng)
    # Should produce exactly 2 probes from the (0, 1) pair (one per context).
    assert len(probes) == 2
    pairs = {tuple(p["pair"]) for p in probes}
    assert pairs == {(0, 1)}
    labels = sorted(p["context_label"] for p in probes)
    assert labels == ["A", "B"]
    # Correct keys must flip with the context.
    by_label = {p["context_label"]: p["correct_key"] for p in probes}
    assert by_label["A"] == "key_0"
    assert by_label["B"] == "key_1"


def test_build_probes_drops_pairs_outside_band():
    rng = np.random.default_rng(0)
    passages = ["Q: A? A: 1.", "Q: B? A: 2."]
    keys = ["k0", "k1"]
    cos = np.array(
        [[-1.0, COS_BAND_HI + 0.1], [COS_BAND_HI + 0.1, -1.0]], dtype=np.float32
    )
    assert _build_probes(passages, keys, cos, rng) == []
    cos = np.array(
        [[-1.0, COS_BAND_LO - 0.1], [COS_BAND_LO - 0.1, -1.0]], dtype=np.float32
    )
    assert _build_probes(passages, keys, cos, rng) == []


# ── main() exit code tests (no model load required) ─────────────────────────


def test_main_returns_2_when_datasets_dir_missing(tmp_path, capsys, monkeypatch):
    """If arc.jsonl is missing, the script must exit non-zero with a clear message."""
    monkeypatch.setattr(
        "sys.argv",
        ["wheeler-baseline", "--datasets-dir", str(tmp_path), "--n", "5"],
    )
    rc = main()
    captured = capsys.readouterr()
    assert rc == 2
    assert "arc.jsonl not found" in captured.err


# ── end-to-end (slow + embed; gated behind markers and presence of arc.jsonl) ─


@pytest.mark.slow
@pytest.mark.embed
def test_main_end_to_end_small(tmp_path, monkeypatch):
    """Full harness on a tiny corpus. Skipped if arc.jsonl is unavailable."""
    pytest.importorskip("sentence_transformers")

    # Look for arc.jsonl in the common locations (mirrors the script's default).
    candidates = [
        Path(__file__).parent.parent.parent / "datasets",
        Path(__file__).parent.parent / "datasets",
    ]
    datasets_dir = next((p for p in candidates if (p / "arc.jsonl").exists()), None)
    if datasets_dir is None:
        pytest.skip("arc.jsonl not available in any expected location")

    # Redirect results/ output into the test tmp_path so we don't pollute the repo.
    import scripts.bench.baseline_harness as bh

    monkeypatch.setattr(bh, "REPO_ROOT", tmp_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "wheeler-baseline",
            "--n",
            "8",
            "--seed",
            "1",
            "--datasets-dir",
            str(datasets_dir),
            "--no-warmup",
        ],
    )
    rc = bh.main()
    assert rc == 0
    # JSONL written; meta line parses and has the expected top-level fields.
    out_files = sorted((tmp_path / "results").glob("baseline_harness_*.jsonl"))
    assert out_files, "expected at least one JSONL output file"
    meta = json.loads(out_files[-1].read_text().splitlines()[0])
    assert meta["type"] == "run_meta"
    assert "phase1" in meta and "phase2" in meta
    assert meta["phase1"]["phase"] == "topology_isolation"
