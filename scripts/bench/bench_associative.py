"""Wheeler Memory — Associative Memory Characterization Benchmark

Tests three standard metrics used in the associative memory literature:

  1. Capacity curve       — recall accuracy vs. N stored patterns
  2. Cue degradation      — recall accuracy vs. % of query corrupted/masked
  3. Interference test    — recall accuracy vs. semantic similarity between stored patterns

Compares results against published baselines:
  - Classical Hopfield:  capacity ratio α ≈ 0.138, ~30% noise tolerance
  - Modern Hopfield:     exp(αN) capacity, >90% at 40% masking
  - Sparse Distributed Memory: graceful degradation, high-dim capacity

Usage
-----
    python scripts/bench_associative.py
    python scripts/bench_associative.py --n-trials 50 --max-n 500
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────

def _embed(texts: list[str]):
    from wheeler_memory.embedding import embed_to_frame_batch
    return embed_to_frame_batch(texts)

def _evolve(frame):
    from wheeler_memory.dynamics import evolve_and_interpret
    return evolve_and_interpret(frame)

def _store(text: str, frame, data_dir: Path):
    from wheeler_memory.dynamics import evolve_and_interpret
    from wheeler_memory.brick import MemoryBrick
    from wheeler_memory.storage import store_memory
    result = evolve_and_interpret(frame)
    brick = MemoryBrick.from_evolution_result(result)
    store_memory(text, result, brick, data_dir=data_dir, chunk="science", auto_evict=False)
    return result["attractor"].flatten().astype(np.float32)

def _build_cache(data_dir: Path):
    from wheeler_memory.storage import list_memories
    memories = list_memories(data_dir=data_dir)
    texts, atts = [], []
    for m in memories:
        p = data_dir / "chunks" / m.get("chunk", "general") / "attractors" / f"{m['hex_key']}.npy"
        if not p.exists():
            continue
        att = np.load(p)
        if att.shape != (64, 64):
            continue
        atts.append(att.flatten().astype(np.float32))
        texts.append(m["text"])
    if not atts:
        return None
    matrix = np.stack(atts)
    return {"matrix": matrix, "means": matrix.mean(axis=1), "stds": matrix.std(axis=1), "texts": texts}

def _search(cache, frame, top_k: int = 1) -> list[dict]:
    from wheeler_memory.dynamics import evolve_and_interpret
    from wheeler_memory.constants import SALIENCE_MAX_ITERS_MED, SALIENCE_THRESHOLD_MED
    result = evolve_and_interpret(frame, max_iters=SALIENCE_MAX_ITERS_MED,
                                  stability_threshold=SALIENCE_THRESHOLD_MED)
    q = result["attractor"].flatten().astype(np.float32)
    qm, qs = q.mean(), q.std()
    if qs < 1e-10:
        return []
    centered = cache["matrix"] - cache["means"][:, None]
    dots = centered @ (q - qm)
    valid = cache["stds"] > 1e-10
    sims = np.where(valid, dots / (4096 * cache["stds"] * qs), 0.0)
    k = min(top_k, len(sims))
    idx = np.argpartition(sims, -k)[-k:]
    idx = idx[np.argsort(sims[idx])[::-1]]
    return [{"text": cache["texts"][i], "similarity": float(sims[i])} for i in idx]

def _corrupt_text(text: str, corruption: float, rng: random.Random) -> str:
    """Randomly drop words from text to simulate cue degradation."""
    words = text.split()
    keep = [w for w in words if rng.random() > corruption]
    return " ".join(keep) if keep else words[0]

def _load_facts(n: int) -> list[dict]:
    """Load physics facts from MMLU dev+validation for use as test patterns."""
    from datasets import load_dataset
    CHOICES = ["A", "B", "C", "D"]
    facts = []
    for subject in ["high_school_physics", "conceptual_physics", "college_physics"]:
        for split in ["dev", "validation"]:
            try:
                ds = load_dataset("cais/mmlu", subject, split=split)
                for item in ds:
                    letter = CHOICES[int(item["answer"])]
                    choice_text = item["choices"][int(item["answer"])]
                    facts.append({
                        "text": f"Q: {item['question']} A: {letter}. {choice_text}",
                        "question": item["question"],
                        "letter": letter,
                    })
            except Exception:
                continue
        if len(facts) >= n:
            break
    return facts[:n]

# ── Test 1: Capacity curve ────────────────────────────────────────────────────

def test_capacity_curve(facts: list[dict], ns: list[int], n_trials: int = 3) -> dict:
    """Recall accuracy vs. N stored patterns.

    For each N, store N facts in a fresh temp store, then query each with
    its original question (no choices). A hit = correct letter retrieved.
    Repeated n_trials times with random subsets.
    """
    print("\n── Test 1: Capacity Curve ──────────────────────────────────────")
    print(f"  N values: {ns}")
    print(f"  Trials per N: {n_trials}")
    print(f"  Grid: 64×64 = 4096 cells  |  Classical Hopfield limit: ~565 patterns")
    print()
    print(f"  {'N':>6}  {'α=N/4096':>10}  {'Accuracy':>10}  {'vs Classical':>14}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*14}")

    results = {}
    rng = random.Random(42)

    for n in ns:
        if n > len(facts):
            print(f"  {n:>6}  (insufficient facts, need {n})")
            continue

        trial_accs = []
        for trial in range(n_trials):
            subset = rng.sample(facts, n)
            with tempfile.TemporaryDirectory() as tmp:
                data_dir = Path(tmp)
                texts = [f["text"] for f in subset]
                frames = _embed(texts)
                for text, frame in zip(texts, frames):
                    _store(text, frame, data_dir)
                cache = _build_cache(data_dir)
                if cache is None:
                    continue

                hits = 0
                q_frames = _embed([f["question"] for f in subset])
                for fact, qframe in zip(subset, q_frames):
                    top = _search(cache, qframe, top_k=min(5, n))
                    for hit in top:
                        m = re.search(r"A:\s*([A-D])\.", hit["text"])
                        if m and m.group(1) == fact["letter"]:
                            hits += 1
                            break
                trial_accs.append(hits / n)

        if not trial_accs:
            continue
        acc = np.mean(trial_accs)
        alpha = n / 4096
        # Classical Hopfield degrades sharply above α=0.138
        classical = "OVER LIMIT" if alpha > 0.138 else f"below limit ({0.138:.3f})"
        print(f"  {n:>6}  {alpha:>10.4f}  {acc:>9.1%}  {classical:>14}")
        results[n] = {"accuracy": float(acc), "alpha": alpha, "trials": trial_accs}

    return results

# ── Test 2: Cue degradation ───────────────────────────────────────────────────

def test_cue_degradation(facts: list[dict], n_store: int = 22,
                          corruptions: list[float] = None) -> dict:
    """Recall accuracy vs. fraction of query text removed.

    Classical baseline: ~30% noise tolerance
    Modern Hopfield: >90% at 40% masking
    """
    if corruptions is None:
        corruptions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    print("\n── Test 2: Cue Degradation ─────────────────────────────────────")
    print(f"  Store size: {n_store} facts")
    print(f"  Corruption = fraction of words randomly removed from query")
    print(f"  Classical Hopfield: ~70% accuracy at 30% corruption")
    print(f"  Modern Hopfield:    >90% accuracy at 40% masking")
    print()
    print(f"  {'Corruption':>12}  {'Accuracy':>10}  {'vs Modern Hopfield':>20}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*20}")

    subset = facts[:n_store]
    results = {}
    rng = random.Random(42)

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        texts = [f["text"] for f in subset]
        frames = _embed(texts)
        for text, frame in zip(texts, frames):
            _store(text, frame, data_dir)
        cache = _build_cache(data_dir)

        for corruption in corruptions:
            hits = 0
            queries = [_corrupt_text(f["question"], corruption, rng) for f in subset]
            q_frames = _embed(queries)
            for fact, qframe in zip(subset, q_frames):
                top = _search(cache, qframe, top_k=min(5, n_store))
                for hit in top:
                    m = re.search(r"A:\s*([A-D])\.", hit["text"])
                    if m and m.group(1) == fact["letter"]:
                        hits += 1
                        break
            acc = hits / n_store
            modern_hopfield_target = 0.90 if corruption <= 0.40 else 0.70
            delta = acc - modern_hopfield_target
            comparison = f"{delta:+.1%} vs MH"
            print(f"  {corruption:>11.0%}  {acc:>9.1%}  {comparison:>20}")
            results[corruption] = float(acc)

    return results

# ── Test 3: Interference ──────────────────────────────────────────────────────

def test_interference(facts: list[dict], n_store: int = 20) -> dict:
    """Recall accuracy when stored patterns have varying semantic similarity.

    Groups: same-subject (high interference) vs. cross-subject (low interference).
    Classical prediction: similar patterns interfere more → lower recall.
    """
    print("\n── Test 3: Interference ────────────────────────────────────────")
    print(f"  Store size per group: {n_store}")
    print(f"  Prediction: cross-subject facts should have higher recall than same-subject")
    print()

    from datasets import load_dataset
    CHOICES = ["A", "B", "C", "D"]

    def load_subject(subject, n):
        out = []
        for split in ["dev", "validation"]:
            try:
                ds = load_dataset("cais/mmlu", subject, split=split)
                for item in ds:
                    letter = CHOICES[int(item["answer"])]
                    out.append({
                        "text": f"Q: {item['question']} A: {letter}. {item['choices'][int(item['answer'])]}",
                        "question": item["question"],
                        "letter": letter,
                    })
                    if len(out) >= n:
                        return out
            except Exception:
                pass
        return out

    physics = load_subject("high_school_physics", n_store)
    history = load_subject("high_school_world_history", n_store)
    n = min(len(physics), len(history), n_store)
    physics, history = physics[:n], history[:n]

    results = {}
    for label, store_facts, query_facts in [
        ("Same-subject (physics only)", physics, physics),
        ("Cross-subject (physics+history)", physics + history, physics),
    ]:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            texts = [f["text"] for f in store_facts]
            frames = _embed(texts)
            for text, frame in zip(texts, frames):
                _store(text, frame, data_dir)
            cache = _build_cache(data_dir)

            hits = 0
            q_frames = _embed([f["question"] for f in query_facts])
            for fact, qframe in zip(query_facts, q_frames):
                top = _search(cache, qframe, top_k=min(5, len(store_facts)))
                for hit in top:
                    m = re.search(r"A:\s*([A-D])\.", hit["text"])
                    if m and m.group(1) == fact["letter"]:
                        hits += 1
                        break
            acc = hits / len(query_facts)
            print(f"  {label}: {acc:.1%}  ({hits}/{len(query_facts)})")
            results[label] = float(acc)

    interference = results.get("Same-subject (physics only)", 0) - \
                   results.get("Cross-subject (physics+history)", 0)
    print(f"\n  Interference effect: {interference:+.1%}  (negative = same-subject hurts recall)")
    return results

# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Wheeler associative memory characterization")
    p.add_argument("--max-n", type=int, default=400,
                   help="Max N for capacity curve (default 400)")
    p.add_argument("--n-trials", type=int, default=3,
                   help="Trials per N in capacity curve (default 3)")
    p.add_argument("--skip-capacity", action="store_true")
    p.add_argument("--skip-degradation", action="store_true")
    p.add_argument("--skip-interference", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()

    print("=" * 66)
    print("  WHEELER MEMORY — ASSOCIATIVE MEMORY CHARACTERIZATION")
    print("  Grid: 64×64  |  N_cells: 4096  |  Classical limit α=0.138 → ~565 patterns")
    print("=" * 66)

    # Load facts once
    max_needed = max(args.max_n, 60)
    print(f"\nLoading up to {max_needed} physics Q&A facts...")
    facts = _load_facts(max_needed)
    print(f"  Loaded {len(facts)} facts\n")

    if not facts:
        print("ERROR: Could not load any facts. Is 'datasets' installed?")
        sys.exit(1)

    # Warm up embedding model
    print("Warming up embedding model...")
    _embed([facts[0]["text"]])
    print("  Ready.\n")

    t0 = time.time()

    # Test 1: Capacity curve
    if not args.skip_capacity:
        ns = [5, 10, 22, 50, 100, 200]
        ns = [n for n in ns if n <= min(args.max_n, len(facts))]
        cap_results = test_capacity_curve(facts, ns=ns, n_trials=args.n_trials)

    # Test 2: Cue degradation
    if not args.skip_degradation:
        deg_results = test_cue_degradation(facts, n_store=min(22, len(facts)))

    # Test 3: Interference
    if not args.skip_interference:
        int_results = test_interference(facts)

    elapsed = time.time() - t0
    print(f"\n{'='*66}")
    print(f"  Total elapsed: {elapsed:.1f}s")
    print(f"{'='*66}")

    print("""
Literature baselines:
  Capacity:    Classical Hopfield α=0.138 (~565 patterns @ 4096 cells)
               Modern Hopfield: exponential (orders of magnitude more)
  Degradation: Classical: ~70% at 30% corruption
               Modern Hopfield: >90% at 40% masking
  Interference: Classical: stronger interference from similar patterns
""")


if __name__ == "__main__":
    main()
