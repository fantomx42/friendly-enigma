"""Diagnostic: compare hippocampus vs hippo-word(0.2) on per-question ternary scoring.

For each MMLU question, runs both encoders on all 4 choices and collects:
- Ternary grid stats (local max/min/slope counts)
- Net ternary score per choice per encoder
- Which choice each encoder picks
- Whether they agree, whether each got it right
- Question-level features (word count, vocab diversity, avg word length)

Usage:
    python scripts/diagnose_encoder_divergence.py
    python scripts/diagnose_encoder_divergence.py --subjects sociology virology --samples 50
    python scripts/diagnose_encoder_divergence.py --output my_diagnostic.tsv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np


def _question_features(question: str) -> dict:
    """Compute cheap text features for a question."""
    words = re.split(r'[^a-z0-9]+', question.lower())
    words = [w for w in words if w]
    n = len(words)
    if n == 0:
        return {"word_count": 0, "unique_ratio": 0.0, "avg_word_len": 0.0}
    unique = len(set(words))
    avg_len = sum(len(w) for w in words) / n
    return {
        "word_count": n,
        "unique_ratio": round(unique / n, 4),
        "avg_word_len": round(avg_len, 2),
    }


def run_diagnostic(subjects: list[str], samples: int | None, output_path: Path, blend: float):
    from wheeler_memory.hippocampus import hippocampus_to_frame
    from wheeler_memory.word_encoder import word_to_frame
    from wheeler_memory.dynamics import evolve_and_interpret, snap_to_ternary

    # Import MMLU loader from the benchmark script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from wheeler_mmlu import load_mmlu

    def encode_and_score(question: str, choices: list[str], encoder_fn):
        """Encode all 4 choices, evolve, snap to ternary, return scores + stats."""
        results = []
        for c in choices:
            frame = encoder_fn(f"{question} {c}")
            evo = evolve_and_interpret(frame)
            ternary = snap_to_ternary(evo["attractor"])
            n_pos = int(np.sum(ternary == 1))
            n_neg = int(np.sum(ternary == -1))
            n_zero = int(np.sum(ternary == 0))
            net = n_pos - n_neg
            results.append({
                "net": net, "n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero,
                "ticks": evo["convergence_ticks"],
            })
        scores = [r["net"] for r in results]
        pick = int(max(range(4), key=lambda i: scores[i]))
        return pick, results

    def blend_fn(text: str, size: int = 64) -> np.ndarray:
        h = hippocampus_to_frame(text, size)
        w = word_to_frame(text, size)
        return np.tanh((1 - blend) * h + blend * w).astype(np.float32)

    rows = []
    subject_stats: dict[str, dict] = {}

    print(f"\n{'='*70}")
    print(f"  ENCODER DIVERGENCE DIAGNOSTIC")
    print(f"  Subjects: {subjects}")
    print(f"  Samples: {samples or 'all'}")
    print(f"  Blend α: {blend} (hippo-word)")
    print(f"{'='*70}\n")

    for subj in subjects:
        items = list(load_mmlu([subj], split="test", samples=samples))
        stats = {"hippo_wins": 0, "blend_wins": 0, "both_right": 0,
                 "both_wrong": 0, "agree": 0, "total": 0,
                 "hippo_win_features": [], "blend_win_features": []}

        print(f"  [{subj}] {len(items)} questions")
        t0 = time.time()

        for qi, (_, question, choices, answer_idx) in enumerate(items):
            hippo_pick, hippo_results = encode_and_score(question, choices, hippocampus_to_frame)
            blend_pick, blend_results = encode_and_score(question, choices, blend_fn)

            hippo_correct = hippo_pick == answer_idx
            blend_correct = blend_pick == answer_idx
            agree = hippo_pick == blend_pick

            feats = _question_features(question)

            # Classify outcome
            if hippo_correct and blend_correct:
                outcome = "both_right"
                stats["both_right"] += 1
            elif hippo_correct and not blend_correct:
                outcome = "hippo_wins"
                stats["hippo_wins"] += 1
                stats["hippo_win_features"].append(feats)
            elif blend_correct and not hippo_correct:
                outcome = "blend_wins"
                stats["blend_wins"] += 1
                stats["blend_win_features"].append(feats)
            else:
                outcome = "both_wrong"
                stats["both_wrong"] += 1

            if agree:
                stats["agree"] += 1
            stats["total"] += 1

            # Row for TSV
            row = {
                "subject": subj,
                "qid": qi,
                "answer_idx": answer_idx,
                "hippo_pick": hippo_pick,
                "blend_pick": blend_pick,
                "hippo_correct": int(hippo_correct),
                "blend_correct": int(blend_correct),
                "agree": int(agree),
                "outcome": outcome,
                # Hippo stats for correct answer choice
                "hippo_net_correct": hippo_results[answer_idx]["net"],
                "blend_net_correct": blend_results[answer_idx]["net"],
                # Hippo stats for picked choice
                "hippo_net_picked": hippo_results[hippo_pick]["net"],
                "blend_net_picked": blend_results[blend_pick]["net"],
                # Ternary topology for correct answer (hippo)
                "hippo_pos": hippo_results[answer_idx]["n_pos"],
                "hippo_neg": hippo_results[answer_idx]["n_neg"],
                "hippo_slope": hippo_results[answer_idx]["n_zero"],
                # Ternary topology for correct answer (blend)
                "blend_pos": blend_results[answer_idx]["n_pos"],
                "blend_neg": blend_results[answer_idx]["n_neg"],
                "blend_slope": blend_results[answer_idx]["n_zero"],
                # Question features
                "word_count": feats["word_count"],
                "unique_ratio": feats["unique_ratio"],
                "avg_word_len": feats["avg_word_len"],
            }
            rows.append(row)

        elapsed = time.time() - t0
        subject_stats[subj] = stats
        n = stats["total"]
        print(f"    {elapsed:.1f}s | agree={stats['agree']}/{n} ({100*stats['agree']/n:.0f}%)")
        print(f"    hippo_wins={stats['hippo_wins']} blend_wins={stats['blend_wins']} "
              f"both_right={stats['both_right']} both_wrong={stats['both_wrong']}")

    # Write TSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"\n  Saved {len(rows)} rows to {output_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    for subj, stats in subject_stats.items():
        n = stats["total"]
        print(f"\n  {subj} ({n} questions):")
        print(f"    Agreement rate:  {stats['agree']}/{n} ({100*stats['agree']/n:.0f}%)")
        print(f"    Both right:      {stats['both_right']} ({100*stats['both_right']/n:.1f}%)")
        print(f"    Both wrong:      {stats['both_wrong']} ({100*stats['both_wrong']/n:.1f}%)")
        print(f"    Hippo only:      {stats['hippo_wins']} ({100*stats['hippo_wins']/n:.1f}%)")
        print(f"    Blend only:      {stats['blend_wins']} ({100*stats['blend_wins']/n:.1f}%)")

        # Feature comparison
        def mean_feats(feat_list):
            if not feat_list:
                return {"word_count": 0, "unique_ratio": 0, "avg_word_len": 0}
            return {
                k: round(sum(f[k] for f in feat_list) / len(feat_list), 3)
                for k in feat_list[0]
            }

        if stats["hippo_win_features"] and stats["blend_win_features"]:
            hf = mean_feats(stats["hippo_win_features"])
            bf = mean_feats(stats["blend_win_features"])
            print(f"\n    Question features where each encoder wins exclusively:")
            print(f"    {'Feature':<18} {'Hippo wins':>12} {'Blend wins':>12} {'Delta':>8}")
            print(f"    {'-'*50}")
            for k in hf:
                delta = bf[k] - hf[k]
                print(f"    {k:<18} {hf[k]:>12.3f} {bf[k]:>12.3f} {delta:>+8.3f}")

    # Oracle score (best of both per question)
    oracle_correct = sum(1 for r in rows if r["hippo_correct"] or r["blend_correct"])
    total = len(rows)
    hippo_total = sum(r["hippo_correct"] for r in rows)
    blend_total = sum(r["blend_correct"] for r in rows)
    print(f"\n  Overall:")
    print(f"    Hippocampus:    {hippo_total}/{total} ({100*hippo_total/total:.1f}%)")
    print(f"    Hippo-word:     {blend_total}/{total} ({100*blend_total/total:.1f}%)")
    print(f"    Oracle (best):  {oracle_correct}/{total} ({100*oracle_correct/total:.1f}%)")
    print(f"    Oracle ceiling: +{oracle_correct - max(hippo_total, blend_total)} over best single encoder")


def main():
    parser = argparse.ArgumentParser(description="Diagnose encoder divergence on MMLU ternary")
    parser.add_argument("--subjects", nargs="+", default=["sociology", "virology"])
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--output", default="encoder_divergence.tsv")
    parser.add_argument("--blend", type=float, default=0.2,
                        help="Word blend alpha for hippo-word (default: 0.2)")
    args = parser.parse_args()

    output_path = Path(args.output)
    run_diagnostic(args.subjects, args.samples, output_path, args.blend)


if __name__ == "__main__":
    main()
