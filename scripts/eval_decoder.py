#!/usr/bin/env python3
"""Decoder evaluation — measure response quality vs attractor depth.

Tests the Wheeler-primary decoder across three attractor depth tiers:
  - DEEP: concept directly crystallized in corpus (high similarity expected)
  - SHALLOW: related concept not directly stored (moderate similarity)
  - MISSING: concept with no attractor at all (low/zero similarity)

The key empirical question: how steep is the quality cliff between tiers?

Usage:
    python scripts/eval_decoder.py                    # Wheeler state only
    python scripts/eval_decoder.py --decode           # Also run small model
    python scripts/eval_decoder.py --model qwen2.5:1.5b --decode
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from wheeler_memory.decoder import extract_state, format_state
from wheeler_memory.storage import recall_memory

# ── Test cases ────────────────────────────────────────────────────────────────

# These should be filled in AFTER crystallization — use concepts that
# are known to be in/near/absent from the corpus.

TEST_CASES = [
    # DEEP: directly in corpus (curated ML entries)
    {
        "query": "How does self-attention work in neural networks?",
        "expected_depth": "deep",
        "description": "Self-attention is directly in the curated ML corpus",
    },
    {
        "query": "What is backpropagation and how do neural networks learn?",
        "expected_depth": "deep",
        "description": "Backpropagation is directly in curated ML entries",
    },
    {
        "query": "Explain quantum superposition in physics",
        "expected_depth": "deep",
        "description": "Quantum superposition is directly in curated science entries",
    },

    # SHALLOW: related but not directly stored
    {
        "query": "What is the transformer architecture?",
        "expected_depth": "shallow",
        "description": "Transformer not directly stored but neighbors are (attention, encoder-decoder, etc.)",
    },
    {
        "query": "How does DNA replication work?",
        "expected_depth": "shallow",
        "description": "DNA replication not stored but DNA encoding is",
    },
    {
        "query": "What is the difference between supervised and unsupervised learning?",
        "expected_depth": "shallow",
        "description": "Not directly stored, but ML concepts nearby",
    },

    # MISSING: no attractor at all
    {
        "query": "What is the best recipe for sourdough bread?",
        "expected_depth": "missing",
        "description": "Cooking is not in any corpus domain",
    },
    {
        "query": "Who won the 2024 presidential election?",
        "expected_depth": "missing",
        "description": "Current events not in corpus",
    },
    {
        "query": "How do I fix a leaking faucet?",
        "expected_depth": "missing",
        "description": "Home repair not in corpus",
    },
]

# ── Evaluation logic ─────────────────────────────────────────────────────────


def evaluate_case(
    case: dict,
    data_dir: Path | None = None,
    recall_k: int = 5,
    confidence_floor: float = 0.3,
) -> dict:
    """Evaluate a single test case against Wheeler's memory."""
    query = case["query"]

    # Recall from Wheeler
    hits = recall_memory(
        query,
        top_k=recall_k,
        data_dir=data_dir,
        use_embedding=True,
    )

    # Extract state
    state = extract_state(query, hits, confidence_floor)

    # Compute metrics
    top_sim = hits[0]["similarity"] if hits else 0.0
    top_text = hits[0]["text"][:80] if hits else "(none)"
    avg_sim = sum(h["similarity"] for h in hits) / len(hits) if hits else 0.0

    return {
        "query": query,
        "expected_depth": case["expected_depth"],
        "description": case["description"],
        "confidence": state.confidence,
        "uncertain": state.uncertain,
        "top_similarity": top_sim,
        "avg_similarity": avg_sim,
        "top_hit": top_text,
        "n_hits": len(hits),
        "co_activated": len(state.co_activated),
        "formatted_state": format_state(state),
    }


def run_evaluation(
    data_dir: Path | None = None,
    decode: bool = False,
    model: str = "qwen2.5:1.5b",
    ollama_url: str = "http://localhost:11434",
    confidence_floor: float = 0.3,
):
    """Run full evaluation across all test cases."""
    print("=" * 80)
    print("DECODER EVALUATION — Attractor Depth vs Response Quality")
    print("=" * 80)
    print()

    results = []
    for case in TEST_CASES:
        result = evaluate_case(case, data_dir=data_dir, confidence_floor=confidence_floor)

        # Optionally decode via small model
        if decode:
            try:
                from wheeler_memory.decoder import WheelerPrimaryAgent

                agent = WheelerPrimaryAgent(
                    model=model,
                    ollama_url=ollama_url,
                    data_dir=data_dir,
                    confidence_floor=confidence_floor,
                )
                response = agent.run(case["query"])
                result["response"] = response
                result["response_preview"] = response[:200]
            except Exception as e:
                result["response"] = f"(decode error: {e})"
                result["response_preview"] = result["response"]

        results.append(result)

    # Print results table
    print(f"{'Depth':<10} {'Confidence':>10} {'Top Sim':>8} {'Avg Sim':>8} {'Hits':>5} {'Query'}")
    print(f"{'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*5} {'─'*50}")
    for r in results:
        conf_str = f"{r['confidence']:.2f}" + ("*" if r["uncertain"] else " ")
        print(
            f"{r['expected_depth']:<10} {conf_str:>10} {r['top_similarity']:>8.3f} "
            f"{r['avg_similarity']:>8.3f} {r['n_hits']:>5} {r['query'][:50]}"
        )

    # Print detailed results per tier
    for tier in ("deep", "shallow", "missing"):
        tier_results = [r for r in results if r["expected_depth"] == tier]
        if not tier_results:
            continue

        print(f"\n{'─' * 80}")
        print(f"  TIER: {tier.upper()}")
        print(f"{'─' * 80}")

        for r in tier_results:
            print(f"\n  Query: {r['query']}")
            print(f"  Top hit: {r['top_hit']}")
            print(f"  Confidence: {r['confidence']:.3f} ({'uncertain' if r['uncertain'] else 'confident'})")
            print(f"  Top sim: {r['top_similarity']:.3f}, Avg sim: {r['avg_similarity']:.3f}")
            if decode and "response_preview" in r:
                print(f"  Response: {r['response_preview']}")

    # Save log
    log_path = Path(data_dir or Path.home() / ".wheeler_memory") / "eval_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        for r in results:
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "query": r["query"],
                "expected_depth": r["expected_depth"],
                "confidence": r["confidence"],
                "top_similarity": r["top_similarity"],
                "avg_similarity": r["avg_similarity"],
                "uncertain": r["uncertain"],
                "top_hit": r["top_hit"],
            }
            if decode and "response_preview" in r:
                entry["response_preview"] = r["response_preview"]
            f.write(json.dumps(entry) + "\n")

    print(f"\n  Results logged to {log_path}")

    # Summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY BY TIER")
    print(f"{'=' * 80}")
    for tier in ("deep", "shallow", "missing"):
        tier_results = [r for r in results if r["expected_depth"] == tier]
        if not tier_results:
            continue
        avg_conf = sum(r["confidence"] for r in tier_results) / len(tier_results)
        avg_top = sum(r["top_similarity"] for r in tier_results) / len(tier_results)
        uncertain_pct = sum(1 for r in tier_results if r["uncertain"]) / len(tier_results) * 100
        print(
            f"  {tier:<10}: avg_confidence={avg_conf:.3f}  avg_top_sim={avg_top:.3f}  "
            f"uncertain={uncertain_pct:.0f}%"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Wheeler-primary decoder quality")
    parser.add_argument("--data-dir", default=None, help="Wheeler data directory")
    parser.add_argument("--decode", action="store_true", help="Also run small model decoder")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Decoder model (default: qwen2.5:1.5b)")
    parser.add_argument("--ollama", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--confidence-floor", type=float, default=0.18, help="Confidence floor")
    args = parser.parse_args()

    run_evaluation(
        data_dir=args.data_dir,
        decode=args.decode,
        model=args.model,
        ollama_url=args.ollama,
        confidence_floor=args.confidence_floor,
    )


if __name__ == "__main__":
    main()
