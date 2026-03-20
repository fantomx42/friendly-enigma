#!/usr/bin/env python3
"""Co-activation topology map — which attractors fire together?

Queries pairs of related concepts and records which attractors co-fire,
building an adjacency map of Wheeler's actual attractor landscape.

This reveals:
  - Where the corpus is rich (dense co-activation clusters)
  - Where it is thin (isolated attractors)
  - Where concepts that SHOULD connect don't yet (topology gaps)

Usage:
    python scripts/topology_map.py
    python scripts/topology_map.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from wheeler_memory.storage import recall_memory


# ── Concept pairs to probe ───────────────────────────────────────────────────

# Each pair: (concept_a, concept_b, expected_relation)
# We query each, record top-5 attractors, compute intersection.

CONCEPT_PAIRS = [
    # ML cluster — should be densely connected
    ("self-attention mechanism", "transformer architecture", "core components"),
    ("backpropagation", "gradient descent optimization", "training pipeline"),
    ("BERT masked language model", "GPT autoregressive generation", "pretraining approaches"),
    ("convolutional neural networks", "recurrent neural networks", "architecture types"),
    ("dropout regularization", "overfitting prevention", "training technique"),
    ("encoder-decoder architecture", "sequence to sequence", "architecture pattern"),
    ("word embeddings", "tokenization", "input representation"),

    # Physics cluster — should be connected
    ("quantum superposition", "quantum entanglement", "quantum mechanics"),
    ("Newton's laws of motion", "conservation of energy", "classical mechanics"),
    ("wave-particle duality", "Heisenberg uncertainty principle", "quantum foundations"),
    ("electromagnetic waves", "speed of light", "electromagnetism"),

    # Biology cluster — sparser, expect gaps
    ("DNA replication", "RNA transcription", "molecular biology"),
    ("natural selection evolution", "genetic mutations", "evolutionary biology"),
    ("photosynthesis", "mitochondria ATP", "cell energy"),
    ("immune system", "enzymes catalysts", "biological defense"),

    # Cross-domain — should NOT co-fire strongly
    ("self-attention mechanism", "quantum superposition", "cross: ML-physics"),
    ("backpropagation", "DNA replication", "cross: ML-biology"),
    ("Newton's laws of motion", "convolutional neural networks", "cross: physics-ML"),
]


def query_attractor_set(
    concept: str,
    data_dir: Path | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Query Wheeler and return the active attractor set."""
    hits = recall_memory(
        concept,
        top_k=top_k,
        data_dir=data_dir,
        use_embedding=True,
    )
    return [
        {
            "text": h["text"],
            "similarity": h["similarity"],
            "chunk": h.get("chunk", "?"),
            "hex_key": h.get("hex_key", "?"),
        }
        for h in hits
    ]


def compute_overlap(set_a: list[dict], set_b: list[dict]) -> dict:
    """Compute attractor overlap between two query results."""
    keys_a = {h["hex_key"] for h in set_a}
    keys_b = {h["hex_key"] for h in set_b}

    shared_keys = keys_a & keys_b
    shared_texts = []
    for h in set_a:
        if h["hex_key"] in shared_keys:
            sim_in_b = next(
                (x["similarity"] for x in set_b if x["hex_key"] == h["hex_key"]),
                0.0,
            )
            shared_texts.append({
                "text": h["text"][:80],
                "sim_a": h["similarity"],
                "sim_b": sim_in_b,
            })

    jaccard = len(shared_keys) / len(keys_a | keys_b) if (keys_a | keys_b) else 0.0

    return {
        "overlap_count": len(shared_keys),
        "jaccard": jaccard,
        "total_a": len(keys_a),
        "total_b": len(keys_b),
        "shared": shared_texts,
    }


def run_topology_map(data_dir: Path | None = None):
    """Build the co-activation topology map."""
    print("=" * 75)
    print("CO-ACTIVATION TOPOLOGY MAP")
    print("=" * 75)

    results = []

    for concept_a, concept_b, relation in CONCEPT_PAIRS:
        set_a = query_attractor_set(concept_a, data_dir=data_dir)
        set_b = query_attractor_set(concept_b, data_dir=data_dir)
        overlap = compute_overlap(set_a, set_b)

        top_sim_a = set_a[0]["similarity"] if set_a else 0.0
        top_sim_b = set_b[0]["similarity"] if set_b else 0.0

        result = {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "relation": relation,
            "top_sim_a": top_sim_a,
            "top_sim_b": top_sim_b,
            **overlap,
        }
        results.append(result)

    # Print summary table
    print(f"\n{'Relation':<22} {'Concept A':<30} {'Concept B':<30} {'Overlap':>7} {'Jaccard':>8}")
    print(f"{'─'*22} {'─'*30} {'─'*30} {'─'*7} {'─'*8}")

    for r in results:
        print(
            f"{r['relation']:<22} "
            f"{r['concept_a'][:28]:<30} "
            f"{r['concept_b'][:28]:<30} "
            f"{r['overlap_count']:>3}/{r['total_a']:<3} "
            f"{r['jaccard']:>7.3f}"
        )

    # Detailed per-pair analysis
    print(f"\n{'=' * 75}")
    print("DETAILED CO-ACTIVATION")
    print(f"{'=' * 75}")

    for r in results:
        bridge = "BRIDGE" if r["overlap_count"] > 0 else "GAP"
        print(f"\n  [{bridge}] {r['concept_a']} <-> {r['concept_b']}")
        print(f"  Relation: {r['relation']}")
        print(f"  Top sim A: {r['top_sim_a']:.3f}, Top sim B: {r['top_sim_b']:.3f}")
        print(f"  Overlap: {r['overlap_count']}/{r['total_a']} attractors (Jaccard={r['jaccard']:.3f})")

        if r["shared"]:
            print(f"  Shared attractors:")
            for s in r["shared"]:
                print(f"    '{s['text']}' (sim_a={s['sim_a']:.3f}, sim_b={s['sim_b']:.3f})")

    # Cluster analysis
    print(f"\n{'=' * 75}")
    print("CLUSTER DENSITY")
    print(f"{'=' * 75}")

    clusters = defaultdict(list)
    for r in results:
        domain = r["relation"].split(":")[0] if ":" in r["relation"] else r["relation"]
        clusters[domain].append(r)

    for domain, domain_results in sorted(clusters.items()):
        avg_jaccard = sum(r["jaccard"] for r in domain_results) / len(domain_results)
        avg_overlap = sum(r["overlap_count"] for r in domain_results) / len(domain_results)
        bridges = sum(1 for r in domain_results if r["overlap_count"] > 0)
        gaps = sum(1 for r in domain_results if r["overlap_count"] == 0)

        print(f"\n  {domain}:")
        print(f"    Pairs: {len(domain_results)}, Bridges: {bridges}, Gaps: {gaps}")
        print(f"    Avg Jaccard: {avg_jaccard:.3f}, Avg overlap: {avg_overlap:.1f}/5")

    # Save results
    log_path = Path(data_dir or Path.home() / ".wheeler_memory") / "topology_map.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {log_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Wheeler co-activation topology map")
    parser.add_argument("--data-dir", default=None, help="Wheeler data directory")
    args = parser.parse_args()
    run_topology_map(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
