#!/usr/bin/env python3
"""Semantic Apple Test — does Wheeler's attractor topology predict missing concepts?

Unlike the original apple_test() in theories/synthesis.py (which uses hash_to_frame),
this test uses embed_to_frame for semantic similarity.  It tests whether
embedding-based attractors form topologically meaningful neighborhoods.

Protocol:
  1. Store all domain items EXCEPT the holdout using embed_to_frame
  2. Query for the holdout using recall_memory(use_embedding=True)
  3. Measure which stored memories fire and with what similarity
  4. Verdict: do the semantically closest neighbors activate?

Usage:
    python scripts/apple_test_semantic.py
    python scripts/apple_test_semantic.py --domain ml_architecture
    python scripts/apple_test_semantic.py --data-dir /tmp/apple_test
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

# ── Domain definitions ────────────────────────────────────────────────────────

DOMAINS = {
    "ml_architecture": {
        "items": [
            "Attention mechanism allows models to focus on relevant parts of the input sequence",
            "Self-attention computes relationships between all positions in a sequence simultaneously",
            "Multi-head attention runs multiple attention operations in parallel for richer representations",
            "Positional encoding injects sequence order information since attention is permutation invariant",
            "Encoder-decoder architecture maps input sequences to output sequences through a bottleneck",
            "BERT uses bidirectional masked language modeling to pretrain deep representations",
            "GPT generates text autoregressively by predicting the next token given previous context",
            "Recurrent neural networks maintain hidden state to process sequential data one step at a time",
            "Convolutional neural networks extract hierarchical spatial features using learned filters",
            "Feed-forward networks in transformers apply two linear transformations with a ReLU activation",
            "Residual connections allow gradients to flow directly through the network via skip connections",
            "Layer normalization normalizes activations across features within each training example",
        ],
        "holdout": "Transformer architecture combines self-attention with feed-forward layers and residual connections",
    },
    "physics": {
        "items": [
            "Newton's first law states objects remain at rest or in uniform motion unless acted upon by a force",
            "Conservation of energy states energy cannot be created or destroyed only transformed",
            "Einstein's special relativity shows time dilates and length contracts at high velocities",
            "Electromagnetic waves propagate at the speed of light through oscillating electric and magnetic fields",
            "Wave-particle duality means quantum objects exhibit both wave and particle behavior",
            "Heisenberg's uncertainty principle limits simultaneous knowledge of position and momentum",
            "Quantum superposition allows particles to exist in multiple states until measured",
            "Schrodinger's equation describes how quantum states evolve over time",
            "The standard model classifies all known fundamental particles and three of four forces",
        ],
        "holdout": "Quantum entanglement links distant particles so measuring one instantly determines the other",
    },
    "biology": {
        "items": [
            "DNA encodes genetic information using four nucleotide bases adenine thymine guanine cytosine",
            "RNA transcription copies DNA sequences into messenger RNA for protein synthesis",
            "Ribosomes translate mRNA codons into amino acid sequences to build proteins",
            "Cell division through mitosis produces identical copies for growth and repair",
            "Meiosis produces gametes with half the chromosome number enabling sexual reproduction",
            "Natural selection drives evolution by favoring traits that improve reproductive fitness",
            "Mutations introduce genetic variation that provides raw material for evolution",
            "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy",
            "The immune system defends against pathogens using innate and adaptive responses",
            "Enzymes are biological catalysts that accelerate specific biochemical reactions",
        ],
        "holdout": "Mitochondria generate ATP through oxidative phosphorylation providing the cell's energy currency",
    },
}


# ── Test protocol ─────────────────────────────────────────────────────────────


def semantic_apple_test(
    items: list[str],
    holdout: str,
    domain_name: str,
    data_dir: Path | None = None,
) -> dict:
    """Run the semantic apple test.

    Returns dict with:
      - verdict: "topology" | "lookup" | "silent"
      - holdout_text: the excluded concept
      - top_hits: recall results (which neighbors fired)
      - top_similarity: best Pearson correlation
      - semantic_relevance: are top hits from the same domain?
    """
    from wheeler_memory.brick import MemoryBrick
    from wheeler_memory.dynamics import evolve_and_interpret
    from wheeler_memory.hippocampus import hippocampus_to_frame
    from wheeler_memory.storage import recall_memory, store_memory
    from wheeler_memory.hashing import text_to_hex

    cleanup = False
    if data_dir is None:
        tmp = tempfile.mkdtemp(prefix="wheeler_apple_sem_")
        data_dir = Path(tmp)
        cleanup = True
    else:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

    log = []
    training_items = [item for item in items if item != holdout]

    # Step 1: Store all training items using embedding
    log.append(f"Domain: {domain_name}")
    log.append(f"Storing {len(training_items)} items (holdout excluded)")
    stored_keys = {}
    for item in training_items:
        frame = hippocampus_to_frame(item)
        result = evolve_and_interpret(frame)
        brick = MemoryBrick.from_evolution_result(result)
        hex_key = store_memory(
            item,
            result,
            brick,
            data_dir=data_dir,
            auto_evict=False,
        )
        stored_keys[hex_key] = item
        log.append(
            f"  Stored: '{item[:60]}...' → {result['state']} ({result['convergence_ticks']} ticks)"
        )

    # Step 2: Query for the holdout using semantic recall
    log.append(f"\nQuerying for holdout: '{holdout[:80]}...'")
    hits = recall_memory(
        holdout,
        top_k=5,
        data_dir=data_dir,
        encoder="blended",
    )

    log.append(f"  Recalled {len(hits)} memories:")
    for i, h in enumerate(hits, 1):
        log.append(
            f"    {i}. sim={h['similarity']:.3f} temp={h.get('temperature_tier', '?')} "
            f"'{h['text'][:60]}...'"
        )

    # Step 3: Analyze results
    top_sim = hits[0]["similarity"] if hits else 0.0
    top_text = hits[0]["text"] if hits else ""

    # Verdict logic
    if not hits or top_sim < 0.1:
        verdict = "silent"
        log.append(
            f"\nVERDICT: SILENT — no meaningful activation (top_sim={top_sim:.3f})"
        )
    elif top_sim > 0.25:
        verdict = "topology"
        log.append(
            f"\nVERDICT: TOPOLOGY — semantic neighbors activated (top_sim={top_sim:.3f})"
        )
        log.append(f"  Top hit: '{top_text[:80]}...'")
    else:
        verdict = "weak_topology"
        log.append(
            f"\nVERDICT: WEAK TOPOLOGY — some activation but low similarity (top_sim={top_sim:.3f})"
        )

    # Step 4: Control — what happens with hash_to_frame? (no semantic similarity)
    from wheeler_memory.hashing import hash_to_frame

    hash_frame = hash_to_frame(holdout)
    hash_result = evolve_and_interpret(hash_frame)
    hash_att = hash_result["attractor"].flatten()

    # Compare hash attractor to all stored attractors
    from scipy.stats import pearsonr

    hash_correlations = []
    for h in hits:
        hex_key = h["hex_key"]
        stored_att = np.load(
            data_dir / "chunks" / h["chunk"] / "attractors" / f"{hex_key}.npy"
        )
        corr, _ = pearsonr(hash_att, stored_att.flatten())
        hash_correlations.append(float(corr))

    max_hash_corr = max(hash_correlations) if hash_correlations else 0.0
    log.append(f"\n  Control (hash_to_frame): max correlation = {max_hash_corr:.3f}")
    log.append(f"  Embedding advantage: {top_sim - max_hash_corr:.3f}")

    if cleanup:
        shutil.rmtree(data_dir, ignore_errors=True)

    return {
        "domain": domain_name,
        "verdict": verdict,
        "holdout": holdout,
        "top_similarity": top_sim,
        "top_hit_text": top_text,
        "all_hits": [{"text": h["text"], "similarity": h["similarity"]} for h in hits],
        "hash_max_correlation": max_hash_corr,
        "embedding_advantage": top_sim - max_hash_corr,
        "log": log,
    }


def run_all_domains(data_dir: Path | None = None):
    """Run semantic apple test across all domains."""
    print("=" * 70)
    print("SEMANTIC APPLE TEST")
    print("=" * 70)

    results = {}
    for name, config in DOMAINS.items():
        print(f"\n{'─' * 70}")
        result = semantic_apple_test(
            config["items"],
            config["holdout"],
            name,
            data_dir=data_dir,
        )
        results[name] = result
        for line in result["log"]:
            print(f"  {line}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"{'Domain':<18} {'Verdict':<16} {'Top Sim':>8} {'Hash Ctrl':>10} {'Advantage':>10}"
    )
    print(f"{'─' * 18} {'─' * 16} {'─' * 8} {'─' * 10} {'─' * 10}")
    for name, r in results.items():
        print(
            f"{name:<18} {r['verdict']:<16} {r['top_similarity']:>8.3f} "
            f"{r['hash_max_correlation']:>10.3f} {r['embedding_advantage']:>10.3f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Apple Test for Wheeler Memory"
    )
    parser.add_argument(
        "--domain",
        choices=list(DOMAINS.keys()),
        default=None,
        help="Run specific domain only",
    )
    parser.add_argument(
        "--data-dir", default=None, help="Data directory (default: temp)"
    )
    args = parser.parse_args()

    if args.domain:
        config = DOMAINS[args.domain]
        result = semantic_apple_test(
            config["items"],
            config["holdout"],
            args.domain,
            data_dir=args.data_dir,
        )
        for line in result["log"]:
            print(f"  {line}")
    else:
        run_all_domains(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
