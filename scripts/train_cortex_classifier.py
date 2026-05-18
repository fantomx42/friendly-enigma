#!/usr/bin/env python3
"""Train the cortex L3 classifier on MMLU dev+validation data.

Trains a ~11K parameter neural network to choose among 4 MMLU answer choices
based on cortex features: settled opinions, choice similarities, and SCM layers.

The classifier learns to integrate multiple cortex signals (L1 graph reasoning,
L2 settlement dynamics, SCM coherence scores) into a 4-way classification.

Usage
-----
    python scripts/train_cortex_classifier.py --epochs 10 --lr 0.001
    python scripts/train_cortex_classifier.py --subjects abstract_algebra --epochs 5
    python scripts/train_cortex_classifier.py --seed 42 --output /tmp/classifier.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation between two vectors."""
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_std = x_centered.std()
    y_std = y_centered.std()

    if x_std < 1e-10 or y_std < 1e-10:
        return 0.0

    return float(np.dot(x_centered, y_centered) / (len(x) * x_std * y_std))


def load_training_data(
    subjects: list[str] | None = None,
    max_samples: int | None = None,
    data_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load MMLU dev+validation data and compute cortex features.

    For each question:
    1. Encode question + choices with hippocampus
    2. Retrieve top-K attractors using cache (if available)
    3. Run cortex_reason() for graph reasoning and settlement
    4. Compute SCM scores
    5. Compute co-evolution scores
    6. Build feature vector: settlement + choice_sims + scm_layers + coevo_layers

    Args:
        subjects: List of MMLU subjects (None = all 57)
        max_samples: Max questions to load per subject (None = all)
        data_dir: Path to Wheeler Memory data directory

    Returns:
        Tuple of (features, targets) where:
        - features: (N, 24) array where N = number of questions, 24 = K + 4 + 7 + 3
        - targets: (N,) int array with correct answer index (0-3)
    """
    from wheeler_memory.storage import DEFAULT_DATA_DIR
    from wheeler_memory.cortex import cortex_reason
    from wheeler_memory.cortex_scm import compute_scm
    from wheeler_memory.constants import (
        CORTEX_K,
        CORTEX_CLUSTER_THRESHOLD,
        CORTEX_CONTRADICTION_THRESHOLD,
        CORTEX_SETTLE_MAX_STEPS,
        CORTEX_SETTLE_THRESHOLD,
        CORTEX_SETTLE_INERTIA,
    )

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    if subjects is None:
        # All 57 MMLU subjects
        from scripts.wheeler_mmlu import MMLU_SUBJECTS

        subjects = MMLU_SUBJECTS

    print(f"\nLoading MMLU data for {len(subjects)} subjects...")
    d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    # Build attractor cache once for speed
    cache = None
    try:
        from scripts.wheeler_mmlu import AttractorCache

        cache = AttractorCache(d)
    except Exception as e:
        print(f"  Warning: failed to load attractor cache: {e}")
        print(f"  Will work with cortex input from choices alone")

    features_list = []
    targets_list = []

    cortex_k = CORTEX_K
    total_loaded = 0
    total_valid = 0

    from scripts.wheeler_mmlu import load_mmlu

    for subject in subjects:
        subject_loaded = 0

        for split in ["dev", "validation"]:
            for subject_name, question, choices, answer_idx in load_mmlu(
                [subject], split=split, samples=max_samples
            ):
                try:
                    from wheeler_memory.hippocampus import hippocampus_to_frame_batch

                    # Encode question + each choice
                    query_texts = [f"{question} {c}" for c in choices]
                    frames = hippocampus_to_frame_batch(query_texts)
                    frames_flat = [f.flatten().astype(np.float32) for f in frames]

                    # Compute similarity of each choice to question attractor
                    question_frame_2d = hippocampus_to_frame_batch([question])[0]
                    question_frame = question_frame_2d.flatten().astype(np.float32)

                    choice_sims = np.array(
                        [
                            _pearson_correlation(question_frame, fc)
                            for fc in frames_flat
                        ],
                        dtype=np.float32,
                    )

                    # Retrieve top-K attractors for this question
                    if cache is not None:
                        hits = cache.search(question_frame_2d, top_k=cortex_k)
                        retrieved_sims = np.array(
                            [h.get("similarity", 0.0) for h in hits], dtype=np.float32
                        )

                        if len(retrieved_sims) < cortex_k:
                            retrieved_sims = np.concatenate(
                                [
                                    retrieved_sims,
                                    np.zeros(
                                        cortex_k - len(retrieved_sims), dtype=np.float32
                                    ),
                                ]
                            )
                        # Create fake attractors from question frame for cortex_reason
                        attractors = np.tile(question_frame, (cortex_k, 1)).astype(
                            np.float32
                        )
                    else:
                        # No cache: use choice attractors as input (4 items), pad to cortex_k
                        retrieved_sims = np.zeros(cortex_k, dtype=np.float32)
                        retrieved_sims[: len(frames_flat)] = choice_sims
                        attractors = np.vstack(
                            [
                                np.vstack(frames_flat),
                                np.random.randn(
                                    cortex_k - len(frames_flat), frames_flat[0].shape[0]
                                ).astype(np.float32),
                            ]
                        )

                    # Run cortex reasoning
                    result = cortex_reason(
                        attractors,
                        retrieved_sims,
                        cluster_threshold=CORTEX_CLUSTER_THRESHOLD,
                        contradiction_threshold=CORTEX_CONTRADICTION_THRESHOLD,
                        max_steps=CORTEX_SETTLE_MAX_STEPS,
                        threshold=CORTEX_SETTLE_THRESHOLD,
                        inertia=CORTEX_SETTLE_INERTIA,
                    )

                    settled = result["settlement"]["settled"][:cortex_k].astype(
                        np.float32
                    )
                    graph = result["graph"]

                    # Ensure settled has exactly K elements
                    if len(settled) < cortex_k:
                        settled = np.concatenate(
                            [settled, np.zeros(cortex_k - len(settled))]
                        )
                    settled = settled[:cortex_k]

                    # Compute SCM
                    best_choice_sim = max(choice_sims) if len(choice_sims) > 0 else 0.0
                    prev_settled = retrieved_sims.copy()

                    scm_result = compute_scm(
                        similarities=retrieved_sims,
                        adjacency=graph.adjacency,
                        cluster_labels=graph.cluster_labels,
                        settled=settled,
                        prev_settled=prev_settled,
                        choice_similarity=best_choice_sim,
                    )

                    # Extract SCMResult fields in classifier's expected order:
                    # [temperature, salience, energy, integration, polarity,
                    #  net_warrant, explanation_readiness]. Distinct from
                    # canonical SCM v2.0 namespace — see CANON §3.5.2.
                    scm_layers = np.array(
                        [
                            scm_result.temperature,
                            scm_result.salience,
                            scm_result.energy,
                            scm_result.integration,
                            scm_result.polarity,
                            scm_result.net_warrant,
                            scm_result.explanation_readiness,
                        ],
                        dtype=np.float32,
                    )

                    # Co-evolution: blend question + each choice, evolve together
                    from wheeler_memory.dynamics import apply_ca_dynamics
                    from wheeler_memory.cortex_scm import (
                        score_coevolution_convergence,
                        score_coevolution_spread,
                        score_coevolution_energy,
                    )

                    coevo_ticks_list = []
                    coevo_energy_list = []
                    for cf in frames:  # frames is the list of 4 choice frames (64x64)
                        blended = np.tanh(0.5 * question_frame_2d + 0.5 * cf).astype(
                            np.float32
                        )
                        frame_t = blended.copy()
                        initial_delta = None
                        final_delta = 0.0
                        ticks_t = 0
                        for t in range(200):
                            frame_old_t = frame_t
                            frame_t = apply_ca_dynamics(frame_t)
                            delta_t = float(np.abs(frame_t - frame_old_t).mean())
                            if initial_delta is None:
                                initial_delta = delta_t
                            final_delta = delta_t
                            ticks_t = t + 1
                            if delta_t < 1e-4:
                                break
                        coevo_ticks_list.append(ticks_t)
                        energy_drop = (
                            (initial_delta - final_delta) / max(initial_delta, 1e-10)
                            if initial_delta
                            else 0.0
                        )
                        coevo_energy_list.append(max(0.0, energy_drop))

                    coevo_ticks_arr = np.array(coevo_ticks_list, dtype=np.float32)
                    coevo_energies_arr = np.array(coevo_energy_list, dtype=np.float32)
                    coevo_layers = np.array(
                        [
                            score_coevolution_convergence(
                                coevo_ticks_arr, max_iters=200
                            ),
                            score_coevolution_spread(coevo_ticks_arr),
                            score_coevolution_energy(coevo_energies_arr),
                        ],
                        dtype=np.float32,
                    )

                    # Build feature vector: settlement + choice_sims + scm_layers + coevo_layers
                    features = np.concatenate(
                        [settled, choice_sims, scm_layers, coevo_layers]
                    ).astype(np.float32)

                    # Check shape
                    if features.shape[0] != 24:
                        continue

                    features_list.append(features)
                    targets_list.append(answer_idx)
                    total_valid += 1

                except Exception as e:
                    # Skip on errors (missing data, encoding failures, etc.)
                    pass

                subject_loaded += 1
                if max_samples and subject_loaded >= max_samples:
                    break

            if max_samples and subject_loaded >= max_samples:
                break

        total_loaded += subject_loaded
        print(f"  [{subject}] loaded {subject_loaded} questions")

    print(f"\nLoaded {total_loaded} questions, {total_valid} valid training examples")

    if not features_list:
        raise ValueError("No valid training examples loaded")

    features = np.array(features_list, dtype=np.float32)
    targets = np.array(targets_list, dtype=np.int32)

    return features, targets


def train_classifier(
    features: np.ndarray,
    targets: np.ndarray,
    epochs: int = 10,
    learning_rate: float = 0.001,
    seed: int = 42,
    validation_split: float = 0.2,
) -> dict:
    """Train the classifier with SGD.

    Args:
        features: (N, 24) array
        targets: (N,) int array with correct answer indices
        epochs: Number of training epochs
        learning_rate: SGD learning rate
        seed: Random seed
        validation_split: Fraction of data to hold out for validation

    Returns:
        Dict with keys: weights, history (loss and accuracy per epoch)
    """
    from wheeler_memory.cortex_classifier import init_weights, backward, classify

    np.random.seed(seed)

    N = features.shape[0]
    n_train = int(N * (1.0 - validation_split))

    indices = np.random.permutation(N)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_features = features[train_idx]
    train_targets = targets[train_idx]
    val_features = features[val_idx]
    val_targets = targets[val_idx]

    print(f"\nTraining configuration:")
    print(f"  Train set: {len(train_idx)} examples")
    print(f"  Validation set: {len(val_idx)} examples")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Seed: {seed}")

    weights = init_weights(input_dim=24, seed=seed)
    history = {"train_loss": [], "val_accuracy": []}

    print(f"\n{'=' * 70}")
    print(f"{'Epoch':>6} {'Loss':>12} {'Val Acc':>10}")
    print(f"{'=' * 70}")

    for epoch in range(epochs):
        # Shuffle training data
        train_indices = np.random.permutation(len(train_features))

        epoch_loss = 0.0
        for i in train_indices:
            x = train_features[i]
            target = train_targets[i]

            weights, loss = backward(x, weights, target, learning_rate=learning_rate)
            epoch_loss += loss

        avg_loss = epoch_loss / len(train_features)
        history["train_loss"].append(avg_loss)

        # Validation accuracy
        val_correct = 0
        for x, target in zip(val_features, val_targets):
            pred_idx, _ = classify(
                x[:10],  # settlement (K=10)
                x[10:14],  # choice_sims (4)
                x[14:21],  # scm_layers (7)
                weights,
                x[21:24],  # coevo_layers (3)
            )
            if pred_idx == target:
                val_correct += 1

        val_acc = val_correct / len(val_targets) if len(val_targets) > 0 else 0.0
        history["val_accuracy"].append(val_acc)

        print(f"{epoch + 1:>6} {avg_loss:>12.4f} {val_acc:>10.1%}")

    print(f"{'=' * 70}")

    return {"weights": weights, "history": history}


def main():
    parser = argparse.ArgumentParser(
        description="Train cortex L3 classifier on MMLU dev+validation data"
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="MMLU subjects to train on (default: all 57)",
    )
    parser.add_argument(
        "--samples-per-subject",
        type=int,
        default=None,
        help="Max samples per subject (default: all)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for weights (default: ~/.wheeler_memory/cortex_classifier.npz)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Wheeler Memory data directory",
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        data_dir = (
            Path(args.data_dir) if args.data_dir else Path.home() / ".wheeler_memory"
        )
        output_path = data_dir / "cortex_classifier.npz"

    try:
        # Load training data
        features, targets = load_training_data(
            subjects=args.subjects,
            max_samples=args.samples_per_subject,
            data_dir=args.data_dir,
        )

        # Train classifier
        result = train_classifier(
            features,
            targets,
            epochs=args.epochs,
            learning_rate=args.lr,
            seed=args.seed,
        )

        # Save weights
        from wheeler_memory.cortex_classifier import save_weights

        weights = result["weights"]
        save_weights(weights, output_path)
        print(f"\nWeights saved to {output_path}")

        print(f"\nTraining complete!")
        print(
            f"  Final validation accuracy: {result['history']['val_accuracy'][-1]:.1%}"
        )

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
