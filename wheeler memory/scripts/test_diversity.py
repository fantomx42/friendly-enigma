"""CLI: Attractor diversity validation -- THE key test.

Generates attractors for 20 diverse inputs and checks that they are
genuinely distinct (avg off-diagonal correlation < 0.5, no pair > 0.85).
Saves a visual report to diversity_report.png.
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from wheeler_memory import evolve_and_interpret, hash_to_frame

TEST_INPUTS = [
    "Fix authentication bug in login flow",
    "Deploy Kubernetes cluster on AWS",
    "Buy groceries: milk, eggs, bread",
    "Schedule dentist appointment for Thursday",
    "Quantum entanglement violates Bell inequalities",
    "The mitochondria is the powerhouse of the cell",
    "Review pull request #42 for memory leaks",
    "Plan birthday party for next Saturday",
    "Configure NGINX reverse proxy with TLS",
    "Water the garden every morning at 7am",
    "Implement binary search tree in Rust",
    "Book flight to Tokyo for March conference",
    "Dark matter comprises 27% of the universe",
    "Refactor database schema for multi-tenancy",
    "Practice piano scales for 30 minutes daily",
    "Debug segfault in GPU kernel launch",
    "Write unit tests for payment processing",
    "Organize closet by season and color",
    "Black holes emit Hawking radiation",
    "Compile FFmpeg with hardware acceleration",
]


def main():
    parser = argparse.ArgumentParser(description="Wheeler Memory attractor diversity test")
    parser.add_argument("--output", default="diversity_report.png", help="Output image path")
    args = parser.parse_args()

    n = len(TEST_INPUTS)
    attractors = []
    states = []
    ticks_list = []

    print(f"Evolving {n} test inputs...")
    total_start = time.time()

    for i, text in enumerate(TEST_INPUTS):
        frame = hash_to_frame(text)
        start = time.time()
        result = evolve_and_interpret(frame)
        elapsed = time.time() - start

        attractors.append(result["attractor"].flatten())
        states.append(result["state"])
        ticks_list.append(result["convergence_ticks"])

        label = text[:50]
        print(f"  [{i + 1:2d}/{n}] {result['state']:<11} {result['convergence_ticks']:>4} ticks  {elapsed:.3f}s  {label}")

    total_time = time.time() - total_start

    # Compute correlation matrix
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif j > i:
                c, _ = pearsonr(attractors[i], attractors[j])
                corr_matrix[i, j] = c
                corr_matrix[j, i] = c

    # Statistics
    off_diag = corr_matrix[np.triu_indices(n, k=1)]
    avg_corr = float(np.mean(np.abs(off_diag)))
    max_corr = float(np.max(np.abs(off_diag)))
    min_corr = float(np.min(np.abs(off_diag)))

    print(f"\n{'=' * 60}")
    print(f"DIVERSITY REPORT")
    print(f"{'=' * 60}")
    print(f"Total time:           {total_time:.2f}s")
    print(f"Converged:            {states.count('CONVERGED')}/{n}")
    print(f"Avg ticks:            {np.mean(ticks_list):.0f}")
    print(f"Avg |correlation|:    {avg_corr:.4f}")
    print(f"Max |correlation|:    {max_corr:.4f}")
    print(f"Min |correlation|:    {min_corr:.4f}")

    pass_avg = avg_corr < 0.5
    pass_max = max_corr < 0.85
    print(f"\nAvg < 0.5:  {'PASS' if pass_avg else 'FAIL'} ({avg_corr:.4f})")
    print(f"Max < 0.85: {'PASS' if pass_max else 'FAIL'} ({max_corr:.4f})")
    print(f"Overall:    {'PASS' if (pass_avg and pass_max) else 'FAIL'}")

    # Generate visual report
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Correlation matrix heatmap
    ax = axes[0, 0]
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Attractor Correlation Matrix")
    ax.set_xlabel("Memory Index")
    ax.set_ylabel("Memory Index")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Correlation distribution histogram
    ax = axes[0, 1]
    ax.hist(off_diag, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(0.85, color="red", linestyle="--", label="Max threshold (0.85)")
    ax.axvline(avg_corr, color="orange", linestyle="--", label=f"Avg |r| = {avg_corr:.3f}")
    ax.set_title("Off-Diagonal Correlation Distribution")
    ax.set_xlabel("Pearson Correlation")
    ax.set_ylabel("Count")
    ax.legend()

    # Sample attractors (first 4)
    for idx in range(4):
        row, col = divmod(idx, 2)
        ax = axes[1, col] if idx < 2 else None
        if idx >= 2:
            break
        att = attractors[idx].reshape(64, 64)
        ax.imshow(att, cmap="RdBu_r", vmin=-1, vmax=1)
        label = TEST_INPUTS[idx][:30]
        ax.set_title(f"Attractor {idx}: {label}...")
        ax.axis("off")

    # Replace bottom row with a 4-attractor grid
    axes[1, 0].clear()
    axes[1, 1].clear()
    for idx in range(4):
        ax = fig.add_subplot(2, 4, 5 + idx)
        att = attractors[idx].reshape(64, 64)
        ax.imshow(att, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"#{idx}", fontsize=9)
        ax.axis("off")
    axes[1, 0].set_visible(False)
    axes[1, 1].set_visible(False)

    plt.suptitle(
        f"Wheeler Memory Diversity Report\n"
        f"Avg |r|={avg_corr:.3f}, Max |r|={max_corr:.3f} — "
        f"{'PASS' if (pass_avg and pass_max) else 'FAIL'}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved visual report to {args.output}")


if __name__ == "__main__":
    main()
