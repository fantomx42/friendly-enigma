"""CLI: Inspect and manage the SCM (Structural Coherence Map) trust topology.

Usage:
    wheeler-scm                   Show SCM statistics
    wheeler-scm --heatmap         Display ASCII heatmap of SCM openness
    wheeler-scm --reset           Reset SCM to fully permissive (all zeros)
"""

import argparse
import sys

from wheeler_memory.scm_grid import SCMGrid


def _ascii_heatmap(scm: SCMGrid, width: int = 64) -> str:
    """Render SCM as ASCII heatmap. Brighter = more closed."""
    import numpy as np

    grid = np.abs(scm.grid)
    chars = " .:-=+*#%@"
    lines = []
    # Downsample if needed
    step = max(1, 64 // width)
    for r in range(0, 64, step):
        row = ""
        for c in range(0, 64, step):
            val = float(grid[r, c])
            idx = min(int(val * (len(chars) - 1)), len(chars) - 1)
            row += chars[idx]
        lines.append(row)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Inspect SCM trust topology")
    parser.add_argument(
        "--data-dir", default=None, help="Data directory (default: ~/.wheeler_memory)"
    )
    parser.add_argument(
        "--heatmap", action="store_true", help="Show ASCII heatmap of SCM"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset SCM to fully permissive"
    )
    args = parser.parse_args()

    scm = SCMGrid.load_or_create(args.data_dir)

    if args.reset:
        import numpy as np

        scm.grid[:] = 0.0
        scm.hardening[:] = 0
        scm.save()
        print("SCM reset to fully permissive (all zeros).")
        return

    stats = scm.stats()
    print(f"SCM Trust Topology")
    print(f"==================")
    print(f"  Openness:       {stats['openness']:.1%}")
    print(f"  Mean |SCM|:     {stats['mean_abs_scm']:.4f}")
    print(f"  Max |SCM|:      {stats['max_abs_scm']:.4f}")
    print(f"  Hardened cells: {stats['hardened_cells']} / 4096")
    print(f"  Mean hardening: {stats['mean_hardening']:.1f}")
    print(f"  Max hardening:  {stats['max_hardening']}")
    print(f"  Closed cells:   {stats['closed_cells']} / 4096")
    print()

    if args.heatmap:
        print("SCM Heatmap (brighter = more closed):")
        print(_ascii_heatmap(scm))


if __name__ == "__main__":
    main()
