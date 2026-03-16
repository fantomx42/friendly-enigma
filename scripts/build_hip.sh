#!/usr/bin/env bash
# Rebuild Wheeler Memory HIP kernels.
# Run manually or triggered automatically via the pacman hook after ROCm updates.
#
# Usage:
#   bash scripts/build_hip.sh           # rebuild both v1 and v2
#   bash scripts/build_hip.sh --verify  # rebuild + run CPU vs GPU correctness check
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_DIR="$REPO_DIR/wheeler_memory/gpu"
LOG="/tmp/wheeler-hip-build.log"

echo "=== Wheeler HIP rebuild $(date) ===" | tee "$LOG"
echo "GPU dir: $GPU_DIR" | tee -a "$LOG"

cd "$GPU_DIR"
make clean 2>&1 | tee -a "$LOG"
make all   2>&1 | tee -a "$LOG"

if [[ "${1:-}" == "--verify" ]]; then
    echo "=== Verifying CPU vs GPU correctness ===" | tee -a "$LOG"
    cd "$REPO_DIR"
    source .venv/bin/activate
    python scripts/bench_gpu.py --verify-only 2>&1 | tee -a "$LOG"
fi

echo "=== Done. Log: $LOG ===" | tee -a "$LOG"
