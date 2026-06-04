#!/usr/bin/env bash
# smoke.sh — end-to-end sanity for wheeler-memory.
#
# Exercises the install + the four core CLIs (info, store, recall, bench)
# against an isolated tempdir so the user's real ~/.wheeler_memory/ is
# untouched. Exits 0 on success, 1 on any failure with a clear marker.
#
# Run from the wheeler-memory repo root:
#   ./.claude/skills/run-wheeler-memory/smoke.sh
#
# Expected wall-clock: ~5s on a warm cache (model already loaded once),
# ~25s on a cold start (downloads/loads MiniLM).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "FAIL: .venv missing. Bootstrap with:" >&2
  echo "  python -m venv .venv && .venv/bin/pip install -e '.[embed]'" >&2
  exit 1
fi

VENV_BIN="$REPO_ROOT/.venv/bin"
TMPDIR_TEST="$(mktemp -d -t wheeler-smoke-XXXXXX)"
trap 'rm -rf "$TMPDIR_TEST"' EXIT

echo "==> wheeler-info (system + GPU detection)"
"$VENV_BIN/wheeler-info" 2>/dev/null | "$VENV_BIN/python" -c '
import json, sys
# wheeler-info emits JSON then a colorized footer; parse only the leading object
text = sys.stdin.read()
info, _ = json.JSONDecoder().raw_decode(text)
assert "cpu" in info and "memory" in info, "wheeler-info missing keys"
cores = info["cpu"]["physical_cores"]
mem = info["memory"]["total_gb"]
dev = info.get("optimal_device", "?")
print(f"  cores={cores} mem={mem:.1f}GB device={dev}")
'

echo "==> wheeler-store + wheeler-recall (hash encoder, tempdir=$TMPDIR_TEST)"
"$VENV_BIN/wheeler-store" --data-dir "$TMPDIR_TEST" \
  "The mitochondria is the powerhouse of the cell" >/dev/null
"$VENV_BIN/wheeler-store" --data-dir "$TMPDIR_TEST" \
  "Quantum entanglement violates Bell inequalities" >/dev/null
RECALL_OUT="$("$VENV_BIN/wheeler-recall" --data-dir "$TMPDIR_TEST" "powerhouse cell" 2>/dev/null)"
if ! grep -q "mitochondria\|Quantum" <<<"$RECALL_OUT"; then
  echo "FAIL: wheeler-recall returned no known memories" >&2
  echo "$RECALL_OUT" >&2
  exit 1
fi
echo "  store/recall round-trip ok"

echo "==> wheeler-bench --json --no-save (quality benchmark, target score < 0.25)"
BENCH_JSON="$("$VENV_BIN/wheeler-bench" --json --no-save 2>/dev/null)"
"$VENV_BIN/python" -c '
import json, sys
r = json.loads(sys.argv[1])
score = r["score"]
corr = r["avg_correlation"]
conv = r["n_converged"]
total = r["n_total"]
sec = r["elapsed_seconds"]
print(f"  score={score:.4f}  avg|r|={corr:.4f}  "
      f"converged={conv}/{total}  elapsed={sec:.2f}s")
if score >= 0.25:
    print(f"FAIL: bench score {score:.4f} >= 0.25 quality target", file=sys.stderr)
    sys.exit(1)
' "$BENCH_JSON"

echo "==> OK: wheeler-memory smoke passed"
