#!/usr/bin/env bash
set -e
cd /home/tristan/projects/wheeler-memory
source .venv/bin/activate

PY="python -m scripts.wheeler_learn_words"
SIMLEX="wheeler-simlex"

RESULTS="sweep_decontamination_results.tsv"
printf "k\tspearman_rho\tp_value\n" > "$RESULTS"

for K in 0 1 2 3 4 5 6 8; do
    echo "=== K=$K ==="
    $PY --method context-ri --corpus datasets/wikitext103.jsonl --remove-top-k "$K"

    output=$($SIMLEX --encoder context --mode pearson 2>&1)
    rho=$(echo "$output" | grep -oP 'Spearman rho:\s+\K[+-]?[0-9.]+' | head -1)
    pval=$(echo "$output" | grep -oP 'Spearman rho:.*\(p\s*=\s*\K[0-9.e+-]+')

    printf "%d\t%s\t%s\n" "$K" "$rho" "$pval" >> "$RESULTS"
    echo "  rho=$rho  p=$pval"
done

echo ""
echo "=== Results ==="
column -t -s$'\t' "$RESULTS"
echo ""
best=$(tail -n+2 "$RESULTS" | sort -t$'\t' -k2 -rn | head -1)
echo "Best: K=$(echo "$best" | cut -f1)  rho=$(echo "$best" | cut -f2)"
