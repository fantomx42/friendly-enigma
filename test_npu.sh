#!/bin/bash
# test_npu.sh
# Forces a Context Migration event to test the Librarian (NPU)

HOT_FILE="/home/tristan/ralph_brain/hot/active_task.md"
THRESHOLD=4096

echo "🧪 NPU Stress Test Initiated"
echo "---------------------------"
echo "Current size: $(wc -c < "$HOT_FILE") bytes"

echo "📝 Flooding hot memory with synthetic data..."
# Generate ~5KB of dummy log data
for i in {1..100}; do
    echo "- [Log entry $i] generating synthetic cognitive load for neural processing unit testing." >> "$HOT_FILE"
    echo "  Context: The quick brown fox jumps over the lazy dog to verify vectorization." >> "$HOT_FILE"
done

NEW_SIZE=$(wc -c < "$HOT_FILE")
echo "🔥 New size: $NEW_SIZE bytes (Threshold: $THRESHOLD)"

if [ "$NEW_SIZE" -gt "$THRESHOLD" ]; then
    echo "✅ Threshold exceeded. The Librarian should wake up within 5 seconds."
    echo "👀 Watch the Librarian logs for: 'Migration complete'"
else
    echo "⚠️ Not enough data generated. Run this script again."
fi
