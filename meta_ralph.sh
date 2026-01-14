#!/bin/bash
# meta_ralph.sh - The Recursive Self-Improvement Loop
# Usage: ./meta_ralph.sh

echo "🧬 STARTING META-RALPH PROTOCOL"
echo "Objective: Analyze own source code and implement one specific improvement."

# 1. Define the Meta-Objective
# We use a file to store the meta-prompt to ensure it's loaded correctly
META_OBJECTIVE="Analyze the python code in 'ralph_core/'. Identify ONE specific area for refactoring or optimization (e.g., error handling, type safety, performance). Create a plan and IMPLEMENT the improvement safely. Do not break existing functionality."

# 2. Launch the standard loop with the Meta-Objective
# We use the sandbox mode to ensure he doesn't delete himself by accident
./ralph_loop.sh --sandbox "$META_OBJECTIVE"
