# --- Configuration ---
# Unified model architecture: single 14B model for all core roles
# Eliminates model swapping on single-GPU setups (16GB VRAM)
MODELS = {
    # Core Agents
    "orchestrator": "deepseek-r1:14b",
    "engineer": "qwen2.5-coder:14b",
    "designer": "qwen2.5-coder:14b",
    "translator": "phi3:mini",
    # ASICs (micro-specialists) - small models for quick tasks
    "asic_tiny": "tinyllama:1.1b",
    "asic_small": "qwen2.5-coder:1.5b",
    "asic_ultra": "tinyllama:1.1b",
}

OLLAMA_API = "http://localhost:11434/api/generate"
LLAMA_CPP_API = "http://127.0.0.1:8080/completion"
