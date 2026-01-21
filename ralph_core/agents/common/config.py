# --- Configuration ---
MODELS = {
    # Core Agents
    "orchestrator": "deepseek-r1:14b",
    "engineer": "qwen2.5-coder:14b",
    "designer": "mistral-nemo:12b",
    "translator": "phi3:mini",
    # ASICs (micro-specialists)
    "asic_tiny": "deepseek-coder:1.3b",
    "asic_small": "qwen2.5-coder:1.5b",
    "asic_ultra": "tinyllama:1.1b",
}

OLLAMA_API = "http://localhost:11434/api/generate"
