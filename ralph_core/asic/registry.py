"""
ASIC Registry - Specialist Model Configurations

Defines ultra-small, task-specific models that act as "LLM ASICs" -
specialized processors for narrow tasks that return multiple options
for middle management to evaluate.

Design Philosophy:
- Small models (< 3B params) for speed and efficiency
- Low temperature for deterministic outputs
- Multiple options per query (2-3 candidates)
- Fallback to larger models if specialist unavailable
"""

from dataclasses import dataclass
from typing import Optional
import requests

OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_TAGS_API = "http://localhost:11434/api/tags"


@dataclass
class ASICConfig:
    """Configuration for a specialist ASIC model."""
    model: str              # Ollama model name
    temperature: float      # Generation temperature (low = deterministic)
    num_options: int        # Number of options to generate
    max_tokens: int         # Max tokens per option
    system_prompt: str      # Role-specific system prompt
    fallback_model: str     # Fallback if primary not available


# Registry of specialist ASICs
# Consolidated for single-GPU (16GB): Only 2 small models + phi3:mini for translation
# - tinyllama:1.1b (637MB) for simple pattern tasks
# - qwen2.5-coder:1.5b (986MB) for code-heavy tasks
ASIC_REGISTRY: dict[str, ASICConfig] = {
    "regex": ASICConfig(
        model="tinyllama:1.1b",
        temperature=0.1,
        num_options=3,
        max_tokens=256,
        system_prompt="You are a regex specialist. Output ONLY valid regex patterns, no explanation.",
        fallback_model="qwen2.5-coder:1.5b"
    ),
    "json": ASICConfig(
        model="tinyllama:1.1b",
        temperature=0.1,
        num_options=2,
        max_tokens=512,
        system_prompt="You are a JSON specialist. Output ONLY valid JSON, no explanation.",
        fallback_model="qwen2.5-coder:1.5b"
    ),
    "sql": ASICConfig(
        model="tinyllama:1.1b",
        temperature=0.2,
        num_options=2,
        max_tokens=512,
        system_prompt="You are a SQL specialist. Output ONLY valid SQL queries, no explanation.",
        fallback_model="qwen2.5-coder:1.5b"
    ),
    "docstring": ASICConfig(
        model="tinyllama:1.1b",
        temperature=0.3,
        num_options=2,
        max_tokens=256,
        system_prompt="You are a documentation specialist. Write concise, clear docstrings.",
        fallback_model="qwen2.5-coder:1.5b"
    ),
    "test": ASICConfig(
        model="qwen2.5-coder:1.5b",
        temperature=0.2,
        num_options=3,
        max_tokens=1024,
        system_prompt="You are a test specialist. Write pytest test functions. Output ONLY code.",
        fallback_model="tinyllama:1.1b"
    ),
    "refactor": ASICConfig(
        model="qwen2.5-coder:1.5b",
        temperature=0.2,
        num_options=2,
        max_tokens=1024,
        system_prompt="You are a refactoring specialist. Improve code structure. Output ONLY code.",
        fallback_model="tinyllama:1.1b"
    ),
    "fix": ASICConfig(
        model="qwen2.5-coder:1.5b",
        temperature=0.1,
        num_options=3,
        max_tokens=512,
        system_prompt="You are a bug fix specialist. Fix the code error. Output ONLY the corrected code.",
        fallback_model="tinyllama:1.1b"
    ),
    "translate": ASICConfig(
        model="phi3:mini",
        temperature=0.5,
        num_options=1,
        max_tokens=2048,
        system_prompt="You are a translator. Convert human requests into structured specifications.",
        fallback_model="qwen2.5-coder:1.5b"
    ),
    "tiny_code": ASICConfig(
        model="tinyllama:1.1b",
        temperature=0.2,
        num_options=3,
        max_tokens=256,
        system_prompt="You are a minimal code generator. Output ONLY short code snippets.",
        fallback_model="qwen2.5-coder:1.5b"
    ),
    "small_code": ASICConfig(
        model="qwen2.5-coder:1.5b",
        temperature=0.2,
        num_options=2,
        max_tokens=1024,
        system_prompt="You are a code generator. Output clean, efficient code.",
        fallback_model="tinyllama:1.1b"
    ),
}


def get_asic_config(task_type: str) -> Optional[ASICConfig]:
    """Get configuration for a specific ASIC type."""
    return ASIC_REGISTRY.get(task_type)


def list_available_asics() -> list[str]:
    """List all registered ASIC types."""
    return list(ASIC_REGISTRY.keys())


def check_model_available(model_name: str) -> bool:
    """Check if a model is available in Ollama."""
    try:
        response = requests.get(OLLAMA_TAGS_API, timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            available = [m["name"] for m in models]
            # Check both exact match and base name match
            return model_name in available or any(
                m.startswith(model_name.split(":")[0]) for m in available
            )
    except Exception:
        pass
    return False


def get_model_with_fallback(config: ASICConfig) -> str:
    """Get the best available model, falling back if primary unavailable."""
    if check_model_available(config.model):
        return config.model
    if check_model_available(config.fallback_model):
        return config.fallback_model
    # Last resort - use the largest available coder
    return "qwen2.5-coder:14b"
