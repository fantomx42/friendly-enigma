"""
ASIC Spawner - Dynamic Specialist Model Invocation

Spawns ultra-small specialist models for micro-tasks and returns
multiple OPTIONS for middle management to evaluate.

Key Features:
- Generates multiple candidate solutions (2-3 options)
- Parallel spawning for efficiency
- Automatic fallback to larger models
- Metrics tracking for all calls
"""

import requests
import concurrent.futures
from typing import Optional
from .registry import (
    ASIC_REGISTRY,
    ASICConfig,
    get_asic_config,
    get_model_with_fallback,
    OLLAMA_API,
)

# Import metrics tracker - handle import error gracefully
try:
    from ..metrics import tracker
except ImportError:
    tracker = None


def _call_asic(
    model: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    option_index: int,
) -> tuple[int, str]:
    """
    Internal function to call an ASIC model.
    Returns (option_index, response) for ordering.
    """
    # Vary temperature slightly for diversity in options
    temp_variance = option_index * 0.05
    actual_temp = min(temperature + temp_variance, 1.0)

    full_prompt = f"{system_prompt}\n\n{prompt}"

    # Add diversity instruction for non-first options
    if option_index > 0:
        full_prompt += f"\n\n(Provide a DIFFERENT approach than previous attempts. Option {option_index + 1})"

    try:
        response = requests.post(
            OLLAMA_API,
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": actual_temp,
                    "num_predict": max_tokens,
                }
            },
            timeout=120  # ASICs should be fast
        )

        if response.status_code == 404:
            return (option_index, f"Error: Model {model} not found")

        response.raise_for_status()
        data = response.json()

        # Log metrics if tracker available
        if tracker:
            tracker.log_llm_call(
                model=f"ASIC:{model}",
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                duration_ms=data.get("total_duration", 0) // 1_000_000
            )

        return (option_index, data.get("response", "").strip())

    except Exception as e:
        return (option_index, f"ASIC ERROR: {str(e)}")


def spawn_asic(task_type: str, prompt: str) -> list[str]:
    """
    Spawn an ASIC for the given task type and return multiple options.

    Args:
        task_type: Type of specialist (e.g., "regex", "json", "test")
        prompt: The specific task prompt

    Returns:
        List of candidate solutions (2-3 options typically)

    Example:
        options = spawn_asic("regex", "Match valid email addresses")
        # Returns: [r"^[\\w.-]+@[\\w.-]+\\.\\w+$", r"[a-zA-Z0-9._%+-]+@...", ...]
    """
    config = get_asic_config(task_type)

    if not config:
        # Unknown task type - use small_code as default
        config = ASIC_REGISTRY.get("small_code")
        if not config:
            return [f"Error: Unknown ASIC type '{task_type}' and no fallback available"]

    model = get_model_with_fallback(config)
    options = []

    # Generate options sequentially (simpler, more reliable)
    for i in range(config.num_options):
        _, result = _call_asic(
            model=model,
            prompt=prompt,
            system_prompt=config.system_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            option_index=i,
        )
        options.append(result)

    return options


def spawn_asic_parallel(task_type: str, prompt: str) -> list[str]:
    """
    Spawn an ASIC with parallel option generation for speed.

    Same as spawn_asic but generates all options concurrently.
    Use when latency matters more than resource efficiency.
    """
    config = get_asic_config(task_type)

    if not config:
        config = ASIC_REGISTRY.get("small_code")
        if not config:
            return [f"Error: Unknown ASIC type '{task_type}'"]

    model = get_model_with_fallback(config)

    # Parallel generation
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.num_options) as executor:
        futures = [
            executor.submit(
                _call_asic,
                model=model,
                prompt=prompt,
                system_prompt=config.system_prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                option_index=i,
            )
            for i in range(config.num_options)
        ]

        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Sort by option index to maintain order
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]


def spawn_multiple_asics(tasks: list[tuple[str, str]]) -> dict[str, list[str]]:
    """
    Spawn multiple different ASICs in parallel.

    Args:
        tasks: List of (task_type, prompt) tuples

    Returns:
        Dict mapping task_type to list of options

    Example:
        results = spawn_multiple_asics([
            ("regex", "Match email"),
            ("test", "Test the email validator"),
        ])
    """
    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_to_task = {
            executor.submit(spawn_asic, task_type, prompt): task_type
            for task_type, prompt in tasks
        }

        for future in concurrent.futures.as_completed(future_to_task):
            task_type = future_to_task[future]
            try:
                results[task_type] = future.result()
            except Exception as e:
                results[task_type] = [f"Error: {str(e)}"]

    return results
