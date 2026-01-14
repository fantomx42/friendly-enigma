"""
ASIC (Application-Specific Integrated Circuit) Module

This module provides dynamic spawning of ultra-small, specialized LLMs
for micro-tasks. Each ASIC is optimized for a specific task type and
returns multiple OPTIONS for middle management to evaluate.

Usage:
    from ralph_core.asic import spawn_asic, ASIC_REGISTRY

    options = spawn_asic("regex", "Create a pattern to match email addresses")
    # Returns: ["option1", "option2", "option3"]
"""

from .registry import ASIC_REGISTRY, get_asic_config, list_available_asics
from .spawner import spawn_asic, spawn_asic_parallel
from .handler import handle_message as asic_handle_message, ASIC_HANDLERS

__all__ = [
    "ASIC_REGISTRY",
    "get_asic_config",
    "list_available_asics",
    "spawn_asic",
    "spawn_asic_parallel",
    # Message handling
    "asic_handle_message",
    "ASIC_HANDLERS",
]
