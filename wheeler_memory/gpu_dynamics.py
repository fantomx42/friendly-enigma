"""GPU-accelerated CA dynamics — SHIM.

This module re-exports from wheeler_memory.accel.ca for backwards compatibility.
The actual implementation has moved to wheeler_memory/accel/ca.py.
New code should import from wheeler_memory.accel directly.
"""

from .accel.ca import (  # noqa: F401
    gpu_available,
    gpu_evolve_batch,
    gpu_evolve_single,
    gpu_query_vram,
    gpu_version,
)

__all__ = [
    "gpu_available",
    "gpu_evolve_batch",
    "gpu_evolve_single",
    "gpu_query_vram",
    "gpu_version",
]
