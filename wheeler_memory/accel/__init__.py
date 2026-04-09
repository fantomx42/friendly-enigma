"""Accelerator backends for Wheeler Memory.

Provides GPU (HIP/ROCm) and future NPU acceleration for CA evolution,
encoding, similarity search, and cortex scoring.

Library loading order: accel/hip/ .so files are loaded lazily on first use.
All functions fall back gracefully to CPU when no accelerator is available.
"""

from __future__ import annotations


def gpu_available() -> bool:
    """Check if any GPU backend is ready."""
    from .ca import gpu_available as _ca_available

    return _ca_available()


def gpu_version() -> int | None:
    """Return loaded CA kernel version (1 or 2), or None."""
    from .ca import gpu_version as _ver

    return _ver()


def accel_info() -> dict:
    """Return a summary of available accelerator backends."""
    info: dict = {"gpu": False, "gpu_version": None, "npu": False}
    try:
        from .ca import gpu_available as _ca, gpu_version as _ver

        info["gpu"] = _ca()
        info["gpu_version"] = _ver()
    except Exception:
        pass
    try:
        from ..npu import npu_available as _npu

        info["npu"] = _npu()
    except Exception:
        pass
    return info
