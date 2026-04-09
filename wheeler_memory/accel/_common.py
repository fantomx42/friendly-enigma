"""Shared utilities for accelerator backends.

Common ctypes helpers, path resolution, and library loading patterns
used across all HIP kernel bindings.
"""

from __future__ import annotations

import ctypes
import os

_HIP_DIR = os.path.join(os.path.dirname(__file__), "hip")


def _lib_path(name: str) -> str:
    """Resolve path to a compiled .so in the hip/ directory."""
    return os.path.join(_HIP_DIR, name)


def _try_load(so_name: str) -> ctypes.CDLL | None:
    """Try to load a shared library from accel/hip/. Returns None on failure."""
    path = _lib_path(so_name)
    if not os.path.exists(path):
        return None
    try:
        return ctypes.CDLL(path)
    except OSError:
        return None


def _float_ptr(arr):
    """Get a ctypes float pointer from a numpy array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _int_ptr(arr):
    """Get a ctypes int pointer from a numpy array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
