"""GPU-accelerated CA dynamics via HIP kernel (AMD ROCm).

Loads the compiled HIP shared library and provides a Python interface
matching the CPU evolve_and_interpret API. Falls back gracefully
when the GPU library is not available.

Library preference order (first found wins):
  1. libwheeler_ca.so   — v2, variable grid size (migrated from gpu/ca_kernel_v2.hip)
  2. libwheeler_ca_v1.so — v1, fixed 64x64 grid (legacy)
"""

from __future__ import annotations

import ctypes
import logging
import os

import numpy as np

from ..constants import (
    ALIVE_THRESHOLD,
    MAX_PUSH_STRENGTH,
    MIN_ALIVE_FRACTION,
    SLOPE_FLOW_STRENGTH,
)
from ._common import _float_ptr, _int_ptr, _try_load

_lib = None
_lib_version: int | None = None


def _load_lib():
    """Try to load the HIP shared library (v2 preferred, v1 fallback)."""
    global _lib, _lib_version
    if _lib is not None:
        return _lib

    # Try v2 first
    lib = _try_load("libwheeler_ca.so")
    if lib is not None:
        try:
            lib.ca_evolve_batch_v2.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # frames_in
                ctypes.POINTER(ctypes.c_float),  # frames_out
                ctypes.POINTER(ctypes.c_int),  # ticks_out
                ctypes.POINTER(ctypes.c_int),  # states_out
                ctypes.c_int,  # batch_size
                ctypes.c_int,  # grid_w
                ctypes.c_int,  # max_iters
                ctypes.c_float,  # stability_threshold
                ctypes.c_float,  # push_strength
                ctypes.c_float,  # slope_strength
                ctypes.c_float,  # alive_threshold
                ctypes.c_float,  # min_alive_fraction
            ]
            lib.ca_evolve_batch_v2.restype = ctypes.c_int

            lib.ca_evolve_single_v2.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,  # grid_w
                ctypes.c_int,  # max_iters
                ctypes.c_float,  # stability_threshold
                ctypes.c_float,  # push_strength
                ctypes.c_float,  # slope_strength
                ctypes.c_float,  # alive_threshold
                ctypes.c_float,  # min_alive_fraction
            ]
            lib.ca_evolve_single_v2.restype = ctypes.c_int

            lib.ca_query_vram.argtypes = [ctypes.c_int, ctypes.c_int]
            lib.ca_query_vram.restype = ctypes.c_size_t

            _lib = lib
            _lib_version = 2
            return _lib
        except (AttributeError, OSError) as e:
            logging.warning("Could not bind GPU library v2 symbols: %s", e)

    # Fall back to v1
    lib = _try_load("libwheeler_ca_v1.so")
    if lib is not None:
        try:
            lib.ca_evolve_batch.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.ca_evolve_batch.restype = ctypes.c_int

            lib.ca_evolve_single.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
            ]
            lib.ca_evolve_single.restype = ctypes.c_int

            _lib = lib
            _lib_version = 1
            return _lib
        except (AttributeError, OSError) as e:
            logging.warning("Could not bind GPU library v1 symbols: %s", e)

    return None


def gpu_available() -> bool:
    """Check if the GPU backend is ready.

    Honors the WHEELER_DISABLE_GPU env var: set to "1"/"true"/"yes" to
    force False without probing hardware (used for reproducible CPU-only
    bench runs across kernel binary rebuilds).
    """
    if os.environ.get("WHEELER_DISABLE_GPU", "").lower() in ("1", "true", "yes"):
        return False
    return _load_lib() is not None


def gpu_version() -> int | None:
    """Return the loaded kernel version (1 or 2), or None if not loaded."""
    _load_lib()
    return _lib_version


_STATE_NAMES = {0: "CONVERGED", 1: "OSCILLATING", 2: "CHAOTIC"}


def gpu_evolve_single(
    frame: np.ndarray,
    max_iters: int = 1000,
    stability_threshold: float = 1e-4,
    grid_w: int = 64,
    push_strength: float | None = None,
    slope_strength: float | None = None,
    alive_threshold: float | None = None,
    min_alive_fraction: float | None = None,
) -> dict:
    """Evolve a single frame on GPU.

    Returns dict with same keys as evolve_and_interpret:
      - state, attractor, convergence_ticks, metadata
      - history is NOT available (GPU doesn't store per-tick frames)
    """
    lib = _load_lib()
    if lib is None:
        raise RuntimeError(
            "GPU library not available. Build with: cd wheeler_memory/accel/hip && make"
        )

    if frame.shape != (grid_w, grid_w):
        raise ValueError(
            f"Expected frame shape ({grid_w}, {grid_w}), got {frame.shape}"
        )

    cells = grid_w * grid_w
    frame_in = np.ascontiguousarray(frame.flatten(), dtype=np.float32)
    frame_out = np.zeros(cells, dtype=np.float32)
    ticks = ctypes.c_int(0)
    state = ctypes.c_int(0)

    _push = push_strength if push_strength is not None else MAX_PUSH_STRENGTH
    _slope = slope_strength if slope_strength is not None else SLOPE_FLOW_STRENGTH
    _alive = alive_threshold if alive_threshold is not None else ALIVE_THRESHOLD
    _min_alive = (
        min_alive_fraction if min_alive_fraction is not None else MIN_ALIVE_FRACTION
    )

    if _lib_version == 2:
        ret = lib.ca_evolve_single_v2(
            _float_ptr(frame_in),
            _float_ptr(frame_out),
            ctypes.byref(ticks),
            ctypes.byref(state),
            grid_w,
            max_iters,
            ctypes.c_float(stability_threshold),
            ctypes.c_float(_push),
            ctypes.c_float(_slope),
            ctypes.c_float(_alive),
            ctypes.c_float(_min_alive),
        )
    else:
        ret = lib.ca_evolve_single(
            _float_ptr(frame_in),
            _float_ptr(frame_out),
            ctypes.byref(ticks),
            ctypes.byref(state),
            max_iters,
        )

    if ret != 0:
        raise RuntimeError("GPU kernel execution failed")

    return {
        "state": _STATE_NAMES.get(state.value, "CHAOTIC"),
        "attractor": frame_out.reshape(grid_w, grid_w),
        "convergence_ticks": ticks.value,
        "history": [],
        "metadata": {"backend": f"gpu_v{_lib_version}", "grid_w": grid_w},
    }


def gpu_evolve_batch(
    frames: list[np.ndarray],
    max_iters: int = 1000,
    stability_threshold: float = 1e-4,
    grid_w: int = 64,
    push_strength: float | None = None,
    slope_strength: float | None = None,
    alive_threshold: float | None = None,
    min_alive_fraction: float | None = None,
) -> list[dict]:
    """Evolve a batch of frames on GPU in parallel.

    This is where the real speedup lives — 71x at batch=1000 on RX 9070 XT.
    """
    lib = _load_lib()
    if lib is None:
        raise RuntimeError(
            "GPU library not available. Build with: cd wheeler_memory/accel/hip && make"
        )

    batch_size = len(frames)
    if batch_size == 0:
        return []

    cells = grid_w * grid_w
    flat_in = np.zeros(batch_size * cells, dtype=np.float32)
    for i, f in enumerate(frames):
        flat_in[i * cells : (i + 1) * cells] = f.flatten().astype(np.float32)

    flat_out = np.zeros_like(flat_in)
    ticks_out = np.zeros(batch_size, dtype=np.int32)
    states_out = np.zeros(batch_size, dtype=np.int32)

    _push = push_strength if push_strength is not None else MAX_PUSH_STRENGTH
    _slope = slope_strength if slope_strength is not None else SLOPE_FLOW_STRENGTH
    _alive = alive_threshold if alive_threshold is not None else ALIVE_THRESHOLD
    _min_alive = (
        min_alive_fraction if min_alive_fraction is not None else MIN_ALIVE_FRACTION
    )

    if _lib_version == 2:
        ret = lib.ca_evolve_batch_v2(
            _float_ptr(flat_in),
            _float_ptr(flat_out),
            _int_ptr(ticks_out),
            _int_ptr(states_out),
            batch_size,
            grid_w,
            max_iters,
            ctypes.c_float(stability_threshold),
            ctypes.c_float(_push),
            ctypes.c_float(_slope),
            ctypes.c_float(_alive),
            ctypes.c_float(_min_alive),
        )
    else:
        ret = lib.ca_evolve_batch(
            _float_ptr(flat_in),
            _float_ptr(flat_out),
            _int_ptr(ticks_out),
            _int_ptr(states_out),
            batch_size,
            max_iters,
        )

    if ret != 0:
        raise RuntimeError("GPU kernel execution failed")

    results = []
    for i in range(batch_size):
        results.append(
            {
                "state": _STATE_NAMES.get(int(states_out[i]), "CHAOTIC"),
                "attractor": flat_out[i * cells : (i + 1) * cells]
                .reshape(grid_w, grid_w)
                .copy(),
                "convergence_ticks": int(ticks_out[i]),
                "history": [],
                "metadata": {"backend": f"gpu_v{_lib_version}", "grid_w": grid_w},
            }
        )

    return results


def gpu_query_vram(batch_size: int, grid_w: int = 64) -> int | None:
    """Estimate VRAM bytes needed for a given batch + grid config (v2 only)."""
    lib = _load_lib()
    if lib is None or _lib_version != 2:
        return None
    return lib.ca_query_vram(batch_size, grid_w)
