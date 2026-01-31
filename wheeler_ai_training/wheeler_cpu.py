import numpy as np
from dataclasses import dataclass
from typing import Union

# Mock/Polyfill for missing WheelerMemoryCPU
# Implements basic dynamics to support the system.

NUMBA_AVAILABLE = False
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    pass

@dataclass
class Trajectory:
    final_frame: np.ndarray
    history: list = None

class WheelerMemoryCPU:
    """
    CPU implementation of Wheeler Memory dynamics.
    Simulates a continuous cellular automata / reaction-diffusion system.
    """
    def __init__(self, width: int, height: int, use_numba: bool = True):
        self.width = width
        self.height = height
        self.use_numba = use_numba and NUMBA_AVAILABLE

    def run_dynamics(self, input_data: Union[bytes, np.ndarray], max_ticks: int = 50) -> Trajectory:
        """
        Run the dynamical system for a set number of ticks.
        """
        # Handle input types
        if isinstance(input_data, bytes):
            # If raw bytes passed, create a seeded random frame (fallback)
            # This matches the original call signature in wheeler_ai.py
            frame = self._bytes_to_frame(input_data)
        elif isinstance(input_data, np.ndarray):
            frame = input_data.astype(np.float32).copy()
        else:
            frame = np.zeros((self.height, self.width), dtype=np.float32)

        # Run dynamics
        # For this shim, we implement a simple "smoothing + preservation" dynamic
        # Real Wheeler dynamics would look for attractors.
        
        current_frame = frame
        
        for _ in range(max_ticks):
            # Simple 3x3 box blur (diffusion) + non-linearity
            # This creates "blobs" from points
            
            # Pad for convolution
            padded = np.pad(current_frame, 1, mode='wrap')
            
            # Manual convolution (slow but works without scipy)
            # Center + neighbors
            # This is a very rough approximation of reaction-diffusion
            avg = (
                padded[0:-2, 0:-2] + padded[0:-2, 1:-1] + padded[0:-2, 2:] +
                padded[1:-1, 0:-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
                padded[2:, 0:-2]   + padded[2:, 1:-1]   + padded[2:, 2:]
            ) / 9.0
            
            # Non-linearity (activation)
            # Sigmoid-like to keep values in [-1, 1]
            current_frame = np.tanh(avg * 1.5 + current_frame * 0.5)

        return Trajectory(final_frame=current_frame)

    def _bytes_to_frame(self, data: bytes) -> np.ndarray:
        """Convert bytes to a static noise pattern."""
        # Use hash to seed RNG for determinism
        import hashlib
        seed = int(hashlib.sha256(data).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        return rng.uniform(-1, 1, (self.height, self.width)).astype(np.float32)
