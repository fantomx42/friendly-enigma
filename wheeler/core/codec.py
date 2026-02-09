import hashlib
import numpy as np
import torch
from typing import Tuple

class TextCodec:
    def __init__(self, width: int = 128, height: int = 128):
        self.width = width
        self.height = height
        self.stamp_size = 8
    
    def _generate_stamp(self, char: str) -> np.ndarray:
        """Generate a deterministic 8x8 stamp for a character."""
        # Seed generator with character hash
        seed = int(hashlib.sha256(char.encode('utf-8')).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        return rng.randn(self.stamp_size, self.stamp_size).astype(np.float32)

    def _get_position(self, index: int, char: str) -> Tuple[int, int]:
        """Determine deterministic position for a character based on index."""
        # Simple spiral or hashed position?
        # Hashed position ensures distributed encoding.
        key = f"{index}:{char}"
        h = int(hashlib.sha256(key.encode('utf-8')).hexdigest()[:8], 16)
        
        # Keep away from very edges to avoid OOB
        max_x = self.width - self.stamp_size
        max_y = self.height - self.stamp_size
        
        x = h % max_x
        y = (h // max_x) % max_y
        return x, y

    def encode(self, text: str) -> torch.Tensor:
        """Encode text into a 128x128 grid."""
        grid = np.zeros((self.width, self.height), dtype=np.float32)
        
        if not text:
            return torch.from_numpy(grid)
            
        for i, char in enumerate(text):
            stamp = self._generate_stamp(char)
            x, y = self._get_position(i, char)
            
            # Add stamp to grid
            grid[x:x+self.stamp_size, y:y+self.stamp_size] += stamp
            
        return torch.from_numpy(grid)
