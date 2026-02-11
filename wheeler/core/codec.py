import hashlib
import numpy as np
import torch
import re
from typing import Tuple, List

class TextCodec:
    def __init__(self, width: int = 128, height: int = 128):
        self.width = width
        self.height = height
    
    def _generate_word_pattern(self, word: str) -> np.ndarray:
        """Generate a deterministic full-size grid for a word."""
        # Normalize word
        word = word.lower()
        
        # Seed generator with word hash
        seed = int(hashlib.sha256(word.encode('utf-8')).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        
        # Generate full grid of noise
        return rng.randn(self.width, self.height).astype(np.float32)

    def _tokenize(self, text: str) -> List[str]:
        """Split text into words, removing punctuation."""
        # Simple regex tokenizer
        text = text.lower()
        # Keep only alphanumeric
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def encode(self, text: str) -> torch.Tensor:
        """Encode text into a 128x128 grid using Bag-of-Words."""
        grid = np.zeros((self.width, self.height), dtype=np.float32)
        
        words = self._tokenize(text)
        if not words:
            return torch.from_numpy(grid)
            
        for word in words:
            pattern = self._generate_word_pattern(word)
            grid += pattern
            
        # Normalize to prevent explosion
        # Tanh is good for squashing to [-1, 1] range, typical for neural/dynamic inputs
        # But simple division by sqrt(N) maintains unit variance properties if inputs are unit variance
        # Let's use sqrt(N) scaling
        scale = np.sqrt(len(words)) if len(words) > 0 else 1.0
        grid = (grid / scale).astype(np.float32)
        
        return torch.from_numpy(grid)