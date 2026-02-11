import torch
from typing import List, Optional

class ReasoningEngine:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def blend(self, frames: List[torch.Tensor], weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        Weighted superposition of multiple frames.
        """
        if not frames:
            return torch.zeros((128, 128), device=self.device)
        
        if weights is None:
            weights = [1.0 / len(frames)] * len(frames)
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
            
        result = torch.zeros_like(frames[0], device=self.device)
        for frame, weight in zip(frames, weights):
            result += frame.to(self.device) * weight
            
        return torch.clamp(result, -10.0, 10.0)

    def contrast(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
        """
        Finds the difference between two frames.
        """
        diff = frame_a.to(self.device) - frame_b.to(self.device)
        return torch.clamp(diff, -10.0, 10.0)

    def amplify(self, frame: torch.Tensor, strength: float = 1.5) -> torch.Tensor:
        """
        Non-linear enhancement of dominant patterns.
        Using sign(x) * |x|^(1/strength) where strength > 1.0 to pull small values up,
        or we can use a different power. Let's stick to the logic from the numpy version
        which used x**(1/strength) to broaden patterns.
        """
        frame_dev = frame.to(self.device)
        amplified = torch.sign(frame_dev) * (torch.abs(frame_dev) ** (1.0 / strength))
        return torch.clamp(amplified, -10.0, 10.0)
