import torch
import torch.nn.functional as F
from typing import List

class DynamicsEngine:
    def __init__(self, width: int = 128, height: int = 128, device: str = "cpu"):
        self.width = width
        self.height = height
        self.device = device
        
        # Reaction-diffusion parameters
        self.dt = 0.1
        self.diffusion = 0.2
        self.decay = 0.01
        
        # Fixed Laplacian kernel for diffusion
        # [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        self.kernel = torch.tensor([
            [[[0.0, 1.0, 0.0],
              [1.0, -4.0, 1.0],
              [0.0, 1.0, 0.0]]]
        ], dtype=torch.float32).to(self.device)

    def _pad_circular(self, x: torch.Tensor, pad: int) -> torch.Tensor:
        """Circular padding for wrapped boundaries."""
        # x shape: (1, 1, H, W)
        return F.pad(x, (pad, pad, pad, pad), mode='circular')

    def step(self, grid: torch.Tensor) -> torch.Tensor:
        """Perform one step of reaction-diffusion dynamics."""
        # Ensure input is on correct device and has batch/channel dims
        # Expected grid: (H, W) or (1, H, W) or (1, 1, H, W)
        if grid.dim() == 2:
            x = grid.unsqueeze(0).unsqueeze(0)
        elif grid.dim() == 3:
            x = grid.unsqueeze(0)
        else:
            x = grid

        x = x.to(self.device)
        
        # Wrapped boundaries via padding
        x_padded = self._pad_circular(x, 1)
        
        # Diffusion term: Laplacian
        laplacian = F.conv2d(x_padded, self.kernel)
        
        # Reaction term: simple non-linearity (tanh) - decay
        # reaction = torch.tanh(x) - self.decay * x
        # Combined update: x += (D * laplacian + reaction) * dt
        # Simplified: x += (diffusion * laplacian - decay * x + tanh(x)) * dt
        
        # We want attractors, so we need a balancing force.
        # Let's use a variation of Neural CA:
        # dx = Diffusion + Perception -> Update
        
        # For this MVP, we stick to the architecture description:
        # "reaction-diffusion with tanh activation"
        
        update = (self.diffusion * laplacian) - (self.decay * x) + torch.tanh(x)
        
        new_x = x + update * self.dt
        
        # Clamp or normalize to keep values stable? 
        # Tanh keeps update bounded, but x can grow.
        # Let's apply a soft clip via tanh on the output or just let it evolve.
        # To ensure stability/attractors, usually a saturation function is used.
        new_x = torch.clamp(new_x, -10.0, 10.0) 
        
        return new_x.squeeze()

    def run(self, grid: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """Run dynamics for N steps."""
        current = grid
        for _ in range(steps):
            current = self.step(current)
        return current

    def run_trajectory(self, grid: torch.Tensor, steps: int = 10) -> List[torch.Tensor]:
        """Run dynamics and return all intermediate frames."""
        current = grid
        trajectory = [current]
        for _ in range(steps):
            current = self.step(current)
            trajectory.append(current)
        return trajectory

    def run_with_stats(self, grid: torch.Tensor, steps: int = 10) -> tuple[torch.Tensor, float]:
        """Run dynamics and return final grid + stability score."""
        current = grid
        stability = 0.0
        
        for _ in range(steps):
            prev = current
            current = self.step(current)
            
            # Calculate stability (inverse of change)
            diff = torch.mean(torch.abs(current - prev))
            # Normalize diff roughly? 
            # If diff is 0, stability is 1.0.
            # If diff is large, stability is low.
            # Simple exponential decay: exp(-diff)
            stability = float(torch.exp(-diff * 10).item())
            
        return current, stability
