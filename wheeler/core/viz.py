import torch
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List

def render_frame(frame: torch.Tensor, output_path: str, title: str = "Wheeler Frame"):
    """Renders a single frame as a heatmap."""
    if isinstance(frame, torch.Tensor):
        data = frame.detach().cpu().numpy()
    else:
        data = frame

    plt.figure(figsize=(10, 8))
    sns.heatmap(data, cmap="magma", cbar=True)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def render_comparison(f1: torch.Tensor, f2: torch.Tensor, output_path: str, title: str = "Frame Comparison"):
    """Renders two frames side-by-side."""
    if isinstance(f1, torch.Tensor):
        d1 = f1.detach().cpu().numpy()
    else:
        d1 = f1
    if isinstance(f2, torch.Tensor):
        d2 = f2.detach().cpu().numpy()
    else:
        d2 = f2

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    sns.heatmap(d1, ax=axes[0], cmap="magma", cbar=False)
    axes[0].set_title("Frame A")
    axes[0].axis('off')
    
    sns.heatmap(d2, ax=axes[1], cmap="magma", cbar=False)
    axes[1].set_title("Frame B")
    axes[1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def animate_trajectory(trajectory: List[torch.Tensor], output_path: str, fps: int = 10):
    """Creates an animated GIF from a trajectory of frames."""
    from PIL import Image
    
    images = []
    for frame in trajectory:
        # Convert to numpy and normalize to [0, 255]
        if isinstance(frame, torch.Tensor):
            data = frame.detach().cpu().numpy()
        else:
            data = frame
            
        # Normalize for visualization
        # We'll use a fixed range or dynamic per-frame?
        # Fixed range [-2, 2] is usually good for these dynamics
        data = np.clip(data, -2.0, 2.0)
        data = ((data + 2.0) / 4.0 * 255).astype(np.uint8)
        
        # Create image from grayscale data
        img = Image.fromarray(data, mode='L')
        # Apply a colormap? Pillow 'L' is grayscale. 
        # For color, we'd need to map it. Let's keep it simple grayscale for now.
        images.append(img.convert('RGB'))
        
    if images:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=1000 // fps,
            loop=0
        )
