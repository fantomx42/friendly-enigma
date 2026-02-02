import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import openvino as ov

class WheelerDynamics(nn.Module):
    """
    PyTorch implementation of Wheeler Memory Dynamics (Reaction-Diffusion).
    Takes a 2D grid and returns the next state.
    """
    def __init__(self):
        super().__init__()
        # Fixed kernel for 3x3 diffusion/averaging
        # 1/9 for all elements
        self.kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32) / 9.0
        
    def forward(self, x):
        """
        x: (batch, 1, height, width)
        """
        # Padding 'circular' wraps around (toroidal geometry)
        padded = F.pad(x, (1, 1, 1, 1), mode='circular')
        
        # Convolve (Diffusion)
        avg = F.conv2d(padded, self.kernel)
        
        # Reaction / Activation
        # new = tanh(avg * 1.5 + old * 0.5)
        out = torch.tanh(avg * 1.5 + x * 0.5)
        
        return out

def export_dynamics():
    print("Exporting Wheeler Dynamics to OpenVINO...")
    model = WheelerDynamics()
    model.eval()
    
    # Dummy input (1, 1, 64, 64)
    dummy_input = torch.randn(1, 1, 64, 64)
    
    # Output path
    output_dir = "ai_tech_stack/models"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "wheeler_dynamics.xml")
    
    # Export using OpenVINO
    # First convert to ONNX (intermediate) or use OV direct conversion
    core = ov.Core()
    
    ov_model = ov.convert_model(model, example_input=dummy_input)
    ov.save_model(ov_model, model_path)
    
    print(f"Exported to {model_path}")

    # Also export 128x128 version
    print("Exporting Wheeler Dynamics (128x128)...")
    model128 = WheelerDynamics()
    model128.eval()
    dummy128 = torch.randn(1, 1, 128, 128)
    model128_path = os.path.join(output_dir, "wheeler_dynamics_128.xml")
    ov_model128 = ov.convert_model(model128, example_input=dummy128)
    ov.save_model(ov_model128, model128_path)
    print(f"Exported to {model128_path}")

if __name__ == "__main__":
    export_dynamics()
