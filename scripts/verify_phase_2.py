import torch
import os
from wheeler.core.codec import TextCodec
from wheeler.core.engine import DynamicsEngine
from wheeler.core.viz import render_comparison

def verify():
    print("--- Phase 2 Manual Verification ---")
    codec = TextCodec()
    engine = DynamicsEngine()
    
    test_text = "Wheeler Memory System"
    print(f"1. Encoding text: '{test_text}'")
    initial_grid = codec.encode(test_text)
    
    print("2. Running dynamics (10 steps)...")
    attractor, stability = engine.run_with_stats(initial_grid, steps=10)
    
    print(f"3. Results:")
    print(f"   - Stability Score: {stability:.4f}")
    
    output_dir = "docs/images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "phase_2_verification.png")
    
    print(f"4. Rendering comparison to {output_path}...")
    render_comparison(
        initial_grid, 
        attractor, 
        output_path, 
        title=f"Phase 2 Verification: '{test_text}' (Stability: {stability:.4f})"
    )
    print("--- Done ---")

if __name__ == "__main__":
    verify()
