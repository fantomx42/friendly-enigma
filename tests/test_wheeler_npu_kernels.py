import numpy as np
import pytest
from wheeler_ai_training.wheeler_cpu import WheelerMemoryCPU

def test_single_tick_reference():
    """Define the reference output for a single tick of Wheeler dynamics."""
    width, height = 64, 64
    engine = WheelerMemoryCPU(width, height, use_numba=False)
    
    # Seeded input for reproducibility
    rng = np.random.RandomState(42)
    input_frame = rng.uniform(-1, 1, (height, width)).astype(np.float32)
    
    # Run 1 tick
    trajectory = engine.run_dynamics(input_frame, max_ticks=1)
    output_frame = trajectory.final_frame
    
    # Verify shape
    assert output_frame.shape == (height, width)
    
    # Verify values are in range
    assert np.all(output_frame >= -1.0)
    assert np.all(output_frame <= 1.0)
    
    # Save reference for NPU comparison later
    np.save("ai_tech_stack/tests/reference_tick_input.npy", input_frame)
    np.save("ai_tech_stack/tests/reference_tick_output.npy", output_frame)
    print("Reference tick input and output saved.")

def test_npu_tick_accuracy():
    """Verify that the NPU implementation matches the CPU reference for a single tick."""
    import sys
    import os
    # Add the specific directory to path to avoid package-level imports
    npu_engine_dir = os.path.join(os.getcwd(), "ai_tech_stack/ralph_core")
    if npu_engine_dir not in sys.path:
        sys.path.insert(0, npu_engine_dir)
    
    from npu_engine import NPUWheelerEngine
    
    input_frame = np.load("ai_tech_stack/tests/reference_tick_input.npy")
    expected_output = np.load("ai_tech_stack/tests/reference_tick_output.npy")
    
    engine = NPUWheelerEngine(device="NPU")
    actual_output = engine.run_tick(input_frame)
    
    # Compare with tolerance (floats can have slight differences)
    # OpenVINO NPU might have different precision, let's check.
    mse = np.mean((expected_output - actual_output)**2)
    print(f"Mean Squared Error: {mse}")
    
    # High tolerance initially to see what we get
    np.testing.assert_allclose(actual_output, expected_output, atol=1e-3, rtol=1e-3)
    print("NPU tick output matches CPU reference (within FP16 tolerance)!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "npu":
        test_npu_tick_accuracy()
    else:
        test_single_tick_reference()
