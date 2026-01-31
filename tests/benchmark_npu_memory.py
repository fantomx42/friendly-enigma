import numpy as np
import time
import sys
import os

# Add ai_tech_stack to path
sys.path.append(os.path.join(os.getcwd(), "ai_tech_stack"))

def run_accuracy_benchmark(iterations=100):
    from ralph_core.npu_engine import NPUWheelerEngine
    from wheeler_ai_training.wheeler_cpu import WheelerMemoryCPU
    
    print(f"Running Accuracy Benchmark ({iterations} iterations)...")
    
    width, height = 128, 128
    npu_engine = NPUWheelerEngine(model_path="ai_tech_stack/models/wheeler_dynamics_128.xml", device="NPU")
    cpu_engine = WheelerMemoryCPU(width, height, use_numba=False)
    
    mses = []
    
    for i in range(iterations):
        # Random input
        input_frame = np.random.uniform(-1, 1, (height, width)).astype(np.float32)
        
        # CPU result
        cpu_traj = cpu_engine.run_dynamics(input_frame, max_ticks=1)
        cpu_out = cpu_traj.final_frame
        
        # NPU result
        npu_out = npu_engine.run_tick(input_frame)
        
        # MSE
        mse = np.mean((cpu_out - npu_out)**2)
        mses.append(mse)
        
        if (i+1) % 20 == 0:
            print(f"  Iteration {i+1}/{iterations}...")

    avg_mse = np.mean(mses)
    print(f"Average Mean Squared Error: {avg_mse}")
    assert avg_mse < 1e-5, f"Accuracy too low! Avg MSE: {avg_mse}"
    print("Accuracy test PASSED.")

def run_performance_benchmark(ticks=50):
    from ralph_core.npu_engine import NPUWheelerEngine
    from wheeler_ai_training.wheeler_cpu import WheelerMemoryCPU
    
    print(f"Running Performance Benchmark ({ticks} ticks)...")
    
    width, height = 128, 128
    input_frame = np.random.uniform(-1, 1, (height, width)).astype(np.float32)
    
    # CPU
    cpu_engine = WheelerMemoryCPU(width, height, use_numba=False)
    start = time.perf_counter()
    cpu_engine.run_dynamics(input_frame, max_ticks=ticks)
    cpu_time = (time.perf_counter() - start) * 1000
    print(f"CPU Time ({ticks} ticks): {cpu_time:.2f} ms ({cpu_time/ticks:.2f} ms/tick)")
    
    # NPU
    npu_engine = NPUWheelerEngine(model_path="ai_tech_stack/models/wheeler_dynamics_128.xml", device="NPU")
    start = time.perf_counter()
    npu_engine.run_dynamics(input_frame, max_ticks=ticks)
    npu_time = (time.perf_counter() - start) * 1000
    print(f"NPU Time ({ticks} ticks): {npu_time:.2f} ms ({npu_time/ticks:.2f} ms/tick)")
    
    improvement = (cpu_time - npu_time) / cpu_time * 100
    print(f"Performance Improvement: {improvement:.2f}%")

if __name__ == "__main__":
    try:
        run_accuracy_benchmark()
        print("-" * 30)
        run_performance_benchmark()
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
