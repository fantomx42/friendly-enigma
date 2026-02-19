# GPU Acceleration

Wheeler Memory supports GPU-accelerated CA evolution via **HIP kernels** on AMD GPUs (ROCm).

## Requirements

- AMD GPU with ROCm support
- `hipcc` compiler (comes with ROCm)
- Tested on: **AMD Radeon RX 9070 XT** (gfx1201, RDNA 4)

## Build

```bash
cd wheeler_memory/gpu
make                    # default: gfx1201
GPU_ARCH=gfx1100 make   # for RDNA 3 (RX 7000 series)
```

This compiles `libwheeler_ca.so` — the HIP kernel shared library.

## Performance

Batch processing is where GPU shines. Single inputs are comparable to CPU.

| Batch Size | CPU (s) | GPU (s) | Speedup | GPU samples/s |
|------------|---------|---------|---------|---------------|
| 1 | 0.0024 | 0.0021 | 1.1× | 472 |
| 10 | 0.0221 | 0.0085 | 2.6× | 1,172 |
| 100 | 0.2388 | 0.0055 | **43.8×** | 18,321 |
| 500 | 1.1904 | 0.0205 | **58.1×** | 24,423 |
| 1,000 | 2.3838 | 0.0337 | **70.7×** | 29,648 |

> First GPU call has ~250ms cold-start (HIP runtime init). Subsequent calls run in single-digit milliseconds.

## CLI Tools

```bash
# Benchmark GPU vs CPU
wheeler-bench-gpu

# Correctness check only
wheeler-bench-gpu --verify-only

# Custom batch sizes
wheeler-bench-gpu --batch-sizes 100,500,2000,5000

# Run diversity test on GPU
wheeler-diversity-math --n 1000 --gpu
```

## Python API

```python
from wheeler_memory import gpu_available, gpu_evolve_batch, gpu_evolve_single, hash_to_frame

if gpu_available():
    # Single input
    frame = hash_to_frame("some text")
    result = gpu_evolve_single(frame)
    # result["state"], result["attractor"], result["convergence_ticks"]

    # Batch (where the speedup is)
    frames = [hash_to_frame(t) for t in texts]
    results = gpu_evolve_batch(frames)
```

## Architecture

```
Input frames (B × 64×64)
    ↓
hipMemcpy → GPU device memory
    ↓
ca_step_kernel: B × 4096 threads in parallel
    (each thread: read 4 neighbors → classify → update)
    ↓
reduce_delta_kernel: mean |Δ| per frame (shared-mem)
    ↓
Check convergence → repeat or stop
    ↓
hipMemcpy → CPU result
```

The GPU path produces numerically identical results to CPU (verified via `np.allclose` with `atol=1e-4`).

## Files

| File | Purpose |
|------|---------|
| `wheeler_memory/gpu/ca_kernel.hip` | HIP C++ kernel source |
| `wheeler_memory/gpu/Makefile` | Build script |
| `wheeler_memory/gpu/libwheeler_ca.so` | Compiled library (not tracked in git) |
| `wheeler_memory/gpu_dynamics.py` | Python ctypes bindings |
| `scripts/bench_gpu.py` | Benchmark CLI |

## Fallback

If `libwheeler_ca.so` is not built, `gpu_available()` returns `False` and all existing CPU functionality works unchanged. The GPU backend is strictly opt-in.
