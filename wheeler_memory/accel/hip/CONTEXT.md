# wheeler_memory/accel/hip/ — HIP Kernel Sources

Compiled GPU kernels for AMD GPUs (ROCm/HIP). Targets RDNA4 (gfx1201, RX 9070 XT).

## Kernels

| Source | Library | Status | Purpose |
|--------|---------|--------|---------|
| `ca_evolve.hip` | `libwheeler_ca.so` | Working | CA evolution, variable grid, batch (v2) |
| `ca_evolve_v1.hip` | `libwheeler_ca_v1.so` | Legacy | CA evolution, 64x64 fixed grid (v1) |
| `hip_encode.hip` | `libwheeler_encode.so` | Planned | Hippocampus + context-RI encoding |
| `hip_similarity.hip` | `libwheeler_similarity.so` | Planned | Batch Pearson correlation search |
| `hip_cortex.hip` | `libwheeler_cortex.so` | Planned | Settlement CA on graph |

## Building

```bash
make            # Build all available kernels
make ca         # Just the CA evolution kernel
make clean      # Remove all .so files
make test       # Verify exported symbols

GPU_ARCH=gfx1100 make   # RDNA 3 (RX 7000 series)
```

## RDNA4 Architecture Notes (gfx1201)

- **Wave32/Wave64**: Default Wave32 for compute shaders
- **LDS**: 64KB per WGP — fits 2x 64x64 float grids (32KB) with room for scratch
- **WMMA**: Wave Matrix Multiply Accumulate — hardware matrix multiply, use for encoding kernel
- **VOPD**: Dual-issue VALU — compiler exploits this automatically with -O3
- **Packed FP16**: 2x throughput for half-precision math
- **L2 cache**: 8MB, 256-byte cacheline
- **Infinity Cache (L3)**: 64MB

## Python Bindings

All bindings live in `wheeler_memory/accel/`:
- `ca.py` — CA evolution (gpu_evolve_single, gpu_evolve_batch)
- `encode.py` — Encoding (planned)
- `similarity.py` — Similarity search (planned)
- `cortex.py` — Cortex scoring (planned)
