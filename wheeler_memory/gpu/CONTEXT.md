# wheeler_memory/gpu/ — HIP/CUDA Kernel Sources

Compiled GPU kernels for CA evolution. Provides ~10-100x speedup over CPU for batch evolution.

## Files
- **wheeler_ca.hip** — HIP kernel source. Implements the 3-state CA rule on GPU.
- **libwheeler_ca.so** — Compiled shared library (built via Makefile).
- **Makefile** — Build rules for hipcc compilation.

## How It Works
`gpu_dynamics.py` (in parent dir) loads `libwheeler_ca.so` via ctypes. If the .so doesn't exist or GPU isn't available, the system falls back to CPU (numpy) dynamics transparently.

## Building
```bash
cd wheeler_memory/gpu && make
# or use the install script:
bash scripts/tools/install_hip_hook.sh
```

Requires ROCm/HIP toolchain. CUDA systems need hipcc compatibility layer or a separate CUDA build.
