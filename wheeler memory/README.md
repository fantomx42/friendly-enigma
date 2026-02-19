# Wheeler Memory

A cellular automata-based associative memory system where meaning is what survives symbolic pressure.

Instead of "save text, search later" (traditional RAG), Wheeler Memory evolves input through 3-state cellular automata dynamics until it converges to a stable attractor — a physical pattern that *is* the memory. Recall works by Pearson correlation between query attractors and stored attractors.

## Documentation

- **[Installation](docs/install.md)** - Getting started
- **[concepts](docs/concepts.md)** - Theory, CA dynamics, and the "It from Bit" philosophy
- **[Architecture](docs/architecture.md)** - Chunking, Bricks, Rotation, and Temperature
- **[CLI Reference](docs/cli.md)** - `wheeler-store`, `wheeler-recall`, and other tools
- **[Python API](docs/api.md)** - Integrating Wheeler Memory into your code
- **[GPU Acceleration](docs/gpu.md)** - HIP kernels for AMD GPUs (ROCm)
- **[Future Work](docs/future.md)** - Roadmap and planned features

## Quick Start

### Install

```bash
git clone https://github.com/fantomx42/friendly-enigma.git
cd "wheeler memory"
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### GPU (optional)

```bash
cd wheeler_memory/gpu && make    # requires hipcc / ROCm
```

### Usage

```bash
# Store a memory
wheeler-store "fix the python debug error"

# Recall a memory
wheeler-recall "python bug"

# GPU benchmark
wheeler-bench-gpu
```

## GPU Performance

| Batch | CPU | GPU (RX 9070 XT) | Speedup |
|-------|-----|-------------------|---------|
| 100 | 239ms | 5.5ms | **43.8×** |
| 1,000 | 2.4s | 34ms | **70.7×** |

## License

MIT
