# wheeler_memory/npu/ — Neural Processing Unit Integration

Scaffolding for hardware inference accelerators. Two targets:

## Intel NPU (current hardware)

Present in the Core Ultra 7 265K as an integrated PCIe device.

### Hardware Specs (from Intel datasheet Section 10.0)
- 2 NCE (Neural Compute Engine) tiles
- Each tile: 2048 MACs (512 MPEs x 4 MACs each) + 2 ACT-SHAVE DSPs
- Total: 4K MACs, 8 TOPS INT8, 4 TFLOPS FP16
- 4MB CMX (Connection MatriX) near-compute SRAM
- Managed by LeonRT microcontroller

### Software Stack
- **OpenVINO** toolkit is the primary access path
- Workflow: Train model → Export ONNX → OpenVINO IR → Compile for NPU → Run
- Python API: `openvino.runtime.Core` → compile_model → infer
- INT8 quantization via NNCF (Neural Network Compression Framework)

### Linux Driver Status (as of 2026-04)
- Package: `intel-npu-driver` (AUR on Arch/CachyOS)
- Status: Functional but ecosystem is young for custom workloads
- Check: `cat /sys/class/accel/accel0/device/device_type` should show NPU
- OpenVINO device check: `"NPU" in Core().available_devices`

### Wheeler Memory Use Cases
1. **Cortex L3 classifier** — Small model, INT8-friendly, low latency
   - Currently pure numpy in cortex_classifier.py
   - Could export as ONNX → quantize INT8 → run on NPU
   - Frees CPU/GPU for encoding + CA evolution
2. **Embedding inference** — If using sentence-transformers (MiniLM)
   - ONNX export of MiniLM → OpenVINO INT8 → NPU
   - Currently runs on CPU; NPU would free cores

### Install (Arch/CachyOS)
```bash
# NPU driver
yay -S intel-npu-driver

# OpenVINO
pip install openvino openvino-dev nncf

# Verify
python -c "from openvino.runtime import Core; print(Core().available_devices)"
# Should include 'NPU'
```

---

## Google Coral Edge TPU (future hardware)

Dual M.2 Coral TPU for INT8 inference. Not yet purchased.

### Planned Hardware
- 2x Google Coral M.2 A+E key modules (PCIe)
- Each: 4 TOPS INT8, ~2W TDP
- Total: 8 TOPS dual-chip
- PyCoral / libedgetpu API

### Use Case Vision
- Model parallelism: Split cortex classifier across 2 TPUs
- Or batch split: Each TPU handles half the inference batch
- Ultra-low-power always-on inference while GPU does heavy compute

### Why Dual Coral
- Single Coral = 4 TOPS, competitive with Intel NPU per-tile
- Dual = 8 TOPS with independent scheduling
- PCIe M.2 = no USB bottleneck
- Original idea: offload inference completely from CPU/GPU

See `coral/CONTEXT.md` for hardware-specific notes when purchased.

---

## Architecture Vision

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  CPU (265K)  │     │  GPU (9070XT) │     │  NPU / Coral  │
│              │     │              │     │              │
│ Orchestrate  │────▶│ Encode       │     │ Classify     │
│ I/O          │     │ CA evolve    │     │ Score        │
│ Graph (L1)   │     │ Similarity   │     │ Embed (INT8) │
│              │     │ Cortex (L2)  │     │              │
└─────────────┘     └──────────────┘     └──────────────┘
     │                    ▲                    ▲
     └────────────────────┴────────────────────┘
              PCIe bus / shared memory
```

All three accelerators active simultaneously. CPU orchestrates, GPU does
the heavy float32 math, NPU/Coral handles low-precision inference.
