# wheeler_memory/npu/coral/ — Google Coral Edge TPU

**Status: Future hardware — not yet purchased or installed.**

## Planned Hardware

- 2x Coral M.2 Accelerator (A+E key, PCIe interface)
- Each module: Google Edge TPU ASIC, 4 TOPS INT8, ~2W TDP
- Form factor: M.2 2230 A+E key
- Interface: PCIe Gen2 x1 (each)

## Software Stack

- **PyCoral**: Python API for Edge TPU inference
- **libedgetpu**: Low-level C++ runtime
- **TensorFlow Lite**: Model format (`.tflite` with Edge TPU compilation)
- Workflow: Train → Export TFLite → Edge TPU compile → PyCoral infer

## Dual-TPU Architecture

Two independent TPUs enable:

1. **Model parallelism**: Split a model across both TPUs (pipeline stages)
2. **Data parallelism**: Each TPU processes half the inference batch
3. **Task parallelism**: TPU-1 handles classifier, TPU-2 handles embeddings

PyCoral supports addressing individual TPUs by device path.

## Install (when hardware arrives)

```bash
# Arch/CachyOS
yay -S libedgetpu pycoral

# Or pip
pip install pycoral tflite-runtime

# Verify
python -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())"
```

## Integration Points

- `coral/tpu_bridge.py` — PyCoral inference wrapper (stub)
- Maps to same `classify()` interface as `openvino_bridge.py`
- Device routing in `npu/__init__.py` picks best available backend
