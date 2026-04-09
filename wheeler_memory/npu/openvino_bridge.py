"""OpenVINO bridge for Intel NPU inference.

Stub module — provides the interface for running INT8/FP16 models on
the Intel NPU integrated in Core Ultra processors. Actual implementation
depends on OpenVINO installation and NPU driver availability.

Usage (when implemented):
    from wheeler_memory.npu.openvino_bridge import NPUClassifier
    clf = NPUClassifier("cortex_model.onnx")
    scores = clf.classify(features)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class NPUClassifier:
    """Run a classifier model on Intel NPU via OpenVINO.

    Stub — raises NotImplementedError until model export pipeline is built.
    """

    def __init__(self, model_path: str, device: str = "NPU"):
        self.model_path = model_path
        self.device = device
        self._compiled = None

    def load(self) -> None:
        """Compile and load model onto NPU."""
        raise NotImplementedError(
            "NPU classifier not yet implemented. "
            "Requires: pip install openvino nncf, "
            "exported ONNX model, and intel-npu-driver."
        )

    def classify(self, features: np.ndarray) -> np.ndarray:
        """Run inference on NPU. Returns score array."""
        raise NotImplementedError("NPU classifier not yet implemented.")

    def benchmark(self, n_iterations: int = 100) -> dict:
        """Benchmark NPU throughput."""
        raise NotImplementedError("NPU classifier not yet implemented.")


def quantize_model(onnx_path: str, output_path: str, calibration_data=None) -> str:
    """Quantize an ONNX model to INT8 for NPU deployment.

    Uses OpenVINO NNCF for post-training quantization.

    Stub — raises NotImplementedError until quantization pipeline is built.
    """
    raise NotImplementedError(
        "Model quantization not yet implemented. "
        "Requires: pip install openvino-dev nncf"
    )
