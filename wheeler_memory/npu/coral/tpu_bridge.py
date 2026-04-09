"""Google Coral Edge TPU bridge (future).

Stub module for dual Coral M.2 TPU inference. Provides the same
classify() interface as openvino_bridge.py for device-agnostic routing.

Not yet implemented — hardware not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class CoralClassifier:
    """Run a classifier on Google Coral Edge TPU.

    Stub — raises NotImplementedError until hardware is available.
    """

    def __init__(self, tflite_path: str, device_index: int = 0):
        self.tflite_path = tflite_path
        self.device_index = device_index

    def load(self) -> None:
        """Load compiled TFLite model onto Edge TPU."""
        raise NotImplementedError(
            "Coral TPU bridge not yet implemented. "
            "Requires: Coral M.2 hardware + pip install pycoral"
        )

    def classify(self, features: np.ndarray) -> np.ndarray:
        """Run INT8 inference on Edge TPU."""
        raise NotImplementedError("Coral TPU bridge not yet implemented.")


class DualCoralPipeline:
    """Coordinate inference across two Coral Edge TPUs.

    Supports data-parallel (batch split) and task-parallel (model split) modes.
    Stub — raises NotImplementedError until hardware is available.
    """

    def __init__(self, tflite_path: str, mode: str = "data_parallel"):
        self.tflite_path = tflite_path
        self.mode = mode  # "data_parallel" or "model_parallel"

    def load(self) -> None:
        raise NotImplementedError("Dual Coral pipeline not yet implemented.")

    def classify_batch(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Dual Coral pipeline not yet implemented.")
