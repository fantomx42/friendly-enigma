import os
import numpy as np
import openvino as ov
from typing import Optional, Union
from dataclasses import dataclass

@dataclass
class Trajectory:
    final_frame: np.ndarray

class NPUWheelerEngine:
    """
    OpenVINO-based engine for running Wheeler dynamics on the Intel NPU.
    """
    def __init__(self, model_path: str = "ai_tech_stack/models/wheeler_dynamics.xml", device: str = "NPU"):
        self.device = device
        self.core = ov.Core()
        
        if not os.path.exists(model_path):
            # Try to find or create if missing? No, we should have it.
            raise FileNotFoundError(f"Wheeler NPU model not found at {model_path}.")
        
        print(f"[NPUWheelerEngine] Loading model from {model_path}...")
        self.model = self.core.read_model(model_path)
        
        # Get input shape
        self.input_shape = self.model.input().shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]
        
        print(f"[NPUWheelerEngine] Compiling model for {device} ({self.width}x{self.height})...")
        try:
            self.compiled_model = self.core.compile_model(self.model, device)
            self.infer_request = self.compiled_model.create_infer_request()
        except Exception as e:
            print(f"[NPUWheelerEngine] Failed to compile for {device}: {e}. Falling back to CPU.")
            self.device = "CPU"
            self.compiled_model = self.core.compile_model(self.model, "CPU")
            self.infer_request = self.compiled_model.create_infer_request()

    def run_tick(self, frame: np.ndarray) -> np.ndarray:
        """
        Runs a single tick of Wheeler dynamics on the NPU.
        """
        # Reshape/Crop/Pad to match model input shape if necessary
        # For now assume exact match or handled by caller
        if frame.ndim == 2:
            input_data = frame[np.newaxis, np.newaxis, :, :]
        else:
            input_data = frame
            
        results = self.compiled_model([input_data])
        output = results[0]
        
        return output.reshape(frame.shape[-2:])

    def run_dynamics(self, frame: np.ndarray, max_ticks: int = 50) -> Trajectory:
        """
        Runs multiple ticks of Wheeler dynamics. Returns a Trajectory-like object.
        """
        current_frame = frame
        for _ in range(max_ticks):
            current_frame = self.run_tick(current_frame)
        return Trajectory(final_frame=current_frame)
