import numpy as np
import openvino as ov
from openvino import opset13 as ops
import os

def create_wheeler_npu_model(width=64, height=64, save_path="ai_tech_stack/models/wheeler_dynamics.xml"):
    """
    Creates an OpenVINO model for Wheeler dynamics.
    Logic: 
    1. Circular Padding (Wrap)
    2. 3x3 Mean Blur
    3. Tanh(avg * 1.5 + input * 0.5)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Input: [1, 1, height, width]
    input_node = ops.parameter([1, 1, height, width], ov.Type.f32, name="input")
    
    # Circular Padding simulation:
    # Pad H (top/bottom)
    # axis=2 is H. split_lengths=[height-1, 1] -> output(0) is first height-1 rows, output(1) is last row.
    split_h = ops.variadic_split(input_node, axis=2, split_lengths=[height-1, 1])
    top_padding = split_h.output(1) # Last row to be placed at top
    bottom_padding = ops.variadic_split(input_node, axis=2, split_lengths=[1, height-1]).output(0) # First row to be placed at bottom
    
    padded_h = ops.concat([top_padding, input_node, bottom_padding], axis=2)
    
    # Pad W (left/right)
    split_w = ops.variadic_split(padded_h, axis=3, split_lengths=[width-1, 1])
    left_padding = split_w.output(1) # Last col to be placed at left
    right_padding = ops.variadic_split(padded_h, axis=3, split_lengths=[1, width-1]).output(0) # First col to be placed at right
    
    padded_full = ops.concat([left_padding, padded_h, right_padding], axis=3)
    
    # 3x3 Mean Filter Convolution
    kernel_data = np.ones((1, 1, 3, 3), dtype=np.float32) / 9.0
    kernel = ops.constant(kernel_data, ov.Type.f32, name="kernel")
    
    # Conv with no padding (we did it manually)
    conv = ops.convolution(
        padded_full, 
        kernel, 
        strides=[1, 1], 
        dilations=[1, 1], 
        pads_begin=[0, 0], 
        pads_end=[0, 0], 
        auto_pad="explicit"
    )
    
    # Non-linearity: Tanh(conv * 1.5 + input * 0.5)
    scale_conv = ops.multiply(conv, ops.constant(1.5, ov.Type.f32))
    scale_input = ops.multiply(input_node, ops.constant(0.5, ov.Type.f32))
    add = ops.add(scale_conv, scale_input)
    output = ops.tanh(add)
    
    # Model
    model = ov.Model([output], [input_node], "WheelerDynamics")
    
    # Save
    ov.save_model(model, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    import sys
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    path = sys.argv[3] if len(sys.argv) > 3 else "ai_tech_stack/models/wheeler_dynamics.xml"
    create_wheeler_npu_model(w, h, path)
