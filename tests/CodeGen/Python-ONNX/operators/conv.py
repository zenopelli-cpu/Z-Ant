import numpy as np
import random
from onnx import helper, TensorProto


def generate_conv_model(input_names, output_names):
    """
    Generates a Conv operator model.
    """
    initializers = []
    
    # Operatore Conv: genera input e pesi come initializer
    N = 1
    C = random.randint(1,4)
    H = random.randint(10,50)
    W = random.randint(10,50)
    input_shape = [N, C, H, W]
    data = np.random.randn(*input_shape).astype(np.float32)
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, input_shape, data.flatten().tolist())
    initializers.append(init_tensor)
    kH = random.randint(2, max(2, H//2))
    kW = random.randint(2, max(2, W//2))
    kernel_shape = [kH, kW]
    M = random.randint(1,4)
    weight_shape = [M, C, kH, kW]
    weight_data = np.random.randn(*weight_shape).astype(np.float32)
    init_weight = helper.make_tensor(input_names[1], TensorProto.FLOAT, weight_shape, weight_data.flatten().tolist())
    initializers.append(init_weight)
    
    # Add strides parameter
    strides = [random.randint(1, 3), random.randint(1, 3)]
    
    # Add dilations parameter
    dilations = [random.randint(1, 2), random.randint(1, 2)]
    
    # Add padding parameter (padding for begin and end of each spatial axis)
    # Format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    pad_h = random.randint(0, 2)
    pad_w = random.randint(0, 2)
    pads = [0, 0, pad_h, pad_w, 0, 0, pad_h, pad_w]  # Padding for 4D tensor [N,C,H,W]
    
    # Calculate output dimensions with strides, dilations and padding
    # Formula: output_size = (input_size + pad_begin + pad_end - (kernel_size-1)*dilation - 1) / stride + 1
    H_out = (H + 2*pad_h - (kH-1)*dilations[0] - 1) // strides[0] + 1
    W_out = (W + 2*pad_w - (kW-1)*dilations[1] - 1) // strides[1] + 1
    
    # Ensure valid output dimensions
    if H_out <= 0 or W_out <= 0:
        # Fallback to simpler case if dimensions don't work out
        dilations = [1, 1]
        pad_h = pad_w = 1
        pads = [0, 0, pad_h, pad_w, 0, 0, pad_h, pad_w]
        H_out = (H + 2*pad_h - kH) // strides[0] + 1
        W_out = (W + 2*pad_w - kW) // strides[1] + 1
    
    output_shape = [N, M, H_out, W_out]
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
    
    # Use pads attribute in the node (simplified to just the 4 values for H,W dimensions)
    node = helper.make_node("Conv", inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                            kernel_shape=kernel_shape, strides=strides, dilations=dilations,
                            pads=[pad_h, pad_w, pad_h, pad_w],
                            name=f"Conv_node_k{kernel_shape}_s{strides}_d{dilations}_p{[pad_h, pad_w]}")
    
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
    metadata = {"input_shapes": [input_shape, weight_shape], "output_shapes": [output_shape],
                "kernel_shape": kernel_shape, "strides": strides, "dilations": dilations, 
                "pads": [pad_h, pad_w, pad_h, pad_w]}
    return [input_info], output_info, [node], initializers, metadata