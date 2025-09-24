import numpy as np
import random
from onnx import helper, TensorProto

import numpy as np
import random
from onnx import helper, TensorProto


def calculate_output_size(input_size, kernel_size, stride, pad_before, pad_after, ceil_mode=0):
    """Calculate output size for pooling operation."""
    if ceil_mode:
        return int(np.ceil((input_size + pad_before + pad_after - kernel_size) / stride)) + 1
    else:
        return int(np.floor((input_size + pad_before + pad_after - kernel_size) / stride)) + 1


def generate_maxpool_model(input_names, output_names):
    """
    Generates a MaxPool operator model.
    
    Args:
        input_names: List of input names (expects at least 1 for the input data)
        output_names: List of output names (expects at least 1)
    
    Returns:
        tuple: (input_infos, output_info, nodes, initializers, metadata)
    """
    if len(input_names) < 1:
        raise ValueError("input_names must contain at least 1 element: [data_input]")
    if len(output_names) < 1:
        raise ValueError("output_names must contain at least 1 element")
    
    initializers = []
    
    # Random values for batch and channel dimensions
    N = random.randint(1, 3)  # Batch size
    C = random.randint(1, 4)  # Number of channels
    
    # Support both 2D and 3D pooling
    spatial_type = random.choice(["2D", "3D"])
    
    if spatial_type == "2D":
        H = random.randint(6, 16)  # Height - increased minimum for stability
        W = random.randint(6, 16)  # Width
        input_shape = [N, C, H, W]
        spatial_dims = [H, W]
        
        # Kernel shape for 2D pooling - more conservative
        kernel_h = random.randint(2, min(3, H // 2))
        kernel_w = random.randint(2, min(3, W // 2))
        kernel_shape = [kernel_h, kernel_w]
        
    else:  # 3D pooling
        D = random.randint(4, 8)   # Depth
        H = random.randint(4, 8)   # Height  
        W = random.randint(4, 8)   # Width
        input_shape = [N, C, D, H, W]
        spatial_dims = [D, H, W]
        
        # Kernel shape for 3D pooling - more conservative
        kernel_d = random.randint(2, min(3, D // 2))
        kernel_h = random.randint(2, min(3, H // 2))
        kernel_w = random.randint(2, min(3, W // 2))
        kernel_shape = [kernel_d, kernel_h, kernel_w]
    
    # Random stride parameters (must be positive)
    if spatial_type == "2D":
        strides = [random.randint(1, min(2, kernel_h)), random.randint(1, min(2, kernel_w))]
    else:
        strides = [random.randint(1, min(2, kernel_d)), 
                  random.randint(1, min(2, kernel_h)), 
                  random.randint(1, min(2, kernel_w))]
    
    # Use simpler padding strategy to avoid shape calculation errors
    auto_pad_options = ["NOTSET", "VALID"]  # Removed SAME_* for now due to complexity
    auto_pad = random.choice(auto_pad_options)
    
    # Calculate padding and output dimensions
    ceil_mode = random.choice([0, 1])
    
    if auto_pad == "NOTSET":
        # Conservative explicit padding
        if spatial_type == "2D":
            pad_h = random.randint(0, 1)  # Keep padding small
            pad_w = random.randint(0, 1)
            pads = [pad_h, pad_w, pad_h, pad_w]  # [top, left, bottom, right]
            
            # Calculate output dimensions
            H_out = calculate_output_size(H, kernel_shape[0], strides[0], pads[0], pads[2], ceil_mode)
            W_out = calculate_output_size(W, kernel_shape[1], strides[1], pads[1], pads[3], ceil_mode)
            output_shape = [N, C, H_out, W_out]
            
        else:  # 3D
            pad_d = random.randint(0, 1)
            pad_h = random.randint(0, 1)
            pad_w = random.randint(0, 1)
            pads = [pad_d, pad_h, pad_w, pad_d, pad_h, pad_w]  # [front, top, left, back, bottom, right]
            
            # Calculate output dimensions
            D_out = calculate_output_size(D, kernel_shape[0], strides[0], pads[0], pads[3], ceil_mode)
            H_out = calculate_output_size(H, kernel_shape[1], strides[1], pads[1], pads[4], ceil_mode)
            W_out = calculate_output_size(W, kernel_shape[2], strides[2], pads[2], pads[5], ceil_mode)
            output_shape = [N, C, D_out, H_out, W_out]
            
    else:  # VALID - no padding
        pads = [0] * (len(spatial_dims) * 2)
        
        if spatial_type == "2D":
            H_out = calculate_output_size(H, kernel_shape[0], strides[0], 0, 0, ceil_mode)
            W_out = calculate_output_size(W, kernel_shape[1], strides[1], 0, 0, ceil_mode)
            output_shape = [N, C, H_out, W_out]
        else:  # 3D
            D_out = calculate_output_size(D, kernel_shape[0], strides[0], 0, 0, ceil_mode)
            H_out = calculate_output_size(H, kernel_shape[1], strides[1], 0, 0, ceil_mode)
            W_out = calculate_output_size(W, kernel_shape[2], strides[2], 0, 0, ceil_mode)
            output_shape = [N, C, D_out, H_out, W_out]
    
    # Validate output dimensions are positive
    if any(dim <= 0 for dim in output_shape[2:]):  # Check spatial dimensions
        # Fallback to guaranteed working configuration
        if spatial_type == "2D":
            kernel_shape = [2, 2]
            strides = [1, 1]
            auto_pad = "VALID"
            pads = [0, 0, 0, 0]
            ceil_mode = 0
            H_out = calculate_output_size(H, 2, 1, 0, 0, 0)
            W_out = calculate_output_size(W, 2, 1, 0, 0, 0)
            output_shape = [N, C, max(1, H_out), max(1, W_out)]
        else:  # 3D
            kernel_shape = [2, 2, 2]
            strides = [1, 1, 1]
            auto_pad = "VALID"
            pads = [0, 0, 0, 0, 0, 0]
            ceil_mode = 0
            D_out = calculate_output_size(D, 2, 1, 0, 0, 0)
            H_out = calculate_output_size(H, 2, 1, 0, 0, 0)
            W_out = calculate_output_size(W, 2, 1, 0, 0, 0)
            output_shape = [N, C, max(1, D_out), max(1, H_out), max(1, W_out)]
    
    # Only create input info for the actual dynamic input (no initializers for MaxPool)
    input_info = helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, input_shape)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
    
    # Optional parameters
    storage_order = random.choice([0, 1])  # 0 = row major, 1 = column major
    dilations = [1] * len(kernel_shape)  # MaxPool typically uses dilations = 1
    
    # Create the MaxPool node with all parameters
    node_attrs = {
        "kernel_shape": [int(k) for k in kernel_shape],
        "strides": [int(s) for s in strides],
        "ceil_mode": int(ceil_mode),
        "storage_order": int(storage_order)
    }
    
    # Add pads if not all zeros or if using explicit padding
    if auto_pad == "NOTSET" or any(p != 0 for p in pads):
        node_attrs["pads"] = [int(p) for p in pads]
    
    # Add auto_pad if not NOTSET
    if auto_pad != "NOTSET":
        node_attrs["auto_pad"] = auto_pad
    
    # Add dilations if not all 1s (though MaxPool rarely uses dilations != 1)
    if any(d != 1 for d in dilations):
        node_attrs["dilations"] = [int(d) for d in dilations]
    
    # Create unique node name
    param_id = random.randint(1000, 9999)
    node_name = f"MaxPool_node_{spatial_type}_{param_id}"
    
    node = helper.make_node(
        "MaxPool",
        inputs=[input_names[0]],
        outputs=[output_names[0]],
        name=node_name,
        **node_attrs
    )
    
    metadata = {
        "input_shapes": [input_shape],
        "output_shapes": [output_shape],
        "spatial_type": spatial_type,
        "batch_size": int(N),
        "channels": int(C),
        "spatial_dimensions": [int(d) for d in spatial_dims],
        "kernel_shape": [int(k) for k in kernel_shape],
        "strides": [int(s) for s in strides],
        "pads": [int(p) for p in pads],
        "auto_pad": auto_pad,
        "ceil_mode": int(ceil_mode),
        "storage_order": int(storage_order),
        "dilations": [int(d) for d in dilations],
        "node_name": node_name
    }
    
    return [input_info], output_info, [node], initializers, metadata
