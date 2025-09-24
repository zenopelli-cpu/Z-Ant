import numpy as np
import random
from onnx import helper, TensorProto

def generate_qlinearconv_model(input_names, output_names):
    """
    Generates a QLinearConv operator model.
    
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
    
    # Input tensor dimensions [N, C, H, W]
    batch_size = 1
    in_channels = random.randint(1, 4)
    input_height = random.randint(10, 20)
    input_width = random.randint(10, 20)
    
    # Filter dimensions [out_channels, in_channels/group, kernel_h, kernel_w]
    out_channels = random.randint(1, 4)
    kernel_size = random.randint(3, 5)
    group = 1  # Keep simple for now
    
    input_shape = [batch_size, in_channels, input_height, input_width]
    weight_shape = [out_channels, in_channels // group, kernel_size, kernel_size]
    bias_shape = [out_channels]
    
    # Calculate output dimensions correctly
    pad = kernel_size // 2  # Same padding
    stride = 1
    dilation = 1
    
    # Correct output size calculation for convolution with padding
    output_height = ((input_height + 2 * pad - dilation * (kernel_size - 1) - 1) // stride) + 1
    output_width = ((input_width + 2 * pad - dilation * (kernel_size - 1) - 1) // stride) + 1
    output_shape = [batch_size, out_channels, output_height, output_width]
    
    # Generate quantization parameters with better ranges
    x_scale = np.float32(np.random.uniform(0.001, 0.1))
    x_zero_point = np.uint8(random.randint(0, 255))
    
    w_scale = np.float32(np.random.uniform(0.001, 0.1))
    w_zero_point = np.uint8(128)  # Common choice for weights
    
    y_scale = np.float32(np.random.uniform(0.001, 0.1))
    y_zero_point = np.uint8(random.randint(0, 255))
    
    # Generate quantized weight data (uint8)
    w_data = np.random.randint(0, 256, size=weight_shape, dtype=np.uint8)
    
    # Generate bias (int32) - should be properly scaled
    bias_data = np.random.randint(-1000, 1000, size=bias_shape, dtype=np.int32)
    
    # Create unique names for initializers to avoid conflicts
    param_id = random.randint(1000, 9999)
    
    # Create initializers for all constant parameters
    x_scale_name = f"x_scale_{param_id}"
    x_zero_point_name = f"x_zero_point_{param_id}"
    w_name = f"weight_{param_id}"
    w_scale_name = f"w_scale_{param_id}"
    w_zero_point_name = f"w_zero_point_{param_id}"
    y_scale_name = f"y_scale_{param_id}"
    y_zero_point_name = f"y_zero_point_{param_id}"
    bias_name = f"bias_{param_id}"
    
    # Add initializers
    initializers.append(helper.make_tensor(x_scale_name, TensorProto.FLOAT, [], [x_scale]))
    initializers.append(helper.make_tensor(x_zero_point_name, TensorProto.UINT8, [], [x_zero_point]))
    initializers.append(helper.make_tensor(w_name, TensorProto.UINT8, weight_shape, w_data.flatten().tolist()))
    initializers.append(helper.make_tensor(w_scale_name, TensorProto.FLOAT, [], [w_scale]))
    initializers.append(helper.make_tensor(w_zero_point_name, TensorProto.UINT8, [], [w_zero_point]))
    initializers.append(helper.make_tensor(y_scale_name, TensorProto.FLOAT, [], [y_scale]))
    initializers.append(helper.make_tensor(y_zero_point_name, TensorProto.UINT8, [], [y_zero_point]))
    initializers.append(helper.make_tensor(bias_name, TensorProto.INT32, bias_shape, bias_data.flatten().tolist()))
    
    # Create input info (only for the actual dynamic input)
    input_info = helper.make_tensor_value_info(input_names[0], TensorProto.UINT8, input_shape)
    
    # Create output info (quantized uint8)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.UINT8, output_shape)
    
    # Create QLinearConv node with proper input order
    node = helper.make_node(
        "QLinearConv",
        inputs=[
            input_names[0],      # x (input data - dynamic)
            x_scale_name,        # x_scale (constant)
            x_zero_point_name,   # x_zero_point (constant)
            w_name,              # w (weight - constant)
            w_scale_name,        # w_scale (constant)
            w_zero_point_name,   # w_zero_point (constant)
            y_scale_name,        # y_scale (constant)
            y_zero_point_name,   # y_zero_point (constant)
            bias_name            # bias (constant)
        ],
        outputs=[output_names[0]],
        name=f"QLinearConv_node_{param_id}",
        dilations=[dilation, dilation],
        group=group,
        kernel_shape=[kernel_size, kernel_size],
        pads=[pad, pad, pad, pad],  # [top, left, bottom, right]
        strides=[stride, stride]
    )
    
    metadata = {
        "input_shapes": [input_shape],
        "weight_shape": weight_shape,
        "bias_shape": bias_shape,
        "output_shapes": [output_shape],
        "x_scale": float(x_scale),
        "x_zero_point": int(x_zero_point),
        "w_scale": float(w_scale),
        "w_zero_point": int(w_zero_point),
        "y_scale": float(y_scale),
        "y_zero_point": int(y_zero_point),
        "kernel_size": kernel_size,
        "padding": pad,
        "stride": stride,
        "dilation": dilation,
        "group": group,
        "param_names": {
            "x_scale": x_scale_name,
            "x_zero_point": x_zero_point_name,
            "weight": w_name,
            "w_scale": w_scale_name,
            "w_zero_point": w_zero_point_name,
            "y_scale": y_scale_name,
            "y_zero_point": y_zero_point_name,
            "bias": bias_name
        }
    }
    
    return [input_info], output_info, [node], initializers, metadata