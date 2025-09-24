import numpy as np
import random
from onnx import helper, TensorProto


def generate_convinteger_model(input_names, output_names):
    """
    Generates a ConvInteger operator model.
    """
    initializers = []
    
    # ConvInteger: convolution with integer arithmetic on quantized tensors
    # Inputs: x (quantized), w (quantized), x_zero_point (optional), w_zero_point (optional)
    
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
    
    # Calculate output dimensions
    pad = kernel_size // 2  # Same padding
    output_height = input_height  # Same padding
    output_width = input_width
    output_shape = [batch_size, out_channels, output_height, output_width]
    
    # Generate quantized input data (uint8)
    x_data = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
    x_zero_point = np.random.randint(0, 256, dtype=np.uint8)
    
    # Generate quantized weight data (int8)
    w_data = np.random.randint(-128, 128, size=weight_shape, dtype=np.int8)
    w_zero_point = np.int8(0)  # Zero point for int8 weights
    
    # Create initializers for quantized inputs
    initializers.append(helper.make_tensor(input_names[0], TensorProto.UINT8, input_shape, x_data.flatten().tolist()))
    initializers.append(helper.make_tensor(input_names[1], TensorProto.INT8, weight_shape, w_data.flatten().tolist()))
    initializers.append(helper.make_tensor(input_names[2], TensorProto.UINT8, [], [x_zero_point]))
    initializers.append(helper.make_tensor(input_names[3], TensorProto.INT8, [], [w_zero_point]))
    
    # Create output info (int32 - ConvInteger outputs accumulated results)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.INT32, output_shape)
    
    # Create ConvInteger node
    node = helper.make_node(
        "ConvInteger",
        inputs=[input_names[0], input_names[1], input_names[2], input_names[3]],  # x, w, x_zero_point, w_zero_point
        outputs=[output_names[0]],
        name=f"ConvInteger_node",
        dilations=[1, 1],
        group=group,
        kernel_shape=[kernel_size, kernel_size],
        pads=[pad, pad, pad, pad],
        strides=[1, 1]
    )
    
    # Dummy input info
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
    
    metadata = {
        "input_shapes": [input_shape, weight_shape],
        "output_shapes": [output_shape],
        "x_zero_point": int(x_zero_point),
        "w_zero_point": int(w_zero_point),
        "kernel_size": kernel_size,
        "group": group,
        "output_type": "int32"
    }
    
    return [input_info], output_info, [node], initializers, metadata