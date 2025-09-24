import numpy as np
import random
from onnx import helper, TensorProto


def generate_quantizelinear_model(input_names, output_names):
    """
    Generates a QuantizeLinear operator model.
    """
    initializers = []
    
    # Randomly pick input shape
    N = 1
    C = random.randint(1,4)
    H = random.randint(10,50)
    W = random.randint(10,50)
    shape = [N, C, H, W]
    data = np.random.randn(*shape).astype(np.float32) # Scale data to have values outside typical clip range
    
    # Randomly choose quantization mode
    mode_choice = random.choice(["per_tensor", "per_axis"])

    if mode_choice == "per_tensor":
        # Per-tensor quantization
        mode = "per_tensor"
        y_scale = np.array([round(random.uniform(0.01, 1.0), 4)], dtype=np.float32)
        axis = 0
        bl_size = 0
        
    else:  # per_axis
        # Per-axis quantization
        mode = "per_axis"
        axis = random.randint(0, len(shape) - 1)  # Pick a random axis
        length = shape[axis]
        y_scale = np.random.rand(length).astype(np.float32) * 0.5 + 0.1
        bl_size = 0
    
    # Pick a valid ONNX-compatible dtype for y_zero_point
    valid_dtypes = [TensorProto.UINT8]
    dtype = random.choice(valid_dtypes)
    
    # Get zero point shape (same as scale shape)
    zp_shape = y_scale.shape
    
    # Map ONNX dtype to numpy dtype
    dtype_np = {TensorProto.UINT8: np.uint8}[dtype]
    
    # Generate y_zero_point with matching shape and dtype
    info = np.iinfo(dtype_np)
    scale_zp = np.random.randint(info.min, info.max + 1, size=zp_shape, dtype=dtype_np)
    
    # Assign names
    scale_name = input_names[1]
    zp_name = input_names[2]
    
    # Add scale and zero point initializers
    initializers.append(helper.make_tensor(
        scale_name, TensorProto.FLOAT, list(zp_shape), y_scale.flatten().tolist()))
    initializers.append(helper.make_tensor(
        zp_name, dtype, list(zp_shape), scale_zp.flatten().tolist()))
    
    # Prepare main input tensor (x)
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    initializers.append(init_tensor)
     
    # Create output metadata
    output_info = helper.make_tensor_value_info(output_names[0], dtype, shape)
    
    node = helper.make_node("QuantizeLinear", inputs=[input_names[0], scale_name, zp_name], outputs=[output_names[0]],
                            axis=axis, 
                            name=f"QuantizeLinear_node_ax{axis}_bl{bl_size}")
    
    # Dummy input_info, useful for the rest of the pipeline
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    
    # Metadata for codegen
    metadata = {
        "input_shapes": [shape],
        "scale_shape": list(zp_shape),
        "dtype": dtype,
        "axis": axis,
        "block_size": bl_size,
        "mode": mode
    }
    
    return [input_info], output_info, [node], initializers, metadata