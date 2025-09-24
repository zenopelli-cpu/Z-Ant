import numpy as np
import random
from onnx import helper, TensorProto


def generate_dequantizelinear_model(input_names, output_names):
    """
    Generates a DequantizeLinear operator model.
    """
    initializers = []
    
    mode = "per_tensor"
    # Randomly pick input shape
    shape = [random.randint(1, 4) for _ in range(3)]  # 3D tensor

    # Select quantized input type
    dtype = random.choice([
        TensorProto.UINT8, TensorProto.INT8,
        # TensorProto.UINT16, TensorProto.INT16,
        # TensorProto.UINT4, TensorProto.INT4,  # Uncomment if supported
    ])

    # Determine scale shape (per-tensor or per-axis)
    if random.choice([True, False]):
        # Per-tensor
        y_scale = np.array([round(random.uniform(0.01, 1.0), 4)], dtype=np.float32)
        zp_shape = y_scale.shape
        axis = None
    else:
        # Per-axis
        axis = 1  # Fixed axis for example
        length = shape[axis]
        y_scale = (np.random.rand(length) * 0.5 + 0.1).astype(np.float32)
        zp_shape = y_scale.shape

    # Generate quantized input `x` and zero point
    if dtype in (TensorProto.UINT4, TensorProto.INT4):  # Optional 4-bit support
        max_val = 2**4 - 1 if dtype == TensorProto.UINT4 else 2**3 - 1
        min_val = 0 if dtype == TensorProto.UINT4 else -2**3
        x_data = np.random.randint(min_val, max_val + 1, size=shape, dtype=np.int32)
        y_zero_point = np.random.randint(min_val, max_val + 1, size=zp_shape, dtype=np.int32)
    else:
        info = np.iinfo({
            TensorProto.UINT8: np.uint8, TensorProto.INT8: np.int8,
            TensorProto.UINT16: np.uint16, TensorProto.INT16: np.int16
        }[dtype])
        x_data = np.random.randint(info.min, info.max + 1, size=shape, dtype=info.dtype)
        y_zero_point = np.random.randint(info.min, info.max + 1, size=zp_shape, dtype=info.dtype)

    # Names
    scale_name = input_names[1]
    zp_name = input_names[2]

    # Initializers
    initializers.append(helper.make_tensor(scale_name, TensorProto.FLOAT, list(zp_shape), y_scale.flatten().tolist()))
    initializers.append(helper.make_tensor(zp_name, dtype, list(zp_shape), y_zero_point.flatten().tolist()))
    initializers.append(helper.make_tensor(input_names[0], dtype, shape, x_data.flatten().tolist()))

    # Create output metadata (always float32)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

    node = helper.make_node(
        "DequantizeLinear",
        inputs=[input_names[0], scale_name, zp_name],
        outputs=[output_names[0]],
        axis=axis if axis is not None else None,
        name=f"DequantizeLinear_node_dtype{dtype}_axis{axis}"
    )

    # Dummy input info
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)

    metadata = {
        "input_shapes": [shape],
        "scale_shape": list(zp_shape),
        "dtype": dtype,
        "axis": axis
    }

    return [input_info], output_info, [node], initializers, metadata