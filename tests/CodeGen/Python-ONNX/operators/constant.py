import numpy as np
import random
from onnx import helper, TensorProto


def generate_constant_model(input_names, output_names):
    """
    Generates a Constant operator model.
    """
        # Constant operator: generates a tensor with constant values
    # Choose random data type
    data_types = [
        (TensorProto.FLOAT, np.float32),
        (TensorProto.INT32, np.int32),
        (TensorProto.INT64, np.int64),
        (TensorProto.BOOL, bool)
    ]
    tensor_proto_type, numpy_type = random.choice(data_types)

    # Choose random shape (can be scalar, 1D, 2D, etc.)
    shape_type = random.choice(["scalar", "1d", "2d", "3d"])

    if shape_type == "scalar":
        shape = []
        if tensor_proto_type == TensorProto.FLOAT:
            value = [random.uniform(-10.0, 10.0)]
        elif tensor_proto_type == TensorProto.INT32:
            value = [random.randint(-100, 100)]
        elif tensor_proto_type == TensorProto.INT64:
            value = [random.randint(-100, 100)]
        else:  # BOOL
            value = [random.choice([True, False])]
    elif shape_type == "1d":
        dim_size = random.randint(2, 8)
        shape = [dim_size]
        if tensor_proto_type == TensorProto.FLOAT:
            value = np.random.uniform(-10.0, 10.0, size=shape).astype(numpy_type).flatten().tolist()
        elif tensor_proto_type == TensorProto.INT32:
            value = np.random.randint(-100, 100, size=shape).astype(numpy_type).flatten().tolist()
        elif tensor_proto_type == TensorProto.INT64:
            value = np.random.randint(-100, 100, size=shape).astype(numpy_type).flatten().tolist()
        else:  # BOOL
            value = np.random.choice([True, False], size=shape).astype(numpy_type).flatten().tolist()
    elif shape_type == "2d":
        shape = [random.randint(2, 5), random.randint(2, 5)]
        if tensor_proto_type == TensorProto.FLOAT:
            value = np.random.uniform(-10.0, 10.0, size=shape).astype(numpy_type).flatten().tolist()
        elif tensor_proto_type == TensorProto.INT32:
            value = np.random.randint(-100, 100, size=shape).astype(numpy_type).flatten().tolist()
        elif tensor_proto_type == TensorProto.INT64:
            value = np.random.randint(-100, 100, size=shape).astype(numpy_type).flatten().tolist()
        else:  # BOOL
            value = np.random.choice([True, False], size=shape).astype(numpy_type).flatten().tolist()
    else:  # "3d"
        shape = [random.randint(1, 3), random.randint(2, 4), random.randint(2, 4)]
        if tensor_proto_type == TensorProto.FLOAT:
            value = np.random.uniform(-10.0, 10.0, size=shape).astype(numpy_type).flatten().tolist()
        elif tensor_proto_type == TensorProto.INT32:
            value = np.random.randint(-100, 100, size=shape).astype(numpy_type).flatten().tolist()
        elif tensor_proto_type == TensorProto.INT64:
            value = np.random.randint(-100, 100, size=shape).astype(numpy_type).flatten().tolist()
        else:  # BOOL
            value = np.random.choice([True, False], size=shape).astype(numpy_type).flatten().tolist()

    # Create the constant tensor
    constant_tensor = helper.make_tensor(
        name="constant_value",
        data_type=tensor_proto_type,
        dims=shape,
        vals=value
    )

    # Output info
    output_info = helper.make_tensor_value_info(output_names[0], tensor_proto_type, shape)

    # Input info (dummy, since Constant has no inputs)
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, [1])

    # Metadata
    metadata = {
        "input_shapes": [],  # No inputs for Constant
        "output_shapes": [shape],
        "data_type": tensor_proto_type,
        "shape_type": shape_type,
        "value_sample": value[:5] if isinstance(value, list) and len(value) > 5 else value  # First 5 values for debug
    }

    # Create the Constant node with the value attribute
    node = helper.make_node(
        "Constant",
        inputs=[],  # Constant has no inputs
        outputs=[output_names[0]],
        value=constant_tensor,
        name=f"Constant_node_{tensor_proto_type}_{len(shape)}d"
    )

    return [input_info], output_info, [node], [], metadata  # Empty initializers since value is in the node
