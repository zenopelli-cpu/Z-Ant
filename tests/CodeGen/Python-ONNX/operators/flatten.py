import numpy as np
import random
from onnx import helper, TensorProto


def generate_flatten_model(input_names, output_names):
    """
    Generates a Flatten operator model.
    """
    initializers = []
    
    # Generate casual rank and shape
    rank = random.randint(1, 4)
    if rank == 0:
        shape = []  
    else:
        shape = [random.randint(1, 10) for _ in range(rank)]

    # Generate casual data
    total_size = 1 if rank == 0 else np.prod(shape)
    data = np.random.randn(total_size).astype(np.float32)
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    initializers.append(init_tensor)

    # choosing right axis
    axis = random.randint(-rank, rank) if rank > 0 else 0  # for scalar, axis is 0

    # Calculate output shape
    if rank == 0:
        out_shape = [1, 1]  # Scalare -> [1, 1]
    else:
        outer_dim = 1
        normalized_axis = axis if axis >= 0 else axis + rank
        for i in range(normalized_axis):
            outer_dim *= shape[i]
        inner_dim = 1
        for i in range(normalized_axis, rank):
            inner_dim *= shape[i]
        out_shape = [outer_dim, inner_dim]

    # create input and output info
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)

    # Create Flatten node
    node = helper.make_node(
        "Flatten",
        inputs=[input_names[0]],
        outputs=[output_names[0]],
        axis=axis,
        name=f"Flatten_node_axis{axis}"
    )

    # Metadati
    metadata = {
        "input_shapes": [shape],
        "output_shapes": [out_shape],
        "axis": axis
    }

    return [input_info], output_info, [node], initializers, metadata