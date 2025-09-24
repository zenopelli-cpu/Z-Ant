import numpy as np
import random
from onnx import helper, TensorProto


def generate_gather_model(input_names, output_names):
    """
    Generates a Gather operator model.
    """
    initializers = []
    
    # First input: data; second input: indices
    shape = [5, random.randint(5,10)]
    data = np.random.randn(*shape).astype(np.float32)
    init_data = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    initializers.append(init_data)
    
    # Pick a random axis
    axis = random.randint(0, len(shape)-1)
    
    # Ensure indices are within bounds of the chosen axis
    max_index = shape[axis] - 1
    indices_shape = [random.randint(1,3)]
    indices_data = np.random.randint(0, max_index + 1, size=indices_shape).astype(np.int64)
    init_indices = helper.make_tensor(input_names[1], TensorProto.INT64, indices_shape, indices_data.flatten().tolist())
    initializers.append(init_indices)
    
    # Calculate output shape
    out_shape = list(shape)
    out_shape[axis] = indices_shape[0]
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
    
    node = helper.make_node("Gather", inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                          axis=axis, name=f"Gather_node_axis{axis}")
    
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    metadata = {
        "input_shapes": [shape, indices_shape], 
        "output_shapes": [out_shape],
        "axis": axis,
        "indices": indices_data.tolist()
    }
    return [input_info], output_info, [node], initializers, metadata