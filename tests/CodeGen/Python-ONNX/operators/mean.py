import numpy as np
import random
from onnx import helper, TensorProto


def generate_mean_model(input_names, output_names):
    """
    Generates a Mean operator model.
    """
    initializers = []
    
    num_inputs = random.randint(1, 5)
    max_dims = 3 
    
    # 1. Generate a potential "output" shape first
    output_shape = [random.randint(1, 4) for _ in range(max_dims)]

    shapes = []
    
    for i in range(num_inputs):
        # 2. Derive compatible input shape from the output shape
        current_shape = []
        for dim_size in output_shape:
            # Each dimension is either the same as output_shape or 1
            current_shape.append(random.choice([1, dim_size]))
        shapes.append(current_shape)

        # data generation for each input tensor
        data = np.random.randn(*current_shape).astype(np.float32)
        tensor_name = input_names[i]
        init_tensor = helper.make_tensor(tensor_name, TensorProto.FLOAT, current_shape, data.flatten().tolist())
        initializers.append(init_tensor)
    
    # The actual output shape is already determined by output_shape list
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
    node = helper.make_node(
        "Mean",
        inputs=[input_names[i] for i in range(num_inputs)],
        outputs=[output_names[0]],
        name=f"Mean_node_{num_inputs}_inputs"
    )
    
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shapes[0])
    metadata = {
        "input_shapes": shapes,
        "output_shapes": [output_shape]
    }
    
    return [input_info], output_info, [node], initializers, metadata