import numpy as np
import random
from onnx import helper, TensorProto

def generate_clip_model(input_names, output_names):
    """Generate Clip operator model."""
    initializers = []
    
    # Generate input tensor with predictable shape and values
    shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
    data = np.random.randn(*shape).astype(np.float32)
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    initializers.append(init_tensor)

    # Generate min value tensor (optional)
    min_value = round(random.uniform(-2.0, 0.0), 2)
    min_tensor = helper.make_tensor(input_names[1], TensorProto.FLOAT, [], [min_value])
    initializers.append(min_tensor)

    # Generate max value tensor (optional)
    max_value = round(random.uniform(0.0, 2.0), 2)
    max_tensor = helper.make_tensor(input_names[2], TensorProto.FLOAT, [], [max_value])
    initializers.append(max_tensor)

    # Output shape is same as input shape
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

    # Create node with all three inputs
    node = helper.make_node(
        "Clip",
        inputs=[input_names[0], input_names[1], input_names[2]],
        outputs=[output_names[0]],
        name=f"Clip_node_min{min_value}_max{max_value}"
    )

    # Input info placeholder (not used in practice)
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)

    metadata = {
        "input_shapes": [shape],
        "output_shapes": [shape],
        "min_value": min_value,
        "max_value": max_value
    }

    return [input_info], output_info, [node], initializers, metadata