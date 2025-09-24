import numpy as np
import random
from onnx import helper, TensorProto


def generate_squeeze_model(input_names, output_names):
    """
    Generates a Squeeze operator model.
    """
    initializers = []
    
    # Generate input shape with at least one dimension of size 1
    shape = [1, random.randint(1, 3), 1, random.randint(5, 10)]
    data = np.random.randn(*shape).astype(np.float32)

    # Create initializer for the input tensor
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    initializers.append(init_tensor)

    # Randomly decide whether to include axes
    use_axes = random.choice([True, False])

    if use_axes:
        # Find which axes in the shape are 1
        squeezable_axes = [i for i, dim in enumerate(shape) if dim == 1]
        if not squeezable_axes:
            # No valid axes to squeeze; force at least one
            squeezable_axes = [0]
        # Pick a random subset of the squeezable axes
        num_axes = random.randint(1, len(squeezable_axes))
        selected_axes = random.sample(squeezable_axes, num_axes)

        # Create an INT64 tensor with selected_axes
        axes_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(selected_axes)], selected_axes)
        initializers.append(axes_tensor)

        # Node takes 2 inputs: data and axes
        node = helper.make_node("Squeeze", inputs=[input_names[0], input_names[1]],
                                outputs=[output_names[0]], name=f"Squeeze_node")

        # Compute output shape manually
        out_shape = [dim for i, dim in enumerate(shape) if i not in selected_axes]

    else:
        # No axes specified: remove all dims of size 1
        node = helper.make_node("Squeeze", inputs=[input_names[0]], outputs=[output_names[0]],
                                name=f"Squeeze_node")
        out_shape = [dim for dim in shape if dim != 1]

    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)

    metadata = {
        "input_shapes": [shape],
        "output_shapes": [out_shape],
        "axes": selected_axes if use_axes else None
    }

    return [input_info], output_info, [node], initializers, metadata