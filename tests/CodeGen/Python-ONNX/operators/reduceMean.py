import numpy as np
import random
from onnx import helper, TensorProto

def generate_reducemean_model(input_names, output_names):
    """
    Generates a ReduceMean operator model.
    """
    initializers = []
    # Generate input tensor with 4D shape for typical use case
    shape = [random.randint(2, 6) for _ in range(4)]  # Ensure shape has values > 1 for meaningful reduction
    data = np.random.randn(*shape).astype(np.float32)
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    initializers.append(init_tensor)
    
    # Define input_info
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    
    # Choose random axes to reduce along (optional input)
    # Can be empty (reduce all), single axis, or multiple axes
    axes_options = [
        [],  # reduce all axes
        [random.randint(0, len(shape)-1)],  # reduce single random axis
        sorted(random.sample(range(len(shape)), k=random.randint(1, min(2, len(shape)))))  # reduce multiple axes
    ]
    axes = random.choice(axes_options)
    
    # ReduceMean attributes
    keepdims = random.choice([0, 1])  # whether to keep dimensions
    noop_with_empty_axes = random.choice([0, 1])  # behavior when axes is empty
    
    # Calculate output shape
    if len(axes) == 0 and noop_with_empty_axes == 1:
        # No reduction when axes is empty and noop_with_empty_axes is True
        out_shape = shape.copy()
    elif len(axes) == 0:
        # Reduce all axes
        out_shape = [1] * len(shape) if keepdims else []
    else:
        # Reduce specified axes
        out_shape = []
        for i, dim in enumerate(shape):
            if i in axes:
                if keepdims:
                    out_shape.append(1)
                # else: dimension is removed
            else:
                out_shape.append(dim)
    
    # Create node inputs - only add axes if it's not empty
    node_inputs = [input_names[0]]
    
    # Only create axes tensor if axes is not empty
    if len(axes) > 0:
        axes_tensor = helper.make_tensor(f"axes_{random.randint(1000,9999)}", TensorProto.INT64, [len(axes)], axes)
        initializers.append(axes_tensor)
        node_inputs.append(axes_tensor.name)
    
    # Create output info
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
    
    # Create the ReduceMean node
    node = helper.make_node(
        "ReduceMean",
        inputs=node_inputs,
        outputs=[output_names[0]],
        keepdims=keepdims,
        noop_with_empty_axes=noop_with_empty_axes,
        name=f"ReduceMean_node"
    )
    
    metadata = {
        "input_shapes": [shape],
        "output_shapes": [out_shape],
        "axes": axes,
        "keepdims": keepdims,
        "noop_with_empty_axes": noop_with_empty_axes
    }
    
    return [input_info], output_info, [node], initializers, metadata