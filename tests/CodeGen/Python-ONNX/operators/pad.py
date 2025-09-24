import numpy as np
import random
from onnx import helper, TensorProto


def generate_pad_model(input_names, output_names):
    """
    Generates a Pad operator model.
    """
    initializers = []
    
    # Pad operator: pads tensor with specified padding modes
    # Supports constant, reflect, edge, and wrap modes according to ONNX spec
    
    shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
    data = np.random.randn(*shape).astype(np.float32)
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    initializers.append(init_tensor)
    rank = len(shape)
    
    # Generate pads tensor: [begin_pad_0, begin_pad_1, ..., end_pad_0, end_pad_1, ...]
    pads = [random.randint(0, 2) for _ in range(2 * rank)]
    out_shape = [shape[i] + pads[i] + pads[i + rank] for i in range(rank)]
    pads_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(pads)], pads)
    initializers.append(pads_tensor)
    
    # Choose padding mode
    mode = random.choice(["constant", "reflect", "edge"])
    node_inputs = [input_names[0], input_names[1]]
    constant_value = None
    
    # Add constant value if mode is constant
    if mode == "constant":
        constant_value = round(random.uniform(-1.0, 1.0), 2)
        constant_tensor = helper.make_tensor(input_names[2], TensorProto.FLOAT, [], [constant_value])
        initializers.append(constant_tensor)
        node_inputs.append(input_names[2])
    
    # Note: axes parameter is not commonly used in basic Pad, keeping it simple
    axes = None
    
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
    
    node = helper.make_node(
        "Pad",
        inputs=node_inputs,
        outputs=[output_names[0]],
        mode=mode,
        name=f"Pad_node_mode_{mode}"
    )
    
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    
    metadata = {
        "input_shapes": [shape],
        "output_shapes": [out_shape],
        "pads": pads,
        "mode": mode,
        "axes": axes,
        "constant_value": constant_value
    }
    
    return [input_info], output_info, [node], initializers, metadata