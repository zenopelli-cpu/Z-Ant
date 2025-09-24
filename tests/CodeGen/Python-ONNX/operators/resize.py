import numpy as np
import random
from onnx import helper, TensorProto


def generate_resize_model(input_names, output_names):
    """
    Generates a Resize operator model.
    """
    initializers = []
    
    # Generate a random input tensor shape: (N=1, C=random, H=random, W=random)
    shape = [1, random.randint(1, 4), random.randint(10, 50), random.randint(10, 50)]
    data = np.random.randn(*shape).astype(np.float32)
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.ravel())
    initializers.append(init_tensor)

    # Empty ROI tensor (optional in ONNX, but included for compatibility)
    roi_name = input_names[1] + "_roi"
    roi_tensor = helper.make_tensor(roi_name, TensorProto.FLOAT, [0], [])
    initializers.append(roi_tensor)

    # Empty Scales tensor (using Sizes instead)
    scales_name = input_names[2] + "_scales"
    scales_tensor = helper.make_tensor(scales_name, TensorProto.FLOAT, [0], [])
    initializers.append(scales_tensor)

    # Compute new sizes (scaled spatial dimensions)
    scale_factor = round(random.uniform(0.5, 2.0), 2)
    new_height = int(shape[2] * scale_factor)
    new_width = int(shape[3] * scale_factor)

    sizes = [shape[0], shape[1], new_height, new_width]
    sizes_name = input_names[3] + "_sizes"
    sizes_tensor = helper.make_tensor(sizes_name, TensorProto.INT64, [len(sizes)], sizes)
    initializers.append(sizes_tensor)

    # Output tensor info
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, sizes)

    # Choose a resize mode (explicit selection)
    mode = random.choice(["nearest", "linear"])

    # Create ONNX Resize node
    node = helper.make_node(
        "Resize",
        inputs=[input_names[0], roi_name, scales_name, sizes_name],
        outputs=[output_names[0]],
        mode=mode,
        name=f"Resize_mode_{mode}"
    )

    # Input info placeholder (not actually used)
    input_info = helper.make_tensor_value_info("unused_input", TensorProto.FLOAT, shape)

    # Metadata dictionary
    metadata = {
        "input_shapes": [shape],
        "output_shapes": [sizes],
        "mode": mode,
        "scale_factor": scale_factor,
    }

    return [input_info], output_info, [node], initializers, metadata