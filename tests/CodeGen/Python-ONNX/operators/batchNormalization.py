import numpy as np
import random
from onnx import helper, TensorProto


def generate_batchnormalization_model(input_names, output_names):
    """
    Generates a BatchNormalization operator model.
    """
    initializers = []
    
    # BatchNorm has 5 inputs: X, scale, B, mean, var
    # Output shape is the same as input
    shape = [1, random.randint(1, 4), random.randint(10, 50), random.randint(10, 50)]
    C = shape[1]  # Number of channels (second dimension)

    # Create input tensor
    data_X = np.random.randn(*shape).astype(np.float32)
    init_X = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data_X.flatten().tolist())

    # scale (gamma), bias (beta), mean, var — all of shape [C]
    scale = np.random.randn(C).astype(np.float32)
    B = np.random.randn(C).astype(np.float32)
    mean = np.random.randn(C).astype(np.float32)
    var = np.abs(np.random.randn(C)).astype(np.float32)  # ensure variance is non-negative

    init_scale = helper.make_tensor(input_names[1], TensorProto.FLOAT, [C], scale.tolist())
    init_B = helper.make_tensor(input_names[2], TensorProto.FLOAT, [C], B.tolist())
    init_mean = helper.make_tensor(input_names[3], TensorProto.FLOAT, [C], mean.tolist())
    init_var = helper.make_tensor(input_names[4], TensorProto.FLOAT, [C], var.tolist())

    initializers.extend([init_X, init_scale, init_B, init_mean, init_var])

    epsilon = round(random.uniform(1e-5, 1e-2), 6)
    momentum = round(random.uniform(0.8, 0.99), 3)

    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

    node = helper.make_node(
        "BatchNormalization",
        inputs=input_names[:5],
        outputs=[output_names[0]],
        epsilon=epsilon,
        momentum=momentum,
        name=f"BatchNormalization_node"
    )

    metadata = {
        "input_shapes": [shape, [C], [C], [C], [C]],
        "output_shapes": [shape],
        "epsilon": epsilon,
        "momentum": momentum
    }

    return [input_info], output_info, [node], initializers, metadata