import numpy as np
import random
from onnx import helper, TensorProto


def generate_dynamicquantizelinear_model(input_names, output_names):
    """
    Generates a DynamicQuantizeLinear operator model.
    """
    initializers = []
    
    # DynamicQuantizeLinear: quantizes a tensor dynamically
    # Input: x (float) -> Outputs: y (quantized), y_scale (float), y_zero_point (uint8)
    
    shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
    
    # Generate input data (float32)
    data = np.random.randn(*shape).astype(np.float32) * 10  # Scale for better quantization
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    initializers.append(init_tensor)
    
    # Create output infos
    # y: quantized tensor (uint8)
    y_info = helper.make_tensor_value_info(output_names[0], TensorProto.UINT8, shape)
    # y_scale: scale factor (float, scalar)
    y_scale_info = helper.make_tensor_value_info(output_names[1], TensorProto.FLOAT, [])
    # y_zero_point: zero point (uint8, scalar)
    y_zero_point_info = helper.make_tensor_value_info(output_names[2], TensorProto.UINT8, [])
    
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    
    node = helper.make_node(
        "DynamicQuantizeLinear",
        inputs=[input_names[0]],
        outputs=[output_names[0], output_names[1], output_names[2]],  # y, y_scale, y_zero_point
        name=f"DynamicQuantizeLinear_node"
    )
    
    metadata = {
        "input_shapes": [shape],
        "output_shapes": [shape, [], []],  # y_shape, scalar, scalar
        "operation": "dynamic_quantization"
    }
    
    return [input_info], [y_info, y_scale_info, y_zero_point_info], [node], initializers, metadata