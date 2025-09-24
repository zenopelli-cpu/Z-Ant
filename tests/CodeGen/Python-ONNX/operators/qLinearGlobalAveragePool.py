import numpy as np
import random
from onnx import helper, TensorProto


def generate_qlinearglobalaveragepool_model(input_names, output_names):
    """
    Generates a QLinearGlobalAveragePool operator model.
    """
    initializers = []
    
    # QLinearGlobalAveragePool implemented as: DequantizeLinear -> GlobalAveragePool -> QuantizeLinear
    
    # Input tensor dimensions [N, C, H, W]
    batch_size = 1
    channels = random.randint(1, 8)
    input_height = random.randint(8, 16)
    input_width = random.randint(8, 16)
    
    input_shape = [batch_size, channels, input_height, input_width]
    output_shape = [batch_size, channels, 1, 1]  # Global pooling reduces spatial dims to 1x1
    
    # Generate quantized input data (uint8)
    x_data = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
    x_scale = np.random.uniform(0.001, 0.1)
    x_zero_point = np.random.randint(0, 256, dtype=np.uint8)
    
    # Generate output quantization parameters
    y_scale = np.random.uniform(0.001, 0.1)
    y_zero_point = np.random.randint(0, 256, dtype=np.uint8)
    
    # Create initializers for all quantization parameters
    initializers.append(helper.make_tensor(input_names[0], TensorProto.UINT8, input_shape, x_data.flatten().tolist()))
    initializers.append(helper.make_tensor(input_names[1], TensorProto.FLOAT, [], [x_scale]))
    initializers.append(helper.make_tensor(input_names[2], TensorProto.UINT8, [], [x_zero_point]))
    initializers.append(helper.make_tensor(input_names[3], TensorProto.FLOAT, [], [y_scale]))
    initializers.append(helper.make_tensor(input_names[4], TensorProto.UINT8, [], [y_zero_point]))
    
    # Create intermediate tensor names
    dequant_output = f"dequant_{output_names[0]}"
    gap_output = f"gap_{output_names[0]}"
    
    # Create three nodes: DequantizeLinear -> GlobalAveragePool -> QuantizeLinear
    nodes = []
    
    # 1. DequantizeLinear node
    dequant_node = helper.make_node(
        "DequantizeLinear",
        inputs=[input_names[0], input_names[1], input_names[2]],  # x, x_scale, x_zero_point
        outputs=[dequant_output],
        name=f"Dequant_QLinearGlobalAveragePool_node"
    )
    nodes.append(dequant_node)
    
    # 2. GlobalAveragePool node
    gap_node = helper.make_node(
        "GlobalAveragePool",
        inputs=[dequant_output],
        outputs=[gap_output],
        name=f"GAP_QLinearGlobalAveragePool_node"
    )
    nodes.append(gap_node)
    
    # 3. QuantizeLinear node
    quant_node = helper.make_node(
        "QuantizeLinear",
        inputs=[gap_output, input_names[3], input_names[4]],  # gap_output, y_scale, y_zero_point
        outputs=[output_names[0]],
        name=f"Quant_QLinearGlobalAveragePool_node"
    )
    nodes.append(quant_node)
    
    # Create output info (quantized uint8)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.UINT8, output_shape)
    
    # Dummy input info
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
    
    metadata = {
        "input_shapes": [input_shape],
        "output_shapes": [output_shape],
        "x_scale": x_scale,
        "y_scale": y_scale,
        "channels": channels,
        "spatial_size": input_height * input_width,
        "implementation": "dequant_gap_quant"
    }
    
    return [input_info], output_info, nodes, initializers, metadata