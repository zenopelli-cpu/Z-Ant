import numpy as np
import random
from onnx import helper, TensorProto


def generate_qlinearadd_model(input_names, output_names):
    """
    Generates a QLinearAdd operator model.
    """
    initializers = []
    
    # QLinearAdd implemented as composite: DequantizeLinear -> Add -> QuantizeLinear
    
    # Input tensor dimensions [N, C, H, W] (same for both inputs)
    batch_size = 1
    channels = random.randint(1, 8)
    height = random.randint(8, 16)
    width = random.randint(8, 16)
    
    input_shape = [batch_size, channels, height, width]
    output_shape = input_shape  # QLinearAdd preserves shape
    
    # Generate quantized input data A (uint8)
    a_data = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
    a_scale = np.random.uniform(0.001, 0.1)
    a_zero_point = np.random.randint(0, 256, dtype=np.uint8)
    
    # Generate quantized input data B (uint8)
    b_data = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
    b_scale = np.random.uniform(0.001, 0.1)
    b_zero_point = np.random.randint(0, 256, dtype=np.uint8)
    
    # Generate output quantization parameters
    c_scale = np.random.uniform(0.001, 0.1)
    c_zero_point = np.random.randint(0, 256, dtype=np.uint8)
    
    # Create intermediate tensor names
    dequant_a_name = f"dequant_a_{random.randint(1000, 9999)}"
    dequant_b_name = f"dequant_b_{random.randint(1000, 9999)}"
    add_result_name = f"add_result_{random.randint(1000, 9999)}"
    
    # Create initializers for quantized inputs and quantization parameters
    initializers.append(helper.make_tensor(input_names[0], TensorProto.UINT8, input_shape, a_data.flatten().tolist()))
    initializers.append(helper.make_tensor(input_names[1], TensorProto.FLOAT, [], [a_scale]))
    initializers.append(helper.make_tensor(input_names[2], TensorProto.UINT8, [], [a_zero_point]))
    initializers.append(helper.make_tensor(input_names[3], TensorProto.UINT8, input_shape, b_data.flatten().tolist()))
    initializers.append(helper.make_tensor(input_names[4], TensorProto.FLOAT, [], [b_scale]))
    initializers.append(helper.make_tensor(input_names[5], TensorProto.UINT8, [], [b_zero_point]))
    initializers.append(helper.make_tensor(input_names[6], TensorProto.FLOAT, [], [c_scale]))
    initializers.append(helper.make_tensor(input_names[7], TensorProto.UINT8, [], [c_zero_point]))
    
    # Create three nodes: DequantizeLinear A, DequantizeLinear B, Add, QuantizeLinear
    nodes = []
    
    # 1. DequantizeLinear for input A
    dequant_a_node = helper.make_node(
        "DequantizeLinear",
        inputs=[input_names[0], input_names[1], input_names[2]],  # A, A_scale, A_zero_point
        outputs=[dequant_a_name],
        name=f"DequantizeLinear_A_{random.randint(1000, 9999)}"
    )
    nodes.append(dequant_a_node)
    
    # 2. DequantizeLinear for input B
    dequant_b_node = helper.make_node(
        "DequantizeLinear",
        inputs=[input_names[3], input_names[4], input_names[5]],  # B, B_scale, B_zero_point
        outputs=[dequant_b_name],
        name=f"DequantizeLinear_B_{random.randint(1000, 9999)}"
    )
    nodes.append(dequant_b_node)
    
    # 3. Add the dequantized values
    add_node = helper.make_node(
        "Add",
        inputs=[dequant_a_name, dequant_b_name],
        outputs=[add_result_name],
        name=f"Add_{random.randint(1000, 9999)}"
    )
    nodes.append(add_node)
    
    # 4. QuantizeLinear to output
    quant_node = helper.make_node(
        "QuantizeLinear",
        inputs=[add_result_name, input_names[6], input_names[7]],  # add_result, C_scale, C_zero_point
        outputs=[output_names[0]],
        name=f"QuantizeLinear_{random.randint(1000, 9999)}"
    )
    nodes.append(quant_node)
    
    # Create output info (quantized uint8)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.UINT8, output_shape)
    
    # Dummy input info
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
    
    metadata = {
        "input_shapes": [input_shape, input_shape],
        "output_shapes": [output_shape],
        "a_scale": a_scale,
        "b_scale": b_scale,
        "c_scale": c_scale,
        "operation": "quantized_addition_composite"
    }
    
    return [input_info], output_info, nodes, initializers, metadata