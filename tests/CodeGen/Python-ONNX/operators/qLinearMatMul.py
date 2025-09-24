import numpy as np
import random
from onnx import helper, TensorProto


def generate_qlinearmatmul_model(input_names, output_names):
    """
    Generates a QLinearMatMul operator model.
    """
    initializers = []
    
    # QLinearMatMul implemented as composite: DequantizeLinear -> MatMul -> QuantizeLinear
    
    # Input tensor dimensions for matrix multiplication
    batch_size = 1
    m = random.randint(4, 16)  # rows of first matrix
    k = random.randint(4, 16)  # cols of first matrix / rows of second matrix  
    n = random.randint(4, 16)  # cols of second matrix
    
    a_shape = [batch_size, m, k]  # First matrix
    b_shape = [batch_size, k, n]  # Second matrix
    output_shape = [batch_size, m, n]  # Result matrix
    
    # Generate quantized input data A (uint8)
    a_data = np.random.randint(0, 256, size=a_shape, dtype=np.uint8)
    a_scale = np.random.uniform(0.001, 0.1)
    a_zero_point = np.random.randint(0, 256, dtype=np.uint8)
    
    # Generate quantized input data B (uint8)
    b_data = np.random.randint(0, 256, size=b_shape, dtype=np.uint8)
    b_scale = np.random.uniform(0.001, 0.1)
    b_zero_point = np.random.randint(0, 256, dtype=np.uint8)
    
    # Generate output quantization parameters
    c_scale = np.random.uniform(0.001, 0.1)
    c_zero_point = np.random.randint(0, 256, dtype=np.uint8)
    
    # Create intermediate tensor names
    dequant_a_name = f"dequant_a_{random.randint(1000, 9999)}"
    dequant_b_name = f"dequant_b_{random.randint(1000, 9999)}"
    matmul_result_name = f"matmul_result_{random.randint(1000, 9999)}"
    
    # Create initializers for quantized inputs and quantization parameters
    initializers.append(helper.make_tensor(input_names[0], TensorProto.UINT8, a_shape, a_data.flatten().tolist()))
    initializers.append(helper.make_tensor(input_names[1], TensorProto.FLOAT, [], [a_scale]))
    initializers.append(helper.make_tensor(input_names[2], TensorProto.UINT8, [], [a_zero_point]))
    initializers.append(helper.make_tensor(input_names[3], TensorProto.UINT8, b_shape, b_data.flatten().tolist()))
    initializers.append(helper.make_tensor(input_names[4], TensorProto.FLOAT, [], [b_scale]))
    initializers.append(helper.make_tensor(input_names[5], TensorProto.UINT8, [], [b_zero_point]))
    initializers.append(helper.make_tensor(input_names[6], TensorProto.FLOAT, [], [c_scale]))
    initializers.append(helper.make_tensor(input_names[7], TensorProto.UINT8, [], [c_zero_point]))
    
    # Create four nodes: DequantizeLinear A, DequantizeLinear B, MatMul, QuantizeLinear
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
    
    # 3. MatMul the dequantized values
    matmul_node = helper.make_node(
        "MatMul",
        inputs=[dequant_a_name, dequant_b_name],
        outputs=[matmul_result_name],
        name=f"MatMul_{random.randint(1000, 9999)}"
    )
    nodes.append(matmul_node)
    
    # 4. QuantizeLinear to output
    quant_node = helper.make_node(
        "QuantizeLinear",
        inputs=[matmul_result_name, input_names[6], input_names[7]],  # matmul_result, C_scale, C_zero_point
        outputs=[output_names[0]],
        name=f"QuantizeLinear_{random.randint(1000, 9999)}"
    )
    nodes.append(quant_node)
    
    # Create output info (quantized uint8)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.UINT8, output_shape)
    
    # Dummy input info
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, a_shape)
    
    metadata = {
        "input_shapes": [a_shape, b_shape],
        "output_shapes": [output_shape],
        "a_scale": a_scale,
        "b_scale": b_scale,
        "c_scale": c_scale,
        "operation": "quantized_matmul_composite"
    }
    
    return [input_info], output_info, nodes, initializers, metadata