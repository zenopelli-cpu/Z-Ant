import numpy as np
import random
from onnx import helper, TensorProto


def generate_gemm_model(input_names, output_names):
    """
    Generates a Gemm operator model.
    """

    initializers = []
    
    # Gemm: esegue A * B + C
    M_val = random.randint(2,10)
    K_val = random.randint(2,10)
    N_val = random.randint(2,10)
    A_shape = [M_val, K_val]
    B_shape = [K_val, N_val]
    C_shape = [M_val, N_val]  # C must be broadcastable to (M,N)
    
    A_data = np.random.randn(*A_shape).astype(np.float32)
    B_data = np.random.randn(*B_shape).astype(np.float32)
    C_data = np.random.randn(*C_shape).astype(np.float32)
    
    init_tensor_A = helper.make_tensor(input_names[0], TensorProto.FLOAT, A_shape, A_data.flatten().tolist())
    init_tensor_B = helper.make_tensor(input_names[1], TensorProto.FLOAT, B_shape, B_data.flatten().tolist())
    init_tensor_C = helper.make_tensor(input_names[2], TensorProto.FLOAT, C_shape, C_data.flatten().tolist())
    initializers.extend([init_tensor_A, init_tensor_B, init_tensor_C])
    
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, [M_val, N_val])
    
    alpha = round(random.uniform(0.5, 2.0), 2)
    beta = round(random.uniform(0.5, 2.0), 2)
    
    # Fix: Don't use transA/transB to avoid dimension mismatches
    node = helper.make_node("Gemm", inputs=[input_names[0], input_names[1], input_names[2]], outputs=[output_names[0]],
                            alpha=alpha, beta=beta, 
                            name=f"Gemm_node_alpha{alpha}beta{beta}")
    
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, A_shape)
    metadata = {"input_shapes": [A_shape, B_shape, C_shape], "output_shapes": [[M_val, N_val]],
                "alpha": alpha, "beta": beta}
    return [input_info], output_info, [node], initializers, metadata