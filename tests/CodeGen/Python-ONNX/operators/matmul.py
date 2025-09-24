import numpy as np
import random
from onnx import helper, TensorProto


def generate_matmul_model(input_names, output_names):
    """
    Generates a MatMul operator model.
    """
    initializers = []
    
    # Genera due matrici 2D compatibili
    M_val = random.randint(2,10)
    K_val = random.randint(2,10)
    N_val = random.randint(2,10)
    A_shape = [M_val, K_val]
    B_shape = [K_val, N_val]
    A_data = np.random.randn(*A_shape).astype(np.float32)
    B_data = np.random.randn(*B_shape).astype(np.float32)
    init_tensor_A = helper.make_tensor(input_names[0], TensorProto.FLOAT, A_shape, A_data.flatten().tolist())
    init_tensor_B = helper.make_tensor(input_names[1], TensorProto.FLOAT, B_shape, B_data.flatten().tolist())
    initializers.extend([init_tensor_A, init_tensor_B])
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, [M_val, N_val])
    node = helper.make_node("MatMul", inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                            name=f"MatMul_node")
    
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, A_shape)
    metadata = {"input_shapes": [A_shape, B_shape], "output_shapes": [[M_val, N_val]]}
    return [input_info], output_info, [node], initializers, metadata