import numpy as np
import random
from onnx import helper, TensorProto


def generate_transpose_model(input_names, output_names):
    """
    Generates a Transpose operator model.
    """
    initializers = []
    
    # Genera una permutazione casuale per Transpose
    shape = [random.randint(1,4) for _ in range(4)]
    data = np.random.randn(*shape).astype(np.float32)
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    initializers.append(init_tensor)
    rank = len(shape)
    perm = list(range(rank))
    random.shuffle(perm)
    out_shape = [shape[i] for i in perm]
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
    node = helper.make_node("Transpose", inputs=[input_names[0]], outputs=[output_names[0]],
                            perm=perm, name=f"Transpose_node_perm{perm}")
    
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "perm": perm}
    return [input_info], output_info, [node], initializers, metadata