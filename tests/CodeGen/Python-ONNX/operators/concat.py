import numpy as np
import random
from onnx import helper, TensorProto


def generate_concat_model(input_names, output_names):
    """
    Generates a Concat operator model.
    """
    initializers = []
    
    # Due input con forma identica eccetto per la dimensione lungo l'asse di concatenazione
    shape = [1, random.randint(2,5), random.randint(10,50), random.randint(10,50)]
    rank = len(shape)
    axis = random.randint(0, rank-1)
    shape2 = shape.copy()
    shape2[axis] = shape[axis] + random.randint(1,3)
    data0 = np.random.randn(*shape).astype(np.float32)
    data1 = np.random.randn(*shape2).astype(np.float32)
    init_tensor0 = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data0.flatten().tolist())
    init_tensor1 = helper.make_tensor(input_names[1], TensorProto.FLOAT, shape2, data1.flatten().tolist())
    initializers.extend([init_tensor0, init_tensor1])
    out_shape = shape.copy()
    out_shape[axis] = shape[axis] + shape2[axis]
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
    node = helper.make_node("Concat", inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                            axis=axis, name=f"Concat_node_axis{axis}")
    
    # Define input_info before using it
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    metadata = {"input_shapes": [shape, shape2], "output_shapes": [out_shape]}
    return [input_info], output_info, [node], initializers, metadata