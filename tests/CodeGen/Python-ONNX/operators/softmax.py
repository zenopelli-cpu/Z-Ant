import numpy as np
import random
from onnx import helper, TensorProto


def generate_softmax_model(input_names, output_names):
    """
    Generates a Softmax operator model.
    """
    initializers = []
    
    shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
    rank = len(shape)
    axis = random.randint(-rank, rank-1)
    data = np.random.randn(*shape).astype(np.float32)
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    initializers.append(init_tensor)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)
    node = helper.make_node("Softmax", inputs=[input_names[0]], outputs=[output_names[0]], 
                            axis=axis, name=f"Softmax_node_axis{axis}")
    
    # Define input_info before using it
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    metadata = {"input_shapes": [shape], "output_shapes": [shape], "axis": axis}
    return [input_info], output_info, [node], initializers, metadata