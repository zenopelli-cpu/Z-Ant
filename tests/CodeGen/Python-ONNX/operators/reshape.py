import numpy as np
import random
from onnx import helper, TensorProto


def generate_reshape_model(input_names, output_names):
    """
    Generates a Reshape operator model.
    """
    initializers = []
    
    # First input: data; second input: new shape (initializer)
    shape = [random.randint(1,4) for _ in range(4)]
    data = np.random.randn(*shape).astype(np.float32)
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    initializers.append(init_tensor)
    
    # Define input_info before using it
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    
    new_shape = shape.copy()
    random.shuffle(new_shape)
    shape_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(new_shape)], new_shape)
    initializers.append(shape_tensor)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, new_shape)
    
    # Remove the 'shape' attribute - it's not valid for ONNX Reshape
    node = helper.make_node("Reshape", 
                            inputs=[input_names[0], input_names[1]], 
                            outputs=[output_names[0]],
                            name=f"Reshape_node")
    
    metadata = {"input_shapes": [shape, new_shape], "output_shapes": [new_shape]}
    return [input_info], output_info, [node], initializers, metadata