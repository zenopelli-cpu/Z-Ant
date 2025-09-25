import numpy as np
import random
from onnx import helper, TensorProto

def generate_shape_model(input_names, output_names):
    """Generate Shape operator model."""
    initializers = []
    
    # Generate random input shape
    shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
    data = np.random.randn(*shape).astype(np.float32)
    
    # Create initializer tensor for input
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    initializers.append(init_tensor)
    
    # Input info (float tensor)
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    
    # Output info - Shape operator outputs 1D int64 tensor with shape [len(input_shape)]
    # The output contains the dimensions of the input tensor
    output_shape = [len(shape)]  # 1D tensor with length = number of input dimensions
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.INT64, output_shape)
    
    # Create Shape node
    node = helper.make_node("Shape", inputs=[input_names[0]], outputs=[output_names[0]], name="Shape_node")
    
    # Update metadata to reflect correct output shape (1D tensor with input's rank)
    metadata = {"input_shapes": [shape], "output_shapes": [output_shape]}
    
    return [input_info], output_info, [node], initializers, metadata