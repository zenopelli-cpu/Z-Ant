import numpy as np
import random
from onnx import helper, TensorProto


def generate_split_model(input_names, output_names):
    """
    Generates a Split operator model.
    """
    initializers = []
    
    # Create a more realistic neural network test for Split
    shape = [2, 7, 28, 26]  # Fixed shape to match API checks
    axis = 0  # Split along the first dimension
    
    # Create input data
    data = np.random.randn(*shape).astype(np.float32)
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    initializers.append(init_tensor)
    
    # Calculate split output shapes
    out_shape = shape.copy()
    out_shape[axis] = shape[axis] // 2
    
    # Create intermediate output names
    split_output1 = "split_output1"
    split_output2 = "split_output2"
    processed_output1 = "processed_output1"
    processed_output2 = "processed_output2"
    
    # Create the Split node
    split_node = helper.make_node(
        "Split", 
        inputs=[input_names[0]], 
        outputs=[split_output1, split_output2],
        axis=axis, 
        name=f"Split_split_node",
        num_outputs=2  # Add this line
    )
    
    # Process the first split part with Relu
    relu_node = helper.make_node(
        "Relu",
        inputs=[split_output1],
        outputs=[processed_output1],
        name="Relu_after_split"
    )
    
    # Process the second split part with Sigmoid
    sigmoid_node = helper.make_node(
        "Sigmoid",
        inputs=[split_output2],
        outputs=[processed_output2],
        name="Sigmoid_after_split"
    )
    
    # Combine the processed outputs with Add
    add_node = helper.make_node(
        "Add",
        inputs=[processed_output1, processed_output2],
        outputs=[output_names[0]],
        name="Add_after_processing"
    )
    
    # All nodes needed for this model
    node = [split_node, relu_node, sigmoid_node, add_node]
    
    # Create the input tensor info
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    
    metadata = {
        "input_shapes": [shape], 
        "output_shapes": [out_shape],
        "axis": axis,
        "note": "This model splits the input, applies Relu to first part and Sigmoid to second part, then adds them together"
    }
    
    return [input_info], helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape), node, initializers, metadata