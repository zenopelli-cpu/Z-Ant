import numpy as np
import random
from onnx import helper, TensorProto



def generate_unsqueeze_model(input_names, output_names):
    """
    Generates an Unsqueeze operator model.
    
    Args:
        input_names: List of input names (expects at least 1: data input)
        output_names: List of output names (expects at least 1)
    
    Returns:
        tuple: (input_infos, output_info, nodes, initializers, metadata)
    """
    # Validate inputs
    if len(input_names) < 1:
        raise ValueError("input_names must contain at least 1 element: [data_input]")
    if len(output_names) < 1:
        raise ValueError("output_names must contain at least 1 element")
    
    initializers = []
    
    # Generate random input shape (1-4 dimensions, each with size 1-10 for better testing)
    rank = random.randint(1, 4)
    shape = [random.randint(1, 10) for _ in range(rank)]
    
    # Select valid axis for unsqueeze operation
    # Axis can be in range [-rank-1, rank] for Unsqueeze
    axis = random.randint(-rank-1, rank)
    axes = [axis]
    
    # Create axes tensor as initializer (constant)
    # Use a unique name that's not in input_names to avoid conflicts
    axes_name = f"axes_param_{random.randint(1000, 9999)}"
    axes_tensor = helper.make_tensor(
        axes_name, 
        TensorProto.INT64, 
        [len(axes)], 
        axes
    )
    initializers.append(axes_tensor)
    
    # Calculate output shape with proper negative axis handling
    if axis < 0:
        # Convert negative axis to positive: axis + rank + 1
        actual_axis = axis + rank + 1
    else:
        actual_axis = axis
    
    # Clamp axis to valid range to prevent index errors
    actual_axis = max(0, min(actual_axis, rank))
    
    # Create output shape by inserting dimension of size 1
    out_shape = shape.copy()
    out_shape.insert(actual_axis, 1)
    
    # Create only the data input info (not the axes since it's an initializer)
    data_input_info = helper.make_tensor_value_info(
        input_names[0], 
        TensorProto.FLOAT, 
        shape
    )
    
    output_info = helper.make_tensor_value_info(
        output_names[0], 
        TensorProto.FLOAT, 
        out_shape
    )
    
    # Create the Unsqueeze node using data input and axes initializer
    node = helper.make_node(
        "Unsqueeze",
        inputs=[input_names[0], axes_name],  # data input + axes initializer
        outputs=[output_names[0]],
        name=f"Unsqueeze_node_axis{axis}"
    )
    
    # Prepare metadata for testing
    metadata = {
        "input_shapes": [shape], 
        "output_shapes": [out_shape], 
        "axes": axes,
        "original_axis": axis,
        "actual_axis": actual_axis,
        "rank": rank,
        "axes_name": axes_name
    }
    
    return [data_input_info], output_info, [node], initializers, metadata
