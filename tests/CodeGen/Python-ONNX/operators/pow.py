import numpy as np
import random
from onnx import helper, TensorProto

def generate_pow_model(input_names, output_names):
    """
    Generates a Pow operator model with comprehensive broadcasting tests.
    """
    initializers = []
    
    # Choose random data type for base (X) from supported types
    base_dtypes = [
        (TensorProto.FLOAT, np.float32),
        (TensorProto.DOUBLE, np.float64),
        (TensorProto.FLOAT16, np.float16),
        (TensorProto.INT32, np.int32),
        (TensorProto.INT64, np.int64)
    ]
    base_dtype_proto, base_dtype_np = random.choice(base_dtypes)
    
    # Choose random data type for exponent (Y) from supported types
    exp_dtypes = [
        (TensorProto.FLOAT, np.float32),
        (TensorProto.DOUBLE, np.float64),
        (TensorProto.FLOAT16, np.float16),
        (TensorProto.INT32, np.int32),
        (TensorProto.INT64, np.int64),
        (TensorProto.INT16, np.int16),
        (TensorProto.INT8, np.int8),
        (TensorProto.UINT32, np.uint32),
        (TensorProto.UINT64, np.uint64),
        (TensorProto.UINT16, np.uint16),
        (TensorProto.UINT8, np.uint8)
    ]
    exp_dtype_proto, exp_dtype_np = random.choice(exp_dtypes)
    
    # Define various broadcasting scenarios
    broadcast_scenarios = [
        # (base_shape, exponent_shape, description)
        # Scalar broadcasting
        ([5], [1], "scalar_to_1d"),
        ([3, 4], [1], "scalar_to_2d"),
        ([2, 3, 4], [1], "scalar_to_3d"),
        ([1, 3, 4, 5], [1], "scalar_to_4d"),
        
        # Same shape (no broadcasting)
        ([5], [5], "same_1d"),
        ([3, 4], [3, 4], "same_2d"),
        ([2, 3, 4], [2, 3, 4], "same_3d"),
        
        # One dimension is 1
        ([3, 4], [1, 4], "broadcast_first_dim"),
        ([3, 4], [3, 1], "broadcast_last_dim"),
        ([2, 3, 4], [1, 3, 4], "broadcast_first_of_3d"),
        ([2, 3, 4], [2, 1, 4], "broadcast_middle_of_3d"),
        ([2, 3, 4], [2, 3, 1], "broadcast_last_of_3d"),
        
        # Multiple dimensions are 1
        ([3, 4, 5], [1, 1, 5], "broadcast_first_two"),
        ([3, 4, 5], [1, 4, 1], "broadcast_first_and_last"),
        ([2, 3, 4, 5], [1, 1, 4, 5], "broadcast_first_two_of_4d"),
        
        # Different rank broadcasting
        ([3, 4, 5], [5], "lower_rank_to_higher"),
        ([2, 3, 4], [4], "1d_to_3d"),
        ([2, 3, 4, 5], [4, 5], "2d_to_4d"),
        ([3, 4, 5], [1, 5], "2d_to_3d_with_ones"),
        
        # Edge cases
        ([1], [1], "both_scalar"),
        ([1, 1, 1], [1, 1, 1], "all_ones"),
        ([10], [10], "1d_same"),
        ([1, 5], [3, 1], "both_have_ones"),
        ([2, 1, 4], [1, 3, 1], "multiple_ones_both"),
    ]
    
    # Randomly select a broadcasting scenario
    base_shape, exponent_shape, broadcast_type = random.choice(broadcast_scenarios)
    
    # Generate base data (avoid zeros and negatives for safer computation)
    if base_dtype_np in [np.float32, np.float64, np.float16]:
        base_data = np.random.uniform(0.5, 5.0, base_shape).astype(base_dtype_np)
    else:
        base_data = np.random.randint(1, 10, base_shape).astype(base_dtype_np)
    
    base_tensor = helper.make_tensor(
        input_names[0], 
        base_dtype_proto, 
        base_shape, 
        base_data.flatten().tolist()
    )
    initializers.append(base_tensor)
    
    # Generate exponent data (keep exponents small to avoid overflow)
    if exp_dtype_np in [np.float32, np.float64, np.float16]:
        exponent_data = np.random.uniform(0.5, 3.0, exponent_shape).astype(exp_dtype_np)
    elif exp_dtype_np in [np.int8, np.int16, np.int32, np.int64]:
        exponent_data = np.random.randint(1, 4, exponent_shape).astype(exp_dtype_np)
    else:  # unsigned types
        exponent_data = np.random.randint(1, 4, exponent_shape).astype(exp_dtype_np)
    
    exponent_tensor = helper.make_tensor(
        input_names[1],
        exp_dtype_proto,
        exponent_shape,
        exponent_data.flatten().tolist()
    )
    initializers.append(exponent_tensor)
    
    # Compute output shape using numpy broadcasting rules
    try:
        output_shape = list(np.broadcast_shapes(base_shape, exponent_shape))
    except ValueError as e:
        # This shouldn't happen with our predefined scenarios, but fallback just in case
        print(f"Broadcasting error: {e}, using base_shape as fallback")
        output_shape = base_shape.copy()
        broadcast_type = "fallback_no_broadcast"
    
    # Output has same type as base (X)
    output_info = helper.make_tensor_value_info(
        output_names[0], 
        base_dtype_proto, 
        output_shape
    )
    
    # Create Pow node
    node = helper.make_node(
        "Pow",
        inputs=[input_names[0], input_names[1]],
        outputs=[output_names[0]],
        name=f"Pow_{broadcast_type}_base{base_shape}_exp{exponent_shape}"
    )
    
    # Create dummy input info (required by ONNX graph structure)
    input_info = helper.make_tensor_value_info(
        "useless_input", 
        base_dtype_proto, 
        base_shape
    )
    
    # Metadata for tracking and verification
    metadata = {
        "input_shapes": [base_shape, exponent_shape],
        "output_shapes": [output_shape],
        "base_dtype": str(base_dtype_np),
        "exponent_dtype": str(exp_dtype_np),
        "broadcast_type": broadcast_type,
        "base_data_sample": base_data.flatten()[:5].tolist(),  # First 5 elements for debugging
        "exp_data_sample": exponent_data.flatten()[:5].tolist()
    }
    
    return [input_info], output_info, [node], initializers, metadata