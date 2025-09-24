import numpy as np
import random
from onnx import helper, TensorProto


def generate_globalaveragepool_model(input_names, output_names):
    """
    Generates a GlobalAveragePool operator model.
    """
    initializers = []
    
    # GlobalAveragePool operator: applies average pooling across all spatial dimensions
    # Input shape: (N, C, H, W, ...) where N=batch, C=channels, H,W,...=spatial dims
    # Output shape: (N, C, 1, 1, ...) where all spatial dimensions become 1
    
    N = random.randint(1, 2)  # Batch size (smaller for compatibility)
    C = random.randint(1, 3)  # Number of channels (smaller for compatibility)
    
    # Most ONNX Runtime implementations work well with 2D spatial dimensions (H, W)
    # Let's focus on 2D case which is most common and well-supported
    spatial_config = random.choice([
        "2D_small",   # Small 2D: typical for testing
        "2D_medium",  # Medium 2D: more realistic
        "3D_small"    # Small 3D: for advanced testing
    ])
    
    if spatial_config == "2D_small":
        spatial_dims = [random.randint(2, 4), random.randint(2, 4)]  # H, W
    elif spatial_config == "2D_medium":
        spatial_dims = [random.randint(3, 6), random.randint(3, 6)]  # H, W
    else:  # 3D_small
        spatial_dims = [random.randint(2, 3), random.randint(2, 3), random.randint(2, 3)]  # H, W, D
    
    # Complete input shape: [N, C, spatial_dims...]
    input_shape = [N, C] + spatial_dims
    
    # Create input data with predictable values for testing
    # Use smaller range to avoid numerical issues
    total_elements = np.prod(input_shape)
    if total_elements > 1000:  # Avoid very large tensors
        # Fallback to smaller dimensions
        spatial_dims = [3, 3]  # Simple 3x3 spatial
        input_shape = [N, C] + spatial_dims
        total_elements = np.prod(input_shape)
    
    # Generate data in a reasonable range
    data = np.random.uniform(0.1, 10.0, input_shape).astype(np.float32)
    
    # Alternative: use sequential data for predictable testing
    if random.choice([True, False]):
        data = np.arange(1, total_elements + 1, dtype=np.float32).reshape(input_shape)
        data = data / total_elements  # Normalize to avoid large numbers
    
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, input_shape, data.flatten().tolist())
    initializers.append(init_tensor)
    
    # Output shape: same as input but all spatial dimensions become 1
    output_shape = [N, C] + [1] * len(spatial_dims)
    
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
    
    # GlobalAveragePool has no attributes according to ONNX specification
    node = helper.make_node(
        "GlobalAveragePool", 
        inputs=[input_names[0]], 
        outputs=[output_names[0]],
        name=f"GlobalAveragePool_node_N{N}_C{C}_config{spatial_config}"
    )
    
    metadata = {
        "input_shapes": [input_shape],
        "output_shapes": [output_shape],
        "batch_size": int(N),  # Convert to Python int
        "channels": int(C),    # Convert to Python int
        "spatial_dimensions": [int(dim) for dim in spatial_dims],  # Convert list elements
        "spatial_config": spatial_config,
        "total_elements": int(total_elements)  # Convert to Python int
    }
    
    return [input_info], output_info, [node], initializers, metadata