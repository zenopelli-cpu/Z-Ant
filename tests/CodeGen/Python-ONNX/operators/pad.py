import numpy as np
import random
from onnx import helper, TensorProto

def generate_pad_model(input_names, output_names):
    """
    Generates a Pad operator model following ONNX specification exactly.
    """
    initializers = []
    
    # Generate input shape - keep reasonable sizes for testing
    shape = [1, random.randint(1,3), random.randint(8,20), random.randint(8,20)]
    data = np.random.randn(*shape).astype(np.float32)
    
    # Create input tensor
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    initializers.append(init_tensor)
    
    rank = len(shape)
    
    # Choose padding mode - be conservative with reflect mode
    mode = random.choice(["constant", "edge",]) # ---> reflect has been removed due to ORT issues: https://github.com/microsoft/onnxruntime/issues/16401
    
    # Generate pads more carefully based on mode and ONNX constraints
    if mode == "reflect": 
        # For reflect mode, padding should be reasonable relative to dimension size
        # ONNX reflect mode has issues with large padding relative to dimension size
        pads = []
        for i in range(rank):
            dim_size = shape[i]
            max_pad = min(2, max(1, dim_size // 4))  # Conservative padding for reflect
            begin_pad = random.randint(0, max_pad)
            end_pad = random.randint(0, max_pad)
            pads.extend([begin_pad, end_pad])
        
        # Reorder to ONNX format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        begin_pads = pads[::2]  # Extract begin pads
        end_pads = pads[1::2]   # Extract end pads
        pads = begin_pads + end_pads
    else:
        # For constant and edge modes, we can be more liberal with padding
        pads = []
        for i in range(rank):
            begin_pad = random.randint(0, 3)
            end_pad = random.randint(0, 3)
            pads.append(begin_pad)
        for i in range(rank):
            end_pad = random.randint(0, 3)
            pads.append(end_pad)
    
    # Calculate output shape using the corrected pads format
    out_shape = [shape[i] + pads[i] + pads[i + rank] for i in range(rank)]
    
    # Ensure all output dimensions are positive
    for i, dim in enumerate(out_shape):
        if dim <= 0:
            # Fallback: reduce padding to ensure positive dimensions
            total_pad_needed = 1 - shape[i]  # Minimum to get dimension = 1
            pads[i] = max(0, total_pad_needed // 2)  # begin pad
            pads[i + rank] = max(0, total_pad_needed - pads[i])  # end pad
            out_shape[i] = shape[i] + pads[i] + pads[i + rank]
    
    # Create pads tensor - MUST be int64 according to ONNX spec
    pads_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(pads)], pads)
    initializers.append(pads_tensor)
    
    # Set up node inputs
    node_inputs = [input_names[0], input_names[1]]
    constant_value = None
    
    # Add constant value tensor if mode is constant
    if mode == "constant":
        constant_value = round(random.uniform(-2.0, 2.0), 2)
        # Create scalar tensor for constant value
        constant_tensor = helper.make_tensor(
            input_names[2], 
            TensorProto.FLOAT, 
            [],  # Scalar shape
            [constant_value]
        )
        initializers.append(constant_tensor)
        node_inputs.append(input_names[2])
    
    # Create input and output info
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
    
    # Create the Pad node - CRITICAL: mode is an attribute, not input
    node = helper.make_node(
        "Pad",
        inputs=node_inputs,
        outputs=[output_names[0]],
        mode=mode,  # This is an attribute, not an input!
        name=f"Pad_node_mode_{mode}"
    )
    
    metadata = {
        "input_shapes": [shape],
        "output_shapes": [out_shape], 
        "pads": pads,
        "mode": mode,
        "axes": None,  # Not using axes in this generator
        "constant_value": constant_value,
        "onnx_version": "v21+",
        "notes": f"Pads format: [x1_begin, x2_begin, x3_begin, x4_begin, x1_end, x2_end, x3_end, x4_end] = {pads}"
    }
    
    return [input_info], output_info, [node], initializers, metadata