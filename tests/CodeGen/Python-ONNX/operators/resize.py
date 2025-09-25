import numpy as np
import random
from onnx import helper, TensorProto

def generate_resize_model(input_names, output_names):
    """
    Generates a Resize operator model according to ONNX spec.
    """
    initializers = []
    
    # Generate a random input tensor shape: (N=1, C=random, H=random, W=random)
    shape = [1, random.randint(1, 4), random.randint(10, 50), random.randint(10, 50)]
    data = np.random.randn(*shape).astype(np.float32)
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.ravel())
    initializers.append(init_tensor)
    
    # Choose resize mode and coordinate transformation mode
    mode = random.choice(["nearest", "linear", "cubic"])
    coordinate_transformation_mode = random.choice([
        "half_pixel", "half_pixel_symmetric", "pytorch_half_pixel", "align_corners", 
        "asymmetric", "tf_crop_and_resize"
    ])
    
    # Choose whether to use scales or sizes (mutually exclusive per spec)
    use_scales = random.choice([True, False])
    
    # Handle ROI tensor - only used with tf_crop_and_resize
    roi_name = input_names[1] + "_roi"
    if coordinate_transformation_mode == "tf_crop_and_resize":
        # ROI format: [start1, start2, ..., startN, end1, end2, ..., endN]
        # For 4D tensor: [start_batch, start_channel, start_h, start_w, end_batch, end_channel, end_h, end_w]
        roi_data = [
            0.0, 0.0, 0.0, 0.0,  # start values (normalized coordinates)
            1.0, 1.0, 1.0, 1.0   # end values (normalized coordinates)
        ]
        roi_tensor = helper.make_tensor(roi_name, TensorProto.FLOAT, [len(roi_data)], roi_data)
        initializers.append(roi_tensor)

    
    if use_scales:
        # Using scales - sizes should be empty
        # Scales must have same number of elements as input rank OR length of axes if provided
        scale_h = round(random.uniform(0.5, 2.0), 3)
        scale_w = round(random.uniform(0.5, 2.0), 3)
        scales = [1.0, 1.0, scale_h, scale_w]  # Don't scale batch and channel dims
        
        scales_name = input_names[2] + "_scales"
        scales_tensor = helper.make_tensor(scales_name, TensorProto.FLOAT, [len(scales)], scales)
        initializers.append(scales_tensor)
        
        # Empty sizes tensor when using scales (per spec: only one can be specified)
        sizes_name = ""  # Use empty string name to indicate unused input
        # Don't add sizes tensor to initializers when using scales
        
        # Calculate output shape from scales
        output_shape = [
            shape[0],  # batch size unchanged
            shape[1],  # channels unchanged
            max(1, int(shape[2] * scale_h)),  # scaled height (ensure minimum 1)
            max(1, int(shape[3] * scale_w))   # scaled width (ensure minimum 1)
        ]
        
        metadata_extra = {"scales": scales, "scale_h": scale_h, "scale_w": scale_w}
        
    else:
        # Empty scales tensor when using sizes
        scales_name = ""  # Use empty string name to indicate unused input
        # Don't add scales tensor to initializers when using sizes
        
        # Generate target sizes for all dimensions
        # Sizes must have same number of elements as input rank OR length of axes if provided
        new_height = random.randint(5, 100)
        new_width = random.randint(5, 100)
        sizes = [shape[0], shape[1], new_height, new_width]  # All 4 dimensions
        
        sizes_name = input_names[3] + "_sizes"
        sizes_tensor = helper.make_tensor(sizes_name, TensorProto.INT64, [len(sizes)], sizes)
        initializers.append(sizes_tensor)
        
        output_shape = sizes
        metadata_extra = {"sizes": sizes, "new_height": new_height, "new_width": new_width}
    
    # Output tensor info
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
    
    # Node attributes
    node_attrs = {
        "mode": mode,
        "coordinate_transformation_mode": coordinate_transformation_mode,
    }
    
    # Add mode-specific attributes based on spec
    if mode == "nearest":
        nearest_mode = random.choice(["round_prefer_floor", "round_prefer_ceil", "floor", "ceil"])
        node_attrs["nearest_mode"] = nearest_mode
        metadata_extra["nearest_mode"] = nearest_mode
        
    elif mode == "cubic":
        # Cubic coefficient 'a' - spec mentions -0.5 (TensorFlow) and -0.75 (PyTorch)
        cubic_coeff_a = random.choice([-0.5, -0.75])
        node_attrs["cubic_coeff_a"] = cubic_coeff_a
        metadata_extra["cubic_coeff_a"] = cubic_coeff_a
    
    # Antialiasing for linear and cubic modes when downscaling
    if mode in ["linear", "cubic"]:
        antialias = random.choice([0, 1])
        node_attrs["antialias"] = antialias
        metadata_extra["antialias"] = antialias
    
    # Attributes specific to tf_crop_and_resize
    if coordinate_transformation_mode == "tf_crop_and_resize":
        exclude_outside = random.choice([0, 1])
        extrapolation_value = round(random.uniform(-1.0, 1.0), 2)
        node_attrs["exclude_outside"] = exclude_outside
        node_attrs["extrapolation_value"] = extrapolation_value
        metadata_extra["exclude_outside"] = exclude_outside
        metadata_extra["extrapolation_value"] = extrapolation_value
    
    # Keep aspect ratio policy (only applies when using sizes, not scales)
    if not use_scales:
        keep_aspect_ratio_policy = random.choice(["stretch", "not_larger", "not_smaller"])
        node_attrs["keep_aspect_ratio_policy"] = keep_aspect_ratio_policy
        metadata_extra["keep_aspect_ratio_policy"] = keep_aspect_ratio_policy
    
    # Create ONNX Resize node with proper input handling
    node_inputs = [input_names[0], roi_name]
    
    if use_scales:
        node_inputs.extend([scales_name, ""])  # scales provided, sizes empty
    else:
        node_inputs.extend(["", sizes_name])   # scales empty, sizes provided
    
    node = helper.make_node(
        "Resize",
        inputs=node_inputs,
        outputs=[output_names[0]],
        name=f"Resize_{mode}_{coordinate_transformation_mode}",
        **node_attrs
    )
    
    # Input info placeholder (not actually used since input is initializer)
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    
    # Comprehensive metadata dictionary
    metadata = {
        "input_shapes": [shape],
        "output_shapes": [output_shape],
        "mode": mode,
        "coordinate_transformation_mode": coordinate_transformation_mode,
        "use_scales": use_scales,
        **metadata_extra
    }
    
    return [input_info], output_info, [node], initializers, metadata