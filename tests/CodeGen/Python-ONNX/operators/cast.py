import numpy as np
import random
from onnx import helper, TensorProto


def generate_cast_model(input_names, output_names):
    """
    Generates a Cast operator model.
    """
    initializers = []
    
    # Cast operator: converts tensor from one type to another
    shape = [1, random.randint(1,4), random.randint(5,25), random.randint(5,25)]
    
    # Choose input and output types
    input_types = [
        (TensorProto.FLOAT, np.float32),
        (TensorProto.INT32, np.int32),
        (TensorProto.INT64, np.int64),
        (TensorProto.UINT8, np.uint8),
        (TensorProto.INT8, np.int8)
    ]
    
    output_types = [
        TensorProto.FLOAT,
        TensorProto.INT32,
        TensorProto.INT64,
        TensorProto.UINT8,
        TensorProto.INT8
    ]
    
    input_proto_type, input_np_type = random.choice(input_types)
    output_proto_type = random.choice(output_types)
    
    # Generate input data based on input type
    if input_proto_type == TensorProto.FLOAT:
        data = np.random.randn(*shape).astype(input_np_type)
    elif input_proto_type in [TensorProto.INT32, TensorProto.INT64]:
        data = np.random.randint(-100, 100, size=shape).astype(input_np_type)
    else:  # UINT8, INT8
        if input_proto_type == TensorProto.UINT8:
            data = np.random.randint(0, 256, size=shape).astype(input_np_type)
        else:  # INT8
            data = np.random.randint(-128, 128, size=shape).astype(input_np_type)
    
    init_tensor = helper.make_tensor(input_names[0], input_proto_type, shape, data.flatten().tolist())
    initializers.append(init_tensor)
    
    input_info = helper.make_tensor_value_info("useless_input", input_proto_type, shape)
    output_info = helper.make_tensor_value_info(output_names[0], output_proto_type, shape)
    
    node = helper.make_node(
        "Cast",
        inputs=[input_names[0]],
        outputs=[output_names[0]],
        to=output_proto_type,
        name=f"Cast_node_{input_proto_type}_to_{output_proto_type}"
    )
    
    metadata = {
        "input_shapes": [shape],
        "output_shapes": [shape],
        "input_type": input_proto_type,
        "output_type": output_proto_type
    }
    
    return [input_info], output_info, [node], initializers, metadata