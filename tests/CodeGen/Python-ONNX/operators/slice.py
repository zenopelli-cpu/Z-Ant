import numpy as np
import random
from onnx import helper, TensorProto


def generate_slice_model(input_names, output_names):
    """
    Generates a Slice operator model.
    """
    initializers = []
    
    # Primo input: dati; gli altri due (starts ed ends) come initializer
    shape = [random.randint(5,10) for _ in range(4)]
    data = np.random.randn(*shape).astype(np.float32)
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    initializers.append(init_tensor)
    rank = len(shape)
    starts, ends = [], []
    for d in shape:
        start = random.randint(0, d-1)
        end = random.randint(start+1, d)
        starts.append(start)
        ends.append(end)
    starts_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(starts)], starts)
    ends_tensor = helper.make_tensor(input_names[2], TensorProto.INT64, [len(ends)], ends)
    initializers.extend([starts_tensor, ends_tensor])
    out_shape = [ends[i] - starts[i] for i in range(rank)]
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
    node = helper.make_node("Slice", inputs=[input_names[0], input_names[1], input_names[2]],
                            outputs=[output_names[0]], name=f"Slice_node")
    
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "starts": starts, "ends": ends}
    return [input_info], output_info, [node], initializers, metadata