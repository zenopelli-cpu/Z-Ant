import onnx
from onnx import helper, TensorProto
import random
import numpy as np
import argparse

def random_shape(rank, min_dim=1, max_dim=10):
    """Generates a random shape of length 'rank'."""
    return [random.randint(min_dim, max_dim) for _ in range(rank)]

def generate_fuzz_model(op_name):
    """
    Creates inputs, outputs, nodes and any initializers with random parameters 
    for the given op_name.
    """
    initializers = []
    
    if op_name in ["Relu", "Sigmoid", "Ceil", "Tanh", "Identity", "Neg", "Shape"]:
        # Single-input operators with a random shape (rank=4)
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        input_info = [helper.make_tensor_value_info('input0', TensorProto.FLOAT, shape)]
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=['input0'], outputs=['output'])
        return input_info, output_info, [node], initializers

    elif op_name == "LeakyRelu":
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        alpha = round(random.uniform(0.001, 0.2), 3)
        input_info = [helper.make_tensor_value_info('input0', TensorProto.FLOAT, shape)]
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=['input0'], outputs=['output'], alpha=alpha)
        return input_info, output_info, [node], initializers

    elif op_name == "Softmax":
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        rank = len(shape)
        axis = random.randint(-rank, rank-1)
        input_info = [helper.make_tensor_value_info('input0', TensorProto.FLOAT, shape)]
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=['input0'], outputs=['output'], axis=axis)
        return input_info, output_info, [node], initializers

    elif op_name in ["Add", "Sub", "Mul", "Div", "Mean"]:
        # Binary operators: two inputs with the same shape
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        input_info = [
            helper.make_tensor_value_info('input0', TensorProto.FLOAT, shape),
            helper.make_tensor_value_info('input1', TensorProto.FLOAT, shape)
        ]
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=['input0', 'input1'], outputs=['output'])
        return input_info, output_info, [node], initializers

    elif op_name == "Concat":
        # Two inputs with identical shape except along the concatenation axis
        shape = [1, random.randint(2,5), random.randint(10,50), random.randint(10,50)]
        rank = len(shape)
        axis = random.randint(0, rank-1)
        shape2 = shape.copy()
        shape2[axis] = shape[axis] + random.randint(1,3)
        input_info = [
            helper.make_tensor_value_info('input0', TensorProto.FLOAT, shape),
            helper.make_tensor_value_info('input1', TensorProto.FLOAT, shape2)
        ]
        out_shape = shape.copy()
        out_shape[axis] = shape[axis] + shape2[axis]
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=['input0', 'input1'], outputs=['output'], axis=axis)
        return input_info, output_info, [node], initializers

    elif op_name == "Gather":
        # First input: data, second input: indices (provided as an initializer to ensure valid values)
        shape = [5, random.randint(5,10)]
        input_info = [helper.make_tensor_value_info('input0', TensorProto.FLOAT, shape)]
        indices_shape = [random.randint(1,3)]
        input_info.append(helper.make_tensor_value_info('input1', TensorProto.INT64, indices_shape))
        axis = random.randint(0, len(shape)-1)
        indices_data = np.random.randint(0, shape[axis], size=indices_shape).astype(np.int64)
        initializer = helper.make_tensor('input1', TensorProto.INT64, indices_shape, indices_data.flatten().tolist())
        initializers.append(initializer)
        out_shape = list(shape)
        out_shape[axis] = indices_shape[0]
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=['input0', 'input1'], outputs=['output'], axis=axis)
        return input_info, output_info, [node], initializers

    elif op_name == "Pad":
        # Generate random pads for each dimension
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        rank = len(shape)
        pads = [random.randint(0,2) for _ in range(2*rank)]
        out_shape = [shape[i] + pads[i] + pads[i+rank] for i in range(rank)]
        input_info = [helper.make_tensor_value_info('input0', TensorProto.FLOAT, shape)]
        
        # Create pads as a second input tensor (required in newer ONNX versions)
        pads_tensor = helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads)
        initializers.append(pads_tensor)
        input_info.append(helper.make_tensor_value_info('pads', TensorProto.INT64, [len(pads)]))
        
        # Optional constant_value input (using 0.0 as default)
        constant_value = 0.0
        constant_tensor = helper.make_tensor('constant_value', TensorProto.FLOAT, [], [constant_value])
        initializers.append(constant_tensor)
        input_info.append(helper.make_tensor_value_info('constant_value', TensorProto.FLOAT, []))
        
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=['input0', 'pads', 'constant_value'], outputs=['output'])
        return input_info, output_info, [node], initializers

    elif op_name == "Reshape":
        # The second input is an initializer that contains the new shape (e.g., a permutation)
        shape = [random.randint(1,4) for _ in range(4)]
        total_elems = int(np.prod(shape))
        new_shape = shape.copy()
        random.shuffle(new_shape)
        input_info = [helper.make_tensor_value_info('input0', TensorProto.FLOAT, shape)]
        shape_tensor = helper.make_tensor('shape', TensorProto.INT64, [len(new_shape)], new_shape)
        initializers.append(shape_tensor)
        input_info.append(helper.make_tensor_value_info('input1', TensorProto.INT64, [len(new_shape)]))
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, new_shape)
        node = helper.make_node(op_name, inputs=['input0', 'input1'], outputs=['output'])
        return input_info, output_info, [node], initializers

    elif op_name == "Resize":
        # Four inputs: X, roi, scales, sizes
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        input_info = [helper.make_tensor_value_info('input0', TensorProto.FLOAT, shape)]
        roi = []
        roi_tensor = helper.make_tensor('roi', TensorProto.FLOAT, [0], roi)
        initializers.append(roi_tensor)
        input_info.append(helper.make_tensor_value_info('roi', TensorProto.FLOAT, [0]))
        scales = [round(random.uniform(0.5, 2.0), 2) for _ in shape]
        scales_tensor = helper.make_tensor('scales', TensorProto.FLOAT, [len(scales)], scales)
        initializers.append(scales_tensor)
        input_info.append(helper.make_tensor_value_info('scales', TensorProto.FLOAT, [len(scales)]))
        sizes = [int(round(s * dim)) for s, dim in zip(scales, shape)]
        sizes_tensor = helper.make_tensor('sizes', TensorProto.INT64, [len(sizes)], sizes)
        initializers.append(sizes_tensor)
        input_info.append(helper.make_tensor_value_info('sizes', TensorProto.INT64, [len(sizes)]))
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, sizes)
        mode = random.choice(["nearest", "linear"])
        node = helper.make_node(op_name, inputs=['input0', 'roi', 'scales', 'sizes'], outputs=['output'], mode=mode)
        return input_info, output_info, [node], initializers

    elif op_name == "Slice":
        # Create random slice indices for each dimension
        shape = [random.randint(5,10) for _ in range(4)]
        rank = len(shape)
        starts, ends = [], []
        for d in shape:
            start = random.randint(0, d-1)
            end = random.randint(start+1, d)
            starts.append(start)
            ends.append(end)
        
        # Create input tensors for starts and ends
        starts_tensor = helper.make_tensor('starts', TensorProto.INT64, [len(starts)], starts)
        ends_tensor = helper.make_tensor('ends', TensorProto.INT64, [len(ends)], ends)
        initializers.append(starts_tensor)
        initializers.append(ends_tensor)
        
        out_shape = [ends[i] - starts[i] for i in range(rank)]
        input_info = [helper.make_tensor_value_info('input0', TensorProto.FLOAT, shape)]
        input_info.append(helper.make_tensor_value_info('starts', TensorProto.INT64, [len(starts)]))
        input_info.append(helper.make_tensor_value_info('ends', TensorProto.INT64, [len(ends)]))
        
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=['input0', 'starts', 'ends'], outputs=['output'])
        return input_info, output_info, [node], initializers

    elif op_name == "Split":
        # Split into 2 parts along a random axis
        shape = [1, random.randint(4,10), random.randint(10,50), random.randint(10,50)]
        rank = len(shape)
        axis = random.randint(0, rank-1)
        if shape[axis] < 2:
            shape[axis] = 2
        input_info = [helper.make_tensor_value_info('input0', TensorProto.FLOAT, shape)]
        out_shape = shape.copy()
        out_shape[axis] = shape[axis] // 2
        output_info = [
            helper.make_tensor_value_info('output0', TensorProto.FLOAT, out_shape),
            helper.make_tensor_value_info('output1', TensorProto.FLOAT, out_shape)
        ]
        node = helper.make_node(op_name, inputs=['input0'], outputs=['output0', 'output1'], axis=axis)
        return input_info, output_info, [node], initializers

    elif op_name == "Transpose":
        # Generate a random permutation for Transpose
        shape = [random.randint(1,4) for _ in range(4)]
        rank = len(shape)
        input_info = [helper.make_tensor_value_info('input0', TensorProto.FLOAT, shape)]
        perm = list(range(rank))
        random.shuffle(perm)
        out_shape = [shape[i] for i in perm]
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=['input0'], outputs=['output'], perm=perm)
        return input_info, output_info, [node], initializers

    elif op_name == "Unsqueeze":
        # Insert a new dimension at a random axis
        shape = [random.randint(1,4) for _ in range(4)]
        rank = len(shape)
        input_info = [helper.make_tensor_value_info('input0', TensorProto.FLOAT, shape)]
        
        # Create axes as a second input tensor
        axis = random.randint(0, rank)
        axes = [axis]
        axes_tensor = helper.make_tensor('axes', TensorProto.INT64, [len(axes)], axes)
        initializers.append(axes_tensor)
        input_info.append(helper.make_tensor_value_info('axes', TensorProto.INT64, [len(axes)]))
        
        out_shape = shape.copy()
        out_shape.insert(axis, 1)
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=['input0', 'axes'], outputs=['output'])
        return input_info, output_info, [node], initializers

    elif op_name == "Conv":
        # For Conv, generate an input shape [N, C, H, W] with H and W sufficiently large
        N = 1
        C = random.randint(1,4)
        H = random.randint(10,50)
        W = random.randint(10,50)
        input_shape = [N, C, H, W]
        kH = random.randint(2, max(2, H//2))
        kW = random.randint(2, max(2, W//2))
        kernel_shape = [kH, kW]
        M = random.randint(1,4)
        weight_shape = [M, C, kH, kW]
        input_info = [
            helper.make_tensor_value_info('input0', TensorProto.FLOAT, input_shape),
            helper.make_tensor_value_info('input1', TensorProto.FLOAT, weight_shape)
        ]
        H_out = H - kH + 1
        W_out = W - kW + 1
        output_shape = [N, M, H_out, W_out]
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        node = helper.make_node(op_name, inputs=['input0', 'input1'], outputs=['output'], kernel_shape=kernel_shape)
        return input_info, output_info, [node], initializers

    elif op_name == "MatMul":
        # Generate two compatible 2D matrices
        M = random.randint(2,10)
        K = random.randint(2,10)
        N = random.randint(2,10)
        A_shape = [M, K]
        B_shape = [K, N]
        input_info = [
            helper.make_tensor_value_info('input0', TensorProto.FLOAT, A_shape),
            helper.make_tensor_value_info('input1', TensorProto.FLOAT, B_shape)
        ]
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, [M, N])
        node = helper.make_node(op_name, inputs=['input0', 'input1'], outputs=['output'])
        return input_info, output_info, [node], initializers

    elif op_name == "Gemm":
        # Performs A * B + C with compatible shapes
        M = random.randint(2,10)
        K = random.randint(2,10)
        N = random.randint(2,10)
        A_shape = [M, K]
        B_shape = [K, N]
        C_shape = [M, N]
        input_info = [
            helper.make_tensor_value_info('input0', TensorProto.FLOAT, A_shape),
            helper.make_tensor_value_info('input1', TensorProto.FLOAT, B_shape),
            helper.make_tensor_value_info('input2', TensorProto.FLOAT, C_shape)
        ]
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, [M, N])
        alpha = round(random.uniform(0.5, 2.0), 2)
        beta = round(random.uniform(0.5, 2.0), 2)
        transA = random.choice([0, 1])
        transB = random.choice([0, 1])
        node = helper.make_node(op_name, inputs=['input0', 'input1', 'input2'], outputs=['output'],
                                alpha=alpha, beta=beta, transA=transA, transB=transB)
        return input_info, output_info, [node], initializers

    elif op_name == "MaxPool":
        # Generate a pooling layer with random kernel and stride
        N = 1
        C = random.randint(1,4)
        H = random.randint(10,50)
        W = random.randint(10,50)
        input_shape = [N, C, H, W]
        kernel_size = random.randint(2, max(2, min(H, W)//2))
        kernel_shape = [kernel_size, kernel_size]
        strides = [random.randint(1, kernel_size), random.randint(1, kernel_size)]
        H_out = (H - kernel_size) // strides[0] + 1
        W_out = (W - kernel_size) // strides[1] + 1
        output_shape = [N, C, H_out, W_out]
        input_info = [helper.make_tensor_value_info('input0', TensorProto.FLOAT, input_shape)]
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        node = helper.make_node(op_name, inputs=['input0'], outputs=['output'],
                                kernel_shape=kernel_shape, strides=strides)
        return input_info, output_info, [node], initializers

    else:
        # Fallback for any operators not explicitly handled
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        input_info = [helper.make_tensor_value_info('input0', TensorProto.FLOAT, shape)]
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=['input0'], outputs=['output'])
        return input_info, output_info, [node], initializers

def generate_model(op_name, filename):
    input_info, output_info, nodes, initializers = generate_fuzz_model(op_name)
    # If there are multiple outputs, pass them as a list
    graph_outputs = output_info if isinstance(output_info, list) else [output_info]
    graph = helper.make_graph(
        nodes,
        name=f"{op_name}_graph",
        inputs=input_info,
        outputs=graph_outputs,
        initializer=initializers
    )
    model = helper.make_model(graph, producer_name='fuzz_generator')
    onnx.checker.check_model(model)
    onnx.save(model, filename)
    print(f"Fuzzed model for {op_name} saved to: {filename}")

def load_supported_ops(filename="available_operations.txt"):
    try:
        with open(filename, "r") as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using default operations.")
        return []

def main():
    parser = argparse.ArgumentParser(description="Generate fuzzed ONNX models for CI/CD.")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of models to generate for each operation.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for random generation (for reproducibility).")
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        
    supported_ops = load_supported_ops() #TODO : Softmax has errors in parsing, it has been removed from available_operations.txt
    if not supported_ops:
        supported_ops = [  # Fallback default operations
            "LeakyRelu", "Relu", "Sigmoid", "Softmax", "Add", "Ceil", "Div", "Mul", "Sub", "Tanh",
            "Concat", "Gather", "Identity", "Neg", "Reshape", "Resize", "Shape", "Slice", 
            "Split", "Transpose", "Unsqueeze", "Mean", "Conv", "MatMul", "Gemm", "MaxPool"
        ]
    
    for op in supported_ops:
        for i in range(args.iterations):
            filename = f"{op}_{i}.onnx"
            generate_model(op, filename)

if __name__ == "__main__":
    main()


# LeakyRelu
# Relu
# Sigmoid
# Add
# Ceil
# Div
# Mul
# Sub
# Tanh
# Concat
# Gather
# Identity
# Neg
# Reshape
# Resize
# Shape
# Slice
# Split
# Transpose
# Unsqueeze
# Mean
# Conv
# MatMul
# Gemm
# MaxPool