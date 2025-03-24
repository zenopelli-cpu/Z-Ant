import onnx
from onnx import helper, TensorProto
import random
import numpy as np
import argparse
import datetime
from onnx import StringStringEntryProto
import onnxruntime as ort  
import json  
import os  

def random_shape(rank, min_dim=1, max_dim=10):
    """Generates a random shape of length 'rank'."""
    return [random.randint(min_dim, max_dim) for _ in range(rank)]

def generate_fuzz_model(op_name):
    """
    Crea gli initializer, i nodi e gli output con parametri casuali per l'operatore op_name.
    In questo caso tutti gli input vengono inseriti come initializer, per cui la lista degli input
    del grafo viene restituita vuota.
    """
    initializers = []
    
    # Pre-generazione di nomi per input e output
    input_names = [f"{op_name}_param_in_{i}" for i in range(5)]
    output_names = [f"{op_name}_param_out_{i}" for i in range(5)]
    metadata = {}

    if op_name in ["Relu", "Sigmoid", "Ceil", "Tanh", "Identity", "Neg", "Shape"]:
        # Operatori a singolo input con forma casuale (rank=4)
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        # Crea dati casuali e li inserisce come initializer
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)

        input_info = helper.make_tensor_value_info( "useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]], 
                                name=f"{op_name}_node")
        metadata = {"input_shapes": [shape], "output_shapes": [shape]}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "LeakyRelu":
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        alpha = round(random.uniform(0.001, 0.2), 3)
        data = np.random.randn(*shape).astype(np.float32)

        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)

        input_info = helper.make_tensor_value_info( "useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]], 
                                alpha=alpha, name=f"{op_name}node_alpha{alpha}")
        metadata = {"input_shapes": [shape], "output_shapes": [shape], "alpha": alpha}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Softmax":
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        rank = len(shape)
        axis = random.randint(-rank, rank-1)
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]], 
                                axis=axis, name=f"{op_name}node_axis{axis}")
        
        # Define input_info before using it
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [shape], "axis": axis}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name in ["Add", "Sub", "Div", "Mean", "Mul"]:
        # Operatori binari: due input della stessa forma
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        data0 = np.random.randn(*shape).astype(np.float32)
        data1 = np.random.randn(*shape).astype(np.float32)

        init_tensor0 = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data0.flatten().tolist())
        init_tensor1 = helper.make_tensor(input_names[1], TensorProto.FLOAT, shape, data1.flatten().tolist())
        initializers.extend([init_tensor0, init_tensor1])

        input_info = helper.make_tensor_value_info( "useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

        node = helper.make_node(op_name, 
                                inputs=[input_names[0], 
                                input_names[1]], 
                                outputs=[output_names[0]],
                                name=f"{op_name}_node")
        
        metadata = {"input_shapes": [shape, shape], "output_shapes": [shape]}
        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "Concat":
        # Due input con forma identica eccetto per la dimensione lungo l'asse di concatenazione
        shape = [1, random.randint(2,5), random.randint(10,50), random.randint(10,50)]
        rank = len(shape)
        axis = random.randint(0, rank-1)
        shape2 = shape.copy()
        shape2[axis] = shape[axis] + random.randint(1,3)
        data0 = np.random.randn(*shape).astype(np.float32)
        data1 = np.random.randn(*shape2).astype(np.float32)
        init_tensor0 = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data0.flatten().tolist())
        init_tensor1 = helper.make_tensor(input_names[1], TensorProto.FLOAT, shape2, data1.flatten().tolist())
        initializers.extend([init_tensor0, init_tensor1])
        out_shape = shape.copy()
        out_shape[axis] = shape[axis] + shape2[axis]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                                axis=axis, name=f"{op_name}node_axis{axis}")
        
        # Define input_info before using it
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape, shape2], "output_shapes": [out_shape]}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Gather":
        # First input: data; second input: indices
        shape = [5, random.randint(5,10)]
        data = np.random.randn(*shape).astype(np.float32)
        init_data = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_data)
        
        # Pick a random axis
        axis = random.randint(0, len(shape)-1)
        
        # Ensure indices are within bounds of the chosen axis
        max_index = shape[axis] - 1
        indices_shape = [random.randint(1,3)]
        indices_data = np.random.randint(0, max_index + 1, size=indices_shape).astype(np.int64)
        init_indices = helper.make_tensor(input_names[1], TensorProto.INT64, indices_shape, indices_data.flatten().tolist())
        initializers.append(init_indices)
        
        # Calculate output shape
        out_shape = list(shape)
        out_shape[axis] = indices_shape[0]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                              axis=axis, name=f"{op_name}node_axis{axis}")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {
            "input_shapes": [shape, indices_shape], 
            "output_shapes": [out_shape],
            "axis": axis,
            "indices": indices_data.tolist()
        }
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Pad":
        # Operatore Pad: genera dati, pads e constant_value come initializer
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        rank = len(shape)
        pads = [random.randint(0,2) for _ in range(2*rank)]
        out_shape = [shape[i] + pads[i] + pads[i+rank] for i in range(rank)]
        pads_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(pads)], pads)
        initializers.append(pads_tensor)
        constant_value = 0.0
        constant_tensor = helper.make_tensor(input_names[2], TensorProto.FLOAT, [], [constant_value])
        initializers.append(constant_tensor)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1], input_names[2]], outputs=[output_names[0]],
                                name=f"{op_name}_node")
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "pads": pads, "constant_value": constant_value}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Reshape":
        # Primo input: dati; secondo input: nuovo shape (initializer)
        shape = [random.randint(1,4) for _ in range(4)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        
        # Define input_info before using it
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        
        new_shape = shape.copy()
        random.shuffle(new_shape)
        shape_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(new_shape)], new_shape)
        initializers.append(shape_tensor)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, new_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                                name=f"{op_name}_node")
        
        metadata = {"input_shapes": [shape, new_shape], "output_shapes": [new_shape]}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Resize":
        # Quattro input: X, roi, scales, sizes
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        
        # Empty ROI tensor
        roi = []
        roi_name = input_names[1] + "roi"
        roi_tensor = helper.make_tensor(roi_name, TensorProto.FLOAT, [0], roi)
        initializers.append(roi_tensor)
        
        # Use empty scales tensor
        scales = []
        scales_name = input_names[2] + "scales"
        scales_tensor = helper.make_tensor(scales_name, TensorProto.FLOAT, [0], scales)
        initializers.append(scales_tensor)
        
        # Use only sizes, not scales
        sizes = [shape[0], shape[1], 
                 int(shape[2] * round(random.uniform(0.5, 2.0), 2)),
                 int(shape[3] * round(random.uniform(0.5, 2.0), 2))]
        sizes_name = input_names[3] + "sizes"
        sizes_tensor = helper.make_tensor(sizes_name, TensorProto.INT64, [len(sizes)], sizes)
        initializers.append(sizes_tensor)
        
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, sizes)
        mode = random.choice(["nearest", "linear"])
        node = helper.make_node(op_name, inputs=[input_names[0], roi_name, scales_name, sizes_name], 
                                outputs=[output_names[0]], mode=mode, name=f"{op_name}node_mode{mode}")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [sizes], "mode": mode, "sizes": sizes}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Slice":
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
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1], input_names[2]],
                                outputs=[output_names[0]], name=f"{op_name}_node")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "starts": starts, "ends": ends}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Split":
        # Split in 2 parti lungo un asse casuale
        shape = [1, random.randint(4,10), random.randint(10,50), random.randint(10,50)]
        axis = random.randint(0, len(shape)-1)
        
        # Ensure the dimension at the chosen axis is even
        if shape[axis] % 2 != 0:
            shape[axis] += 1
            
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        
        out_shape = shape.copy()
        out_shape[axis] = shape[axis] // 2
        output_info = [
            helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape),
            helper.make_tensor_value_info(output_names[1], TensorProto.FLOAT, out_shape)
        ]
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0], output_names[1]],
                                axis=axis, name=f"{op_name}node_axis{axis}")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape, out_shape], "axis": axis}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Transpose":
        # Genera una permutazione casuale per Transpose
        shape = [random.randint(1,4) for _ in range(4)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        rank = len(shape)
        perm = list(range(rank))
        random.shuffle(perm)
        out_shape = [shape[i] for i in perm]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]],
                                perm=perm, name=f"{op_name}node_perm{perm}")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "perm": perm}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Unsqueeze":
        # Inserisce una dimensione in un asse casuale
        shape = [random.randint(1,4) for _ in range(4)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        rank = len(shape)
        axis = random.randint(0, rank)
        axes = [axis]
        axes_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(axes)], axes)
        initializers.append(axes_tensor)
        out_shape = shape.copy()
        out_shape.insert(axis, 1)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                                name=f"{op_name}node_axis{axis}")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "axes": axes}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Conv":
        # Operatore Conv: genera input e pesi come initializer
        N = 1
        C = random.randint(1,4)
        H = random.randint(10,50)
        W = random.randint(10,50)
        input_shape = [N, C, H, W]
        data = np.random.randn(*input_shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, input_shape, data.flatten().tolist())
        initializers.append(init_tensor)
        kH = random.randint(2, max(2, H//2))
        kW = random.randint(2, max(2, W//2))
        kernel_shape = [kH, kW]
        M = random.randint(1,4)
        weight_shape = [M, C, kH, kW]
        weight_data = np.random.randn(*weight_shape).astype(np.float32)
        init_weight = helper.make_tensor(input_names[1], TensorProto.FLOAT, weight_shape, weight_data.flatten().tolist())
        initializers.append(init_weight)
        
        # Add strides parameter
        strides = [random.randint(1, 3), random.randint(1, 3)]
        
        # Add dilations parameter
        dilations = [random.randint(1, 2), random.randint(1, 2)]
        
        # Add padding parameter (padding for begin and end of each spatial axis)
        # Format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        pad_h = random.randint(0, 2)
        pad_w = random.randint(0, 2)
        pads = [0, 0, pad_h, pad_w, 0, 0, pad_h, pad_w]  # Padding for 4D tensor [N,C,H,W]
        
        # Calculate output dimensions with strides, dilations and padding
        # Formula: output_size = (input_size + pad_begin + pad_end - (kernel_size-1)*dilation - 1) / stride + 1
        H_out = (H + 2*pad_h - (kH-1)*dilations[0] - 1) // strides[0] + 1
        W_out = (W + 2*pad_w - (kW-1)*dilations[1] - 1) // strides[1] + 1
        
        # Ensure valid output dimensions
        if H_out <= 0 or W_out <= 0:
            # Fallback to simpler case if dimensions don't work out
            dilations = [1, 1]
            pad_h = pad_w = 1
            pads = [0, 0, pad_h, pad_w, 0, 0, pad_h, pad_w]
            H_out = (H + 2*pad_h - kH) // strides[0] + 1
            W_out = (W + 2*pad_w - kW) // strides[1] + 1
        
        output_shape = [N, M, H_out, W_out]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
        
        # Use pads attribute in the node (simplified to just the 4 values for H,W dimensions)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                                kernel_shape=kernel_shape, strides=strides, dilations=dilations,
                                pads=[pad_h, pad_w, pad_h, pad_w],
                                name=f"{op_name}node_k{kernel_shape}_s{strides}_d{dilations}_p{[pad_h, pad_w]}")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
        metadata = {"input_shapes": [input_shape, weight_shape], "output_shapes": [output_shape],
                    "kernel_shape": kernel_shape, "strides": strides, "dilations": dilations, 
                    "pads": [pad_h, pad_w, pad_h, pad_w]}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "MatMul":
        # Genera due matrici 2D compatibili
        M_val = random.randint(2,10)
        K_val = random.randint(2,10)
        N_val = random.randint(2,10)
        A_shape = [M_val, K_val]
        B_shape = [K_val, N_val]
        A_data = np.random.randn(*A_shape).astype(np.float32)
        B_data = np.random.randn(*B_shape).astype(np.float32)
        init_tensor_A = helper.make_tensor(input_names[0], TensorProto.FLOAT, A_shape, A_data.flatten().tolist())
        init_tensor_B = helper.make_tensor(input_names[1], TensorProto.FLOAT, B_shape, B_data.flatten().tolist())
        initializers.extend([init_tensor_A, init_tensor_B])
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, [M_val, N_val])
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                                name=f"{op_name}_node")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, A_shape)
        metadata = {"input_shapes": [A_shape, B_shape], "output_shapes": [[M_val, N_val]]}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Gemm":
        # Gemm: esegue A * B + C
        M_val = random.randint(2,10)
        K_val = random.randint(2,10)
        N_val = random.randint(2,10)
        A_shape = [M_val, K_val]
        B_shape = [K_val, N_val]
        C_shape = [M_val, N_val]  # C must be broadcastable to (M,N)
        
        A_data = np.random.randn(*A_shape).astype(np.float32)
        B_data = np.random.randn(*B_shape).astype(np.float32)
        C_data = np.random.randn(*C_shape).astype(np.float32)
        
        init_tensor_A = helper.make_tensor(input_names[0], TensorProto.FLOAT, A_shape, A_data.flatten().tolist())
        init_tensor_B = helper.make_tensor(input_names[1], TensorProto.FLOAT, B_shape, B_data.flatten().tolist())
        init_tensor_C = helper.make_tensor(input_names[2], TensorProto.FLOAT, C_shape, C_data.flatten().tolist())
        initializers.extend([init_tensor_A, init_tensor_B, init_tensor_C])
        
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, [M_val, N_val])
        
        alpha = round(random.uniform(0.5, 2.0), 2)
        beta = round(random.uniform(0.5, 2.0), 2)
        
        # Fix: Don't use transA/transB to avoid dimension mismatches
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1], input_names[2]], outputs=[output_names[0]],
                                alpha=alpha, beta=beta, 
                                name=f"{op_name}node_alpha{alpha}beta{beta}")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, A_shape)
        metadata = {"input_shapes": [A_shape, B_shape, C_shape], "output_shapes": [[M_val, N_val]],
                    "alpha": alpha, "beta": beta}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "MaxPool":
        # Pooling layer with valid kernel, stride, and padding values
        N = 1
        C = random.randint(1,4)
        H = random.randint(10,50)
        W = random.randint(10,50)
        input_shape = [N, C, H, W]
        data = np.random.randn(*input_shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, input_shape, data.flatten().tolist())
        initializers.append(init_tensor)
        
        # First define kernel size
        kernel_size = random.randint(2, max(2, min(H, W)//4))
        kernel_shape = [kernel_size, kernel_size]
        
        # Ensure padding is smaller than kernel size
        pad_h = random.randint(0, kernel_size - 1)
        pad_w = random.randint(0, kernel_size - 1)
        pads = [pad_h, pad_w, pad_h, pad_w]  # [pad_top, pad_left, pad_bottom, pad_right]
        
        # Define reasonable strides
        strides = [random.randint(1, kernel_size), random.randint(1, kernel_size)]
        
        # Calculate output dimensions with padding
        H_out = ((H + 2*pad_h - kernel_size) // strides[0]) + 1
        W_out = ((W + 2*pad_w - kernel_size) // strides[1]) + 1
        
        # If output dimensions are invalid, adjust parameters
        if H_out <= 0 or W_out <= 0:
            # Use minimal valid values
            pad_h = pad_w = 0
            pads = [pad_h, pad_w, pad_h, pad_w]
            strides = [1, 1]
            H_out = ((H + 2*pad_h - kernel_size) // strides[0]) + 1
            W_out = ((W + 2*pad_w - kernel_size) // strides[1]) + 1
        
        output_shape = [N, C, H_out, W_out]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
        
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]],
                              kernel_shape=kernel_shape, 
                              strides=strides, 
                              pads=pads,
                              auto_pad="NOTSET",
                              name=f"{op_name}node_k{kernel_shape}_s{strides}_p{pads}")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
        metadata = {
            "input_shapes": [input_shape], 
            "output_shapes": [output_shape],
            "kernel_shape": kernel_shape, 
            "strides": strides, 
            "pads": pads,
            "auto_pad": "NOTSET"
        }
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Shape":
        # Since Shape is causing issues, let's implement it as a Cast operation instead
        # This will convert a float tensor to int64, which is simpler but still tests int64 output
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        
        # Output will have the same shape but INT64 type
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.INT64, shape)
        
        # Use Cast instead of Shape
        node = helper.make_node("Cast", inputs=[input_names[0]], outputs=[output_names[0]], 
                              to=TensorProto.INT64, name="Cast_node")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [shape], "original_op": "Shape"}
        return [input_info], output_info, [node], initializers, metadata

    else:
        # Caso di fallback per operatori non gestiti esplicitamente
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]],
                                name=f"{op_name}_generic_node")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [shape]}
        return [input_info], output_info, [node], initializers, metadata

def generate_model(op_name, filename, model_id=0):
    """Crea e salva un modello ONNX."""
    input_info, output_info, nodes, initializers, metadata = generate_fuzz_model(op_name)
    
    # Gli output vanno sempre definiti
    graph_outputs = output_info if isinstance(output_info, list) else [output_info]
    
    for node in nodes:
        node.doc_string = f"Test node for {op_name} operation with ID {model_id}"
    
    graph = helper.make_graph(
        nodes,
        name=f"{op_name}test_graph{model_id}",
        inputs=input_info,  # In questo caso vuota, poiché tutti gli input sono initializer
        outputs=graph_outputs,
        initializer=initializers,
        doc_string=f"Test graph for {op_name} operation with configuration: {metadata}"
    )
    
    opset_imports = [helper.make_opsetid("", 13)]
    
    model = helper.make_model(
        graph, 
        producer_name='zant_test_generator',
        producer_version='1.0',
        domain='ai.zant.test',
        model_version=model_id,
        doc_string=f"Test model for {op_name} operation. Generated on {datetime.datetime.now().isoformat()}",
        opset_imports=opset_imports
    )
    
    meta_prop = StringStringEntryProto()
    meta_prop.key = "test_metadata"
    meta_prop.value = str(metadata)
    model.metadata_props.append(meta_prop)
    
    meta_prop = StringStringEntryProto()
    meta_prop.key = "test_op"
    meta_prop.value = op_name
    model.metadata_props.append(meta_prop)
    
    meta_prop = StringStringEntryProto()
    meta_prop.key = "test_id"
    meta_prop.value = str(model_id)
    model.metadata_props.append(meta_prop)
    
    meta_prop = StringStringEntryProto()
    meta_prop.key = "generator_version"
    meta_prop.value = "1.0"
    model.metadata_props.append(meta_prop)
    
    onnx.checker.check_model(model)
    onnx.save(model, filename)
    print(f"Fuzzed model for {op_name} (ID: {model_id}) saved to: {filename}")
    return metadata

def run_model(filename):
    """
    Esegue il modello ONNX.
    Poiché tutti gli input sono definiti come initializer, il feed del runtime sarà vuoto.
    """
    model = onnx.load(filename)
    graph = model.graph
    # Ottieni i nomi degli initializer (gli input runtime saranno vuoti)
    initializer_names = [init.name for init in graph.initializer]
    runtime_inputs = [inp for inp in graph.input if inp.name not in initializer_names]
    
    input_data = {}
    for inp in runtime_inputs:
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        elem_type = inp.type.tensor_type.elem_type
        if elem_type == TensorProto.FLOAT:
            # Fix: Use astype(np.float32) instead of default np.float64
            data = np.random.randn(*shape).astype(np.float32)
        elif elem_type == TensorProto.INT64:
            data = np.random.randint(0, 10, size=shape, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported input type: {elem_type}")
        input_data[inp.name] = data
    
    session = ort.InferenceSession(filename)
    output_names = [out.name for out in graph.output]
    outputs = session.run(output_names, input_data)
    
    # Convert numpy arrays to lists for JSON serialization
    outputs_dict = {name: output.tolist() for name, output in zip(output_names, outputs)}
    input_data_dict = {name: data.tolist() for name, data in input_data.items()}
    
    return {"inputs": input_data_dict, "outputs": outputs_dict}

def load_supported_ops(filename="tests/CodeGen/Python-ONNX/available_operations.txt"):
    """Carica le operazioni supportate da un file oppure restituisce una lista di default."""
    try:
        with open(filename, "r") as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using default operations.")
        return [
            "LeakyRelu", "Relu", "Sigmoid", "Softmax", "Add", "Ceil", "Div", "Mul", "Sub", "Tanh",
            "Concat", "Gather", "Identity", "Neg", "Reshape", "Resize", "Shape", "Slice", 
            "Split", "Transpose", "Unsqueeze", "Mean", "Conv", "MatMul", "Gemm", "MaxPool"
        ]


def main():
    print(f"\n __main__")
    parser = argparse.ArgumentParser(description="Generate fuzzed ONNX models and save execution data in JSON.")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of models to generate for each operation.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for random generation (for reproducibility).")
    parser.add_argument("--output-dir", type=str, default="datasets/oneOpModels",
                        help="Directory to save generated models.")
    parser.add_argument("--metadata-file", type=str, default="datasets/oneOpModels/results.json",
                        help="File to save metadata and execution data.")
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    output_dir = args.output_dir
    if output_dir and not output_dir.endswith('/'):
        output_dir += '/'
    os.makedirs(output_dir, exist_ok=True)
    
    supported_ops = load_supported_ops()
    
    print(f"\n supported_ops : {supported_ops}")
    
    all_models = []
    
    for op in supported_ops:
        for i in range(args.iterations):
            filename = f"{output_dir}{op}_{i}.onnx"
            try: 
                metadata = generate_model(op, filename, i)
                print(f"Successfully generated model for {op} (ID: {i})")
            except Exception as e:
                print(f"Error generating model for {op} (ID: {i}): {e}")

            try:
                data = run_model(filename)
                model_info = {
                    "operation": op,
                    "model_id": i,
                    "inputs": data["inputs"],
                    "outputs": data["outputs"],
                    "metadata": metadata
                }
                
                test_file_name = f"{output_dir}{op}_{i}_user_tests.json"

                
                user_tests = []

                for (in_key, out_key) in zip(data["inputs"].keys(), data["outputs"].keys()):
                    in_array = np.array(data["inputs"][in_key]).flatten().tolist()
                    out_array = np.array(data["outputs"][out_key]).flatten().tolist()
                    
                    test_model_info = {
                        "name": op,
                        "type": "exact",
                        "input": in_array,
                        "output": out_array,
                        "expected_class": 0
                    }
                    user_tests.append(test_model_info)

                with open(test_file_name, 'w') as f:
                    json.dump(user_tests, f, indent=2)
                print(f"Execution data saved to {test_file_name}")
                    
                
                all_models.append(model_info)
                print(f"Successfully ran model for {op} (ID: {i})")
            except Exception as e:
                print(f"Error running model for {op} (ID: {i}): {e}")
    
    with open(args.metadata_file, 'w') as f:
        json.dump(all_models, f, indent=2)
    print(f"Execution data saved to {args.metadata_file}")

if __name__ == "__main__":
    main()