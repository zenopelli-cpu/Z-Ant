# TEMPLATE FOR CREATING REMAINING OPERATOR FILES
# 
# Use this template to create the remaining operator files.
# Replace [OPERATOR_NAME] with the actual operator name (e.g., "Ceil", "Tanh", etc.)
# and modify the logic inside generate_[operator_name]_model function based on
# the corresponding elif block from your original file.

# import numpy as np
# import random
# from onnx import helper, TensorProto


# def generate_[operator_name]_model(input_names, output_names):
#     """
#     Generates a [OPERATOR_NAME] operator model.
#     """
#     initializers = []
    
    # Copy the logic from the corresponding elif op_name == "[OPERATOR_NAME]": block
    # from your original file and paste it here, removing the indentation
    # and making sure to return the same tuple: [input_info], output_info, [node], initializers, metadata
    
    # Example for simple single-input operators (Ceil, Tanh, Identity, Neg, Floor, Sqrt):
    # shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
    # if op_name == "Sqrt":
    #     data = np.abs(np.random.randn(*shape)).astype(np.float32)
    # else:
    #     data = np.random.randn(*shape).astype(np.float32)
    # init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    # initializers.append(init_tensor)
    # 
    # input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    # output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)
    # 
    # node = helper.make_node("[OPERATOR_NAME]", inputs=[input_names[0]], outputs=[output_names[0]], 
    #                         name=f"[OPERATOR_NAME]_node")
    # metadata = {"input_shapes": [shape], "output_shapes": [shape]}
    # return [input_info], output_info, [node], initializers, metadata

# ============================================================================
# REMAINING OPERATORS TO CREATE (based on your original file):
# ============================================================================

# SINGLE INPUT OPERATORS (copy from Relu.py and change operator name):
# - Ceil.py
# - Tanh.py  
# - Identity.py
# - Neg.py
# - Shape.py
# - Floor.py
# - Sqrt.py

# BINARY OPERATORS (copy from Add.py and change operator name):
# - Sub.py
# - Div.py
# - Mul.py

# OPERATORS WITH ATTRIBUTES:
# - Gelu.py (has approximate attribute)
# - LeakyRelu.py (has alpha attribute)

# COMPLEX OPERATORS (need full logic from original file):
# - ReduceMean.py
# - Constant.py
# - OneHot.py
# - Gather.py
# - Elu.py
# - Flatten.py
# - Pad.py
# - Resize.py
# - Slice.py
# - Split.py (creates multiple nodes)
# - Unsqueeze.py
# - Gemm.py
# - AveragePool.py
# - GlobalAveragePool.py
# - Mean.py
# - DequantizeLinear.py
# - QLinearConv.py
# - QLinearGlobalAveragePool.py (creates multiple nodes)
# - QLinearAdd.py (creates multiple nodes)
# - QLinearMatMul.py (creates multiple nodes)
# - ConvInteger.py
# - Cast.py
# - DynamicQuantizeLinear.py (returns multiple outputs)

# ============================================================================
# EXAMPLE COMPLETED FILES:
# ============================================================================

# Example for Ceil (single input operator):


# def generate_ceil_model(input_names, output_names):
#     """
#     Generates a Ceil operator model.
#     """
#     initializers = []
    
#     shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
#     data = np.random.randn(*shape).astype(np.float32)
#     init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
#     initializers.append(init_tensor)

#     input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
#     output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

#     node = helper.make_node("Ceil", inputs=[input_names[0]], outputs=[output_names[0]], 
#                             name=f"Ceil_node")
#     metadata = {"input_shapes": [shape], "output_shapes": [shape]}
#     return [input_info], output_info, [node], initializers, metadata


# # Example for Sub (binary operator):
# def generate_sub_model(input_names, output_names):
#     """
#     Generates a Sub operator model.
#     """
#     initializers = []
    
#     shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
#     data0 = np.random.randn(*shape).astype(np.float32)
#     data1 = np.random.randn(*shape).astype(np.float32)

#     init_tensor0 = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data0.flatten().tolist())
#     init_tensor1 = helper.make_tensor(input_names[1], TensorProto.FLOAT, shape, data1.flatten().tolist())
#     initializers.extend([init_tensor0, init_tensor1])

#     input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
#     output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

#     node = helper.make_node("Sub", 
#                             inputs=[input_names[0], input_names[1]], 
#                             outputs=[output_names[0]],
#                             name=f"Sub_node")
    
#     metadata = {"input_shapes": [shape, shape], "output_shapes": [shape]}
#     return [input_info], output_info, [node], initializers, metadata


# # Example for LeakyRelu (operator with attributes):
# def generate_leakyrelu_model(input_names, output_names):
#     """
#     Generates a LeakyRelu operator model.
#     """
#     initializers = []
    
#     shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
#     alpha = round(random.uniform(0.001, 0.2), 3)
#     data = np.random.randn(*shape).astype(np.float32)

#     init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
#     initializers.append(init_tensor)

#     input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
#     output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

#     node = helper.make_node("LeakyRelu", inputs=[input_names[0]], outputs=[output_names[0]], 
#                             alpha=alpha, name=f"LeakyRelu_node_alpha{alpha}")
#     metadata = {"input_shapes": [shape], "output_shapes": [shape], "alpha": alpha}
#     return [input_info], output_info, [node], initializers, metadata