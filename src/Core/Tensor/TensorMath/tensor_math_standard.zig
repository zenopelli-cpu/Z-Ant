// ---------------------------------------------------------------------------
// ---------------------------- importing methods ----------------------------
// ---------------------------------------------------------------------------
//

// ---------- importing standard reduction and logical methods ----------

// ---------- importing standard structural methods ----------
const shape_math_lib = @import("lib_shape_math.zig");
pub const concatenate = shape_math_lib.concatenate;
pub const get_concatenate_output_shape = shape_math_lib.get_concatenate_output_shape;

pub const calculateStrides = shape_math_lib.calculateStrides;
pub const transpose2D = shape_math_lib.transpose2D;
pub const transposeDefault = shape_math_lib.transposeDefault;
pub const addPaddingAndDilation = shape_math_lib.addPaddingAndDilation;
pub const flip = shape_math_lib.flip;

pub const resize = shape_math_lib.resize;
pub const get_resize_output_shape = shape_math_lib.get_resize_output_shape;

pub const split = shape_math_lib.split;
pub const get_split_output_shapes = shape_math_lib.get_split_output_shapes;

// ---------- importing matrix algebra methods ----------
const op_mat_mul = @import("op_mat_mul.zig");
pub const mat_mul = op_mat_mul.mat_mul;
pub const dot_product_tensor_flat = op_mat_mul.dot_product_tensor_flat;

// ---------- importing standard gemm method ----------
const op_gemm = @import("op_gemm.zig");
pub const gemm = op_gemm.gemm;

// ---------- importing standard Convolution methods ----------
const convolution_math_lib = @import("op_convolution.zig");
pub const multidim_convolution_with_bias = convolution_math_lib.multidim_convolution_with_bias;
pub const convolve_tensor_with_bias = convolution_math_lib.convolve_tensor_with_bias;
pub const convolution_backward_biases = convolution_math_lib.convolution_backward_biases;
pub const convolution_backward_weights = convolution_math_lib.convolution_backward_weights;
pub const convolution_backward_input = convolution_math_lib.convolution_backward_input;
pub const get_convolution_output_shape = convolution_math_lib.get_convolution_output_shape;

// ---------- importing standard Pooling methods ----------
const pooling_math_lib = @import("op_pooling.zig");
pub const pool_tensor = pooling_math_lib.pool_tensor;
pub const multidim_pooling = pooling_math_lib.multidim_pooling;
pub const pool_forward = pooling_math_lib.pool_forward;
pub const pool_backward = pooling_math_lib.pool_backward;

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

const reduction_math_lib = @import("lib_reduction_math.zig");
pub const mean = reduction_math_lib.mean;

// ---------- importing standard Element-Wise math ----------
const elementWise_math_lib = @import("lib_elementWise_math.zig");
pub const add_bias = elementWise_math_lib.add_bias;
pub const sum_tensors = elementWise_math_lib.sum_tensors;
pub const sum_tensor_list = elementWise_math_lib.sum_tensor_list;
pub const sub_tensors = elementWise_math_lib.sub_tensors;
pub const mul = elementWise_math_lib.mul;
pub const div = elementWise_math_lib.div;

// ---------- importing standard basic methods ----------
const logical_math_lib = @import("lib_logical_math.zig");
pub const isOneHot = logical_math_lib.isOneHot;
pub const isSafe = logical_math_lib.isSafe;
pub const equal = logical_math_lib.equal;

// ---------- importing standard activation function methods ----------
const activation_math_lib = @import("lib_activation_function_math.zig");
//ReLU
pub const ReLU = activation_math_lib.ReLU_standard;
pub const ReLU_backward = activation_math_lib.ReLU_backward;
//Leaky ReLU
pub const leakyReLU = activation_math_lib.leakyReLU;
pub const leakyReLU_backward = activation_math_lib.leakyReLU_backward;
//Sigmoid
pub const sigmoid = activation_math_lib.sigmoid;
pub const sigmoid_backward = activation_math_lib.sigmoid_backward;
//Softmax
pub const softmax = activation_math_lib.softmax;
pub const softmax_backward = activation_math_lib.softmax_backward;
