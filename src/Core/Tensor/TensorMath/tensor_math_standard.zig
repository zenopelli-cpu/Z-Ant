// ---------------------------------------------------------------------------
// ---------------------------- importing methods ----------------------------
// ---------------------------------------------------------------------------
//

// ---------- importing standard reduction and logical methods ----------

// ---------- importing standard strucutal methods ----------
const shape_math_lib = @import("lib_shape_math.zig");
pub const concatenate = shape_math_lib.concatenate;
pub const calculateStrides = shape_math_lib.calculateStrides;
pub const transpose2D = shape_math_lib.transpose2D;
pub const transposeDefault = shape_math_lib.transposeDefault;
pub const addPaddingAndDilation = shape_math_lib.addPaddingAndDilation;
pub const flip = shape_math_lib.flip;
pub const resize = shape_math_lib.resize;
pub const split = shape_math_lib.split;

// ---------- importing matrix algebra methods ----------
pub const dot_product_tensor = @import("op_dot_product.zig").dot_product_tensor;

// ---------- importing standard Convolution methods ----------
const convolution_math_lib = @import("op_convolution.zig");
pub const multidim_convolution_with_bias = convolution_math_lib.multidim_convolution_with_bias;
pub const convolve_tensor_with_bias = convolution_math_lib.convolve_tensor_with_bias;
pub const convolution_backward_biases = convolution_math_lib.convolution_backward_biases;
pub const convolution_backward_weights = convolution_math_lib.convolution_backward_weights;
pub const convolution_backward_input = convolution_math_lib.convolution_backward_input;

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
pub const sub_tensors = elementWise_math_lib.sub_tensors;
pub const mul = elementWise_math_lib.mul;
pub const div = elementWise_math_lib.div;

// ---------- importing standard basic methods ----------
const logical_math_lib = @import("lib_logical_math.zig");
pub const isOneHot = logical_math_lib.isOneHot;
pub const isSafe = logical_math_lib.isSafe;
pub const equal = logical_math_lib.equal;
