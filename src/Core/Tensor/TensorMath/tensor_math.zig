// ---------------------------------------------------------------------------
// ---------------------------- importing methods ----------------------------
// ---------------------------------------------------------------------------
//
// ---------- importing standard basic methods ----------
const basic_math_lib = @import("basic_math.zig");
pub const add_bias = basic_math_lib.add_bias;
pub const sum_tensors = basic_math_lib.sum_tensors;
pub const sub_tensors = basic_math_lib.sub_tensors;
pub const mul = basic_math_lib.mul;
pub const isOneHot = basic_math_lib.isOneHot;
pub const isSafe = basic_math_lib.isSafe;

// ---------- importing standard reduction and logical methods ----------
pub const equal = basic_math_lib.equal;
pub const mean = basic_math_lib.mean;

// ---------- importing standard strucutal methods ----------
const structural_math_lib = @import("structural_math.zig");
pub const concatenate = structural_math_lib.concatenate;
pub const calculateStrides = structural_math_lib.calculateStrides;
pub const transpose2D = structural_math_lib.transpose2D;
pub const transposeDefault = structural_math_lib.transposeDefault;
pub const addPaddingAndDilation = structural_math_lib.addPaddingAndDilation;
pub const flip = structural_math_lib.flip;
pub const resize = structural_math_lib.resize;

// ---------- importing matrix algebra methods ----------
const algebraic_math_lib = @import("algebraic_math.zig");
pub const compute_dot_product = algebraic_math_lib.compute_dot_product;
pub const dot_product_tensor = algebraic_math_lib.dot_product_tensor;

// ---------- importing standard convolution methods ----------
const convolution_math_lib = @import("convolution_math.zig");
pub const multidim_convolution_with_bias = convolution_math_lib.multidim_convolution_with_bias;
pub const convolve_tensor_with_bias = convolution_math_lib.convolve_tensor_with_bias;
pub const convolution_backward_biases = convolution_math_lib.convolution_backward_biases;
pub const convolution_backward_weights = convolution_math_lib.convolution_backward_weights;
pub const convolution_backward_input = convolution_math_lib.convolution_backward_input;

// ---------- importing standard pooling methods ----------
const pooling_math_lib = @import("pooling_math.zig");
pub const pool_tensor = pooling_math_lib.pool_tensor;
pub const multidim_pooling = pooling_math_lib.multidim_pooling;
pub const pool_forward = pooling_math_lib.pool_forward;
pub const pool_backward = pooling_math_lib.pool_backward;
