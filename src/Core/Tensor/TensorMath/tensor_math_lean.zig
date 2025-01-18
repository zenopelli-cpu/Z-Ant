// ---------------------------------------------------------------------------
// ---------------------------- importing methods ----------------------------
// ---------------------------------------------------------------------------
//
const lean_elementWise_math_lib = @import("lib_elementWise_math.zig");
//pub const add_bias = lean_elementWise_math_lib.add_bias;
pub const sum_tensors = lean_elementWise_math_lib.lean_sum_tensors;
//pub const sub_tensors = lean_elementWise_math_lib.sub_tensors;
pub const mul = lean_elementWise_math_lib.mul_lean;
pub const div = lean_elementWise_math_lib.div_lean;
