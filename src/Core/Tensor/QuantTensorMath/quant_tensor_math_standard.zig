// ---------------------------------------------------------------------------
// ---------------- importing quantization dedicated methods -----------------
// ---------------------------------------------------------------------------
//

// --- mat_mul
const quant_op_mat_mul = @import("quant_op_mat_mul.zig");
pub const quant_mat_mul = quant_op_mat_mul.quant_mat_mul;
pub const quant_mat_mul_lean = quant_op_mat_mul.quant_lean_mat_mul;
pub const quant_blocked_mat_mul = quant_op_mat_mul.quant_blocked_mat_mul;
pub const quant_blocked_mat_mul_lean = quant_op_mat_mul.quant_lean_blocked_mat_mul;
pub const get_quant_mat_mul_output_shape = quant_op_mat_mul.get_quant_mat_mul_output_shape;
