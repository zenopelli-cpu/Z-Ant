// ---------------------------------------------------------------------------
// ---------------- importing quantization dedicated methods -----------------
// ---------------------------------------------------------------------------
//

// ---------- importing quantization and dequantization methods ----------

//---quantize
const op_quantize = @import("op_quantize.zig");

pub const quantScheme = op_quantize.quantScheme;
pub const quantize = op_quantize.quantize;
pub const lean_quantize_minmax = op_quantize.lean_quantize_minmax;

//---dequantize
const op_dequantize = @import("op_dequantize.zig");

pub const dequantize = op_dequantize.dequantize;
pub const lean_dequantize = op_dequantize.lean_dequantize;

// ---------- importing matrix algebra methods ----------

// --- mat_mul
const quant_op_mat_mul = @import("quant_op_mat_mul.zig");
pub const quant_mat_mul = quant_op_mat_mul.quant_mat_mul;
pub const quant_mat_mul_lean = quant_op_mat_mul.quant_lean_mat_mul;
pub const quant_blocked_mat_mul = quant_op_mat_mul.quant_blocked_mat_mul;
pub const quant_blocked_mat_mul_lean = quant_op_mat_mul.quant_lean_blocked_mat_mul;
pub const get_quant_mat_mul_output_shape = quant_op_mat_mul.get_quant_mat_mul_output_shape;
