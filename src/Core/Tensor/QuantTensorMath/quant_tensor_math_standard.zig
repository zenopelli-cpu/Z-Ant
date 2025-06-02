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

// ---------------------------- DENSE OPERATIONS -----------------------------

// --- mat_mul
const quant_op_mat_mul = @import("quant_op_mat_mul.zig");
pub const quant_mat_mul = quant_op_mat_mul.quant_mat_mul;
pub const quant_mat_mul_lean = quant_op_mat_mul.quant_lean_mat_mul;
pub const quant_blocked_mat_mul = quant_op_mat_mul.quant_blocked_mat_mul;
pub const quant_blocked_mat_mul_lean = quant_op_mat_mul.quant_lean_blocked_mat_mul;
pub const get_quant_mat_mul_output_shape = quant_op_mat_mul.get_quant_mat_mul_output_shape;

// --- gemm
const quant_op_gemm = @import("quant_op_gemm.zig");
pub const quant_gemm = quant_op_gemm.quant_gemm;
pub const quant_gemm_lean = quant_op_gemm.quant_lean_gemm;


// --------------------------- ELEMENT WISE MATH -----------------------------

// --- addition
const quant_add = @import("quant_op_addition.zig");
pub const quant_sum_tensors = quant_add.quant_sum_tensors;
pub const quant_sum_tensors_lean = quant_add.quant_lean_sum_tensors;
