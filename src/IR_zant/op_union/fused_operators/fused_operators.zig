// Conv -> Relu
pub const Fused_Conv_Relu = @import("fused_Conv_Relu.zig").Fused_Conv_Relu;

// DequantizeLinear -> pad -> QuantizeLinear -> QLinearConv
pub const Fused_Dequant_Pad_Quant_QLinConv = @import("fused_Dequant_Pad_Quant_QLinConv.zig").Fused_Dequant_Pad_Quant_QLinConv;

// QuantizeLinear -> DequantizeLinear
pub const Fused_Quant_Dequant = @import("fused_Quant_Dequant.zig").Fused_Quant_Dequant;
