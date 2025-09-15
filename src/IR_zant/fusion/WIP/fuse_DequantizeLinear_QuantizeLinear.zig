//! in this file you will find everything you need to fuse two or more operation
//! into one.
//!
//!  More specifically for this file:
//!     DeQuantizeLinear -> QuantizeLinear
//!  is converted into:
//!     nothing

const std = @import("std");
const zant = @import("zant");
const allocator = zant.utils.allocator.allocator;

const IR_zant = @import("../IR_zant.zig");

const operators = IR_zant.operators;
const Op_union = @import("../op_union/op_union.zig").Op_union;

// Forward declaration to avoid circular dependency
pub const NodeZant = opaque {};

// // Structure to hold DequantizeLinear -> QuantizeLinear pattern results
// const DequantQuantPatternResult = struct {
//     detected: bool,
//     dequant_node: ?*NodeZant = null,
//     quant_node: ?*NodeZant = null,
//     input_quantized_tensor: ?*TensorZant = null,
//     output_quantized_tensor: ?*TensorZant = null,
// };

// // Detects the pattern: DequantizeLinear -> QuantizeLinear (redundant quantization cycle)
// fn detect_dequant_quant_pattern(nodes: []*NodeZant) !DequantQuantPatternResult {
//     if (nodes.len < 2) return DequantQuantPatternResult{ .detected = false };

//     const node1 = nodes[0]; // DequantizeLinear
//     const node2 = nodes[1]; // QuantizeLinear

//     // Check operation types
//     const is_dequant = std.mem.eql(u8, node1.op_type, "DequantizeLinear");
//     const is_quant = std.mem.eql(u8, node2.op_type, "QuantizeLinear");

//     if (!is_dequant or !is_quant) {
//         return DequantQuantPatternResult{ .detected = false };
//     }

//     // Verify tensor connectivity
//     const dequant_outputs = try node1.get_output_tensors();
//     const quant_inputs = try node2.get_input_tensors();

//     if (dequant_outputs.len == 0 or quant_inputs.len < 3) return DequantQuantPatternResult{ .detected = false };
//     if (!std.mem.eql(u8, dequant_outputs[0].name, quant_inputs[0].name)) {
//         return DequantQuantPatternResult{ .detected = false };
//     }

//     // Get input and output tensors
//     const dequant_inputs = try node1.get_input_tensors();
//     const quant_outputs = try node2.get_output_tensors();

//     if (dequant_inputs.len < 1 or quant_outputs.len < 1) return DequantQuantPatternResult{ .detected = false };

//     return DequantQuantPatternResult{
//         .detected = true,
//         .dequant_node = node1,
//         .quant_node = node2,
//         .input_quantized_tensor = dequant_inputs[0], // Original quantized tensor
//         .output_quantized_tensor = quant_outputs[0], // Redundant quantized tensor
//     };
// }

// // Writes a bypass for the redundant DequantizeLinear -> QuantizeLinear pattern
// fn write_dequant_quant_bypass(writer: std.fs.File.Writer, pattern: DequantQuantPatternResult) !void {
//     if (!pattern.detected) return;

//     if (codegen_options.comm) {
//         try writer.print(
//             \\
//             \\    // OPTIMIZED PATTERN: DequantizeLinear -> QuantizeLinear (redundant cycle elimination)
//             \\    // Bypassing unnecessary dequantization-quantization cycle
//             \\
//         , .{});
//     }

//     if (codegen_options.log) {
//         try writer.print(
//             \\
//             \\    if (log_function) |log| {{
//             \\        log(@constCast(@ptrCast("Bypassing redundant DequantizeLinear->QuantizeLinear cycle...\n")));
//             \\    }}
//             \\
//         , .{});
//     }

//     // Create alias: output tensor points to the original quantized input tensor
//     const input_name = try pattern.input_quantized_tensor.?.getNameSanitized();
//     const output_name = try pattern.output_quantized_tensor.?.getNameSanitized();
//     try writer.print("    var tensor_{s} = tensor_{s}; // Alias: bypass redundant quantization cycle\n", .{ output_name, input_name });
// }
