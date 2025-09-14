//! in this file you will find everything you need to fuse two or more operation
//! into one.
//!
//!  More specifically for this file:
//!     DequantizeLinear -> Clip -> QuantizeLinear
//!  is converted into:
//!     nothing

const std = @import("std");
const zant = @import("zant");
const allocator = zant.utils.allocator.allocator;

const IR_zant = @import("../IR_zant.zig");

const operators = IR_zant.operators;
const Op_union = @import("../op_union/op_union.zig").Op_union;

// // Structure to hold pattern detection results
// const QuantizedClipPatternResult = struct {
//     detected: bool,
//     dequant_node: ?*NodeZant = null,
//     clip_node: ?*NodeZant = null,
//     quant_node: ?*NodeZant = null,
//     input_quantized_tensor: ?*TensorZant = null,
//     input_scale_tensor: ?*TensorZant = null,
//     input_zero_point_tensor: ?*TensorZant = null,
//     output_quantized_tensor: ?*TensorZant = null,
//     output_scale_tensor: ?*TensorZant = null,
//     output_zero_point_tensor: ?*TensorZant = null,
//     min_val: f32 = 0.0,
//     max_val: f32 = 6.0,
// };

// // Detects the pattern: DequantizeLinear -> Clip -> QuantizeLinear
// fn detect_quantized_clip_pattern(nodes: []*NodeZant) !QuantizedClipPatternResult {
//     if (nodes.len < 3) return QuantizedClipPatternResult{ .detected = false };

//     const node1 = nodes[0];
//     const node2 = nodes[1];
//     const node3 = nodes[2];

//     // Support both Clip and Relu as the middle op
//     const is_dequant = std.mem.eql(u8, node1.op_type, "DequantizeLinear");
//     const is_quant = std.mem.eql(u8, node3.op_type, "QuantizeLinear");
//     const is_clip = std.mem.eql(u8, node2.op_type, "Clip");
//     const is_relu = std.mem.eql(u8, node2.op_type, "Relu");

//     if (!is_dequant or !is_quant or !(is_clip or is_relu)) {
//         return QuantizedClipPatternResult{ .detected = false };
//     }

//     // Verify tensor connectivity using node-level tensors
//     const dequant_outputs = try node1.get_output_tensors();
//     const mid_inputs = try node2.get_input_tensors();
//     const mid_outputs = try node2.get_output_tensors();
//     const quant_inputs = try node3.get_input_tensors();

//     if (dequant_outputs.len == 0 or mid_inputs.len == 0) return QuantizedClipPatternResult{ .detected = false };
//     if (!std.mem.eql(u8, dequant_outputs[0].name, mid_inputs[0].name)) {
//         return QuantizedClipPatternResult{ .detected = false };
//     }

//     if (mid_outputs.len == 0 or quant_inputs.len == 0) return QuantizedClipPatternResult{ .detected = false };
//     if (!std.mem.eql(u8, mid_outputs[0].name, quant_inputs[0].name)) {
//         return QuantizedClipPatternResult{ .detected = false };
//     }

//     // Extract bounds (Clip: read min/max; Relu: [0, +inf))
//     var min_val: f32 = 0;
//     var max_val: f32 = std.math.floatMax(f32);

//     if (is_clip) {
//         const clip_op = switch (node2.op) {
//             .clip => |op| op,
//             else => return QuantizedClipPatternResult{ .detected = false },
//         };

//         // Force lower bound to 0.0 for quantized in-place optimization semantics
//         min_val = 0.0;

//         if (clip_op.max) |max_tensor| {
//             if (max_tensor.ptr) |tensor_ptr| {
//                 max_val = tensor_ptr.f32.data[0];
//             }
//         }
//     } else {
//         // Relu: lower bound at 0, effectively no upper bound
//         min_val = 0.0;
//         max_val = std.math.floatMax(f32);
//     }

//     // Get tensors for the optimized operation (using node-level APIs)
//     const dequant_inputs = try node1.get_input_tensors();
//     const quant_outputs = try node3.get_output_tensors();
//     const quant_inputs_full = try node3.get_input_tensors();

//     if (dequant_inputs.len < 3 or quant_outputs.len < 1) return QuantizedClipPatternResult{ .detected = false };
//     if (quant_inputs_full.len < 3) return QuantizedClipPatternResult{ .detected = false };

//     return QuantizedClipPatternResult{
//         .detected = true,
//         .dequant_node = node1,
//         .clip_node = node2,
//         .quant_node = node3,
//         .input_quantized_tensor = dequant_inputs[0], // x
//         .input_scale_tensor = dequant_inputs[1], // x_scale
//         .input_zero_point_tensor = dequant_inputs[2], // x_zero_point
//         .output_quantized_tensor = quant_outputs[0], // y
//         .output_scale_tensor = quant_inputs_full[1], // y_scale
//         .output_zero_point_tensor = quant_inputs_full[2], // y_zero_point
//         .min_val = min_val,
//         .max_val = max_val,
//     };
// }

// // Writes the optimized quantized clip pattern using clip_quantized_lean
// fn write_quantized_clip_pattern(writer: std.fs.File.Writer, pattern: QuantizedClipPatternResult) !void {
//     if (!pattern.detected) return;

//     if (codegen_options.comm) {
//         try writer.print(
//             \\
//             \\    // OPTIMIZED PATTERN: DequantizeLinear -> Clip -> QuantizeLinear
//             \\    // Replaced with direct quantized clip to save memory and computation
//             \\
//         , .{});
//     }

//     if (codegen_options.log) {
//         try writer.print(
//             \\
//             \\    if (log_function) |log| {{
//             \\        log(@constCast(@ptrCast("Running optimized QuantizedClip (ReLU/Clip) operation...\n")));
//             \\    }}
//             \\
//         , .{});
//     }

//     // Note: We don't allocate intermediate tensors (dequant and clip outputs)
//     // because the optimization bypasses them entirely using clip_quantized_lean
//     // We also don't allocate the output tensor since we're doing in-place clipping

//     // Call the optimized clip_quantized_lean function
//     try Clip.write_op_quantized_pattern(
//         pattern.input_quantized_tensor.?,
//         pattern.input_scale_tensor.?,
//         pattern.input_zero_point_tensor.?,
//         pattern.output_quantized_tensor.?,
//         pattern.output_scale_tensor.?,
//         pattern.output_zero_point_tensor.?,
//         pattern.min_val,
//         pattern.max_val,
//         writer,
//     );

//     // Create an alias so subsequent operations can reference the output tensor
//     const sanitized_input_name = try pattern.input_quantized_tensor.?.getNameSanitized();
//     const sanitized_output_name = try pattern.output_quantized_tensor.?.getNameSanitized();
//     try writer.print("    var tensor_{s} = tensor_{s}; // Alias for in-place clip result\n", .{ sanitized_output_name, sanitized_input_name });

//     // Note: We don't call deinit() on the input tensor since it's still in use via the alias
//     // The alias will be deinit()'ed later when it's no longer needed
// }
