//! in this file you will find everything you need to fuse two or more operation
//! into one.
//!
//!  More specifically for this file:
//!     DeQuantLinear -> Pad -> QuantLinear -> QLinearConv
//!  is converted into:
//!     QLinearConv

const std = @import("std");
const zant = @import("zant");
const allocator = zant.utils.allocator.allocator;

const IR_zant = @import("../IR_zant.zig");
const NodeZant = IR_zant.NodeZant;

const operators = IR_zant.operators;
const Op_union = @import("../op_union/op_union.zig").Op_union;

pub fn fuse_dequant_pad_qant_qLinConv(fusion_list: std.ArrayList(*NodeZant)) anyerror!NodeZant {

    //checks
    if (fusion_list.items.len != 4) return error.InvalidNumberOfOps;
    if (!std.mem.eql(u8, fusion_list.items[0].op_type, "DeQuantLinear")) return error.UnexpectedOpAtPos0;
    if (!std.mem.eql(u8, fusion_list.items[1].op_type, "Pad")) return error.UnexpectedOpAtPos1;
    if (!std.mem.eql(u8, fusion_list.items[2].op_type, "QuantLinear")) return error.UnexpectedOpAtPos2;
    if (!std.mem.eql(u8, fusion_list.items[3].op_type, "QLinearConv")) return error.UnexpectedOpAtPos3;

    const deQuantLinear_node: *NodeZant = fusion_list.items[0];
    const pad_node: *NodeZant = fusion_list.items[1];
    // const quantLinear_node: *NodeZant = fusion_list.items[2];
    const qLinearConv_node: *NodeZant = fusion_list.items[3];

    // Extract operations from the unions
    const dequant_op: operators.DequantizeLinear = switch (deQuantLinear_node.op) {
        .dequantizeLinear => |d| d,
        else => return error.InvalidDequantizeLinearNode,
    };

    const pad_op: operators.Pad = switch (pad_node.op) {
        .pad => |p| p,
        else => return error.InvalidPadNode,
    };

    // const quant_op = switch (quantLinear_node.op) {
    //     .quantizeLinear => |q| q,
    //     else => return error.InvalidQuantizeLinearNode,
    // };

    const qconv_op: operators.QLinearConv = switch (qLinearConv_node.op) {
        .qlinearconv => |qc| qc,
        else => return error.InvalidQLinearConvNode,
    };

    // Create fused QLinearConv with modified inputs and fused padding
    var fused_qconv: operators.QLinearConv = qconv_op;

    // Use the original quantized input (bypass dequant->pad->quant)
    fused_qconv.input_x = dequant_op.x;
    fused_qconv.input_x_scale = dequant_op.x_scale;
    fused_qconv.input_x_zero_point = dequant_op.x_zero_point.?;

    // Fuse padding from Pad operation into QLinearConv pads
    if (pad_op.input_pads.ptr) |pad_data_AnyTensor| {
        // Extract existing pads from QLinearConv
        var existing_pads: [4]i64 = .{ 0, 0, 0, 0 };
        if (qconv_op.pads) |conv_pads| {
            if (conv_pads.len >= 4) {
                existing_pads[0] = conv_pads[0]; // top
                existing_pads[1] = conv_pads[1]; // left
                existing_pads[2] = conv_pads[2]; // bottom
                existing_pads[3] = conv_pads[3]; // right
            }
        }

        // Extract pad values (simplified - you may need more robust type checking)
        var pad_values: [4]i64 = .{ 0, 0, 0, 0 };
        if (pad_op.input_pads.shape.len > 0) {
            const pad_len = pad_op.input_pads.shape[0];
            switch (pad_op.input_pads.ty) {
                .i64 => {
                    const pad_i64 = pad_data_AnyTensor.get_data_as(i64);
                    if (pad_len >= 4) {
                        pad_values[0] = pad_i64[0]; // H_begin -> top
                        pad_values[1] = pad_i64[1]; // W_begin -> left
                        pad_values[2] = pad_i64[2]; // H_end -> bottom
                        pad_values[3] = pad_i64[3]; // W_end -> right
                    }
                },
                // Add other type cases as needed
                else => {
                    // Handle other types or default behavior
                },
            }
        }

        // Create fused pads
        const final_pads = [4]i64{
            existing_pads[0] + pad_values[0], // top
            existing_pads[1] + pad_values[1], // left
            existing_pads[2] + pad_values[2], // bottom
            existing_pads[3] + pad_values[3], // right
        };

        // Allocate and set the fused pads
        const fused_pads = try allocator.alloc(i64, 4);
        @memcpy(fused_pads, &final_pads);
        fused_qconv.pads = fused_pads;
    }

    return NodeZant{
        .name = qLinearConv_node.name,
        .op_type = qLinearConv_node.op_type, // Indicate it's fused
        .op = Op_union{ .qlinearconv = fused_qconv },
        .next = qLinearConv_node.next,
        .nodeProto = null,
        .ready = false,
        .fusion_list = fusion_list, // Keep reference to original nodes
    };
}
