const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");
const IR_zant = @import("../../IR_zant.zig");

// --- zant IR---
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;
const NodeZant_lib = IR_zant.NodeZant_lib;
const NodeZant = NodeZant_lib.NodeZant;
const GraphZant = IR_zant.GraphZant;
const IR_utils = IR_zant.utils;

// --- union ---
const Op_union = @import("../op_union.zig").Op_union;
const operators = IR_zant.operators;

pub const Fused_Quant_Dequant = struct {
    op_name: []const u8,

    /// FIXED fusion initialization with proper tensor handling
    pub fn init_fused_op(fusion_list: std.ArrayList(*NodeZant)) !Fused_Quant_Dequant {
        // Validation
        if (fusion_list.items.len != 4) return error.WrongNumberOfElements;
        if (!std.mem.eql(u8, fusion_list.items[0].op_type, "DequantizeLinear")) return error.WrongOpAtPos0;
        if (!std.mem.eql(u8, fusion_list.items[1].op_type, "Pad")) return error.WrongOpAtPos1;
        if (!std.mem.eql(u8, fusion_list.items[2].op_type, "QuantizeLinear")) return error.WrongOpAtPos2;
        if (!std.mem.eql(u8, fusion_list.items[3].op_type, "QLinearConv")) return error.WrongOpAtPos3;

        // Extract operations
        const dequant_op = switch (fusion_list.items[0].op) {
            .dequantizeLinear => |d| d,
            else => return error.InvalidDequantizeLinearOperation,
        };

        const pad_op = switch (fusion_list.items[1].op) {
            .pad => |p| p,
            else => return error.InvalidPadOperation,
        };

        const quant_op = switch (fusion_list.items[2].op) {
            .quantizeLinear => |q| q,
            else => return error.InvalidQuantizeLinearOperation,
        };

        const qlinearconv_op = switch (fusion_list.items[3].op) {
            .qlinearconv => |qc| qc,
            else => return error.InvalidQLinearConvOperation,
        };

        // Create the fused QLinearConv operation
        var fused_qconv = qlinearconv_op;

        // ✅ FIX 1: Properly use original quantized input (bypass dequant->requant)
        fused_qconv.input_x = dequant_op.x;
        fused_qconv.input_x_scale = dequant_op.x_scale;
        fused_qconv.input_x_zero_point = dequant_op.x_zero_point.?;

        // ✅ FIX 2: Proper padding fusion with correct indexing
        if (pad_op.input_pads.ptr) |pad_data_AnyTensor| {
            // Get existing pads from QLinearConv (should be initialized to zeros in QLinearConv.init)
            var existing_pads: [4]i64 = .{ 0, 0, 0, 0 };
            if (qlinearconv_op.pads) |conv_pads| {
                // QLinearConv pads format: [top, left, bottom, right] for 2D
                const pads_to_copy = @min(conv_pads.len, 4);
                for (0..pads_to_copy) |i| {
                    existing_pads[i] = conv_pads[i];
                }
            }

            // Extract pad values from Pad operation
            var pad_values: [4]i64 = .{ 0, 0, 0, 0 };
            if (pad_op.input_pads.shape.len > 0) {
                switch (pad_op.input_pads.ty) {
                    .i64 => {
                        const pad_i64 = pad_data_AnyTensor.get_data_as(i64);
                        const pad_len = pad_op.input_pads.shape[0];

                        // ONNX Pad format for NCHW: [N_begin, C_begin, H_begin, W_begin, N_end, C_end, H_end, W_end]
                        if (pad_len >= 8) {
                            // Extract spatial padding only (H and W dimensions)
                            pad_values[0] = pad_i64[2]; // H_begin -> top
                            pad_values[1] = pad_i64[3]; // W_begin -> left
                            pad_values[2] = pad_i64[6]; // H_end -> bottom
                            pad_values[3] = pad_i64[7]; // W_end -> right
                        }
                    },
                    else => {
                        std.debug.print("Warning: Unsupported pad data type in fusion\n", .{});
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

            std.debug.print("Fused padding: original={any}, pad_op={any}, final={any}\n", .{ existing_pads, pad_values, final_pads });
        }

        // ✅ FIX 3: Ensure output tensor is properly set
        // The output should be the same as the original QLinearConv output
        fused_qconv.output_y = qlinearconv_op.output_y;

        return Fused_Quant_Dequant{
            .op_name = try NodeZant_lib.getFusedOpsName(fusion_list),
            .op_DequantizeLinear = dequant_op,
            .op_Pad = pad_op,
            .op_QuantizeLinear = quant_op,
            .op_QLinearConv = qlinearconv_op,
            .fused_qlinearconv = fused_qconv,
        };
    }

    /// Pattern detection function for QuantizeLinear -> DequantizeLinear
    pub fn fn_pattern_detection(graph: *GraphZant, root_node: *NodeZant) anyerror!?std.ArrayList(*NodeZant) {
        _ = graph; // Not used in this sequential pattern

        // Only start detection from DequantizeLinear nodes
        if (!std.mem.eql(u8, root_node.op_type, "QuantizeLinear")) {
            std.debug.print(" -> Not a QuantizeLinear node, skipping", .{});
            return null;
        }

        var node_list = std.ArrayList(*NodeZant).init(allocator);
        errdefer node_list.deinit();

        try node_list.append(root_node);
        std.debug.print(" -> QuantizeLinear node found, checking for DequantizeLinear successor", .{});

        // Check DequantizeLinear -> Pad
        if (root_node.next.items.len != 1) {
            std.debug.print(" -> QuantizeLinear has {} successors (expected 1)", .{root_node.next.items.len});
            node_list.deinit();
            return null;
        }

        const pad_node = root_node.next.items[0];
        if (!std.mem.eql(u8, pad_node.op_type, "DequantizeLinear")) {
            std.debug.print(" -> QuantizeLinear successor is {s} (expected DequantizeLinear)", .{pad_node.op_type});
            node_list.deinit();
            return null;
        }

        try node_list.append(pad_node);

        return node_list;
    }

    /// Pattern fusion function
    pub fn fn_pattern_fusion(graph: *GraphZant, node_list: std.ArrayList(*NodeZant)) anyerror!NodeZant {
        _ = graph; // Not used in this sequential pattern

        // Validate the pattern
        if (node_list.items.len != 4) return error.InvalidNumberOfOps;
        if (!std.mem.eql(u8, node_list.items[0].op_type, "QuantizeLinear")) return error.UnexpectedOpAtPos0;
        if (!std.mem.eql(u8, node_list.items[1].op_type, "DequantizeLinear")) return error.UnexpectedOpAtPos1;

        const last_node = node_list.items[1]; // QLinearConv node

        // Clone the next list instead of direct reference
        var cloned_next = std.ArrayList(*NodeZant).init(allocator);
        for (last_node.next.items) |next_node| {
            try cloned_next.append(next_node);
        }

        //  Clone the fusion_list instead of direct reference
        var cloned_fusion_list = std.ArrayList(*NodeZant).init(allocator);
        for (node_list.items) |node| {
            try cloned_fusion_list.append(node);
        }

        return NodeZant{
            .name = try NodeZant_lib.getFusedOpsName(node_list),
            .op_type = try NodeZant_lib.getFusedOpsType(node_list),
            .op = Op_union{ .useless = operators.Useless{} },
            .next = cloned_next,
            .nodeProto = null,
            .ready = false,
            .fusion_list = null,
        };
    }

    /// Pattern substitution function
    pub fn fn_pattern_sobstitution(graph: *GraphZant, fused_node: *NodeZant, node_list: std.ArrayList(*NodeZant)) anyerror!void {
        _ = fused_node; //the fuses node is totally useless since I removed the pattern completelly

        // Validate inputs
        if (node_list.items.len != 2) return error.InvalidPatternLength;

        const first_node = node_list.items[0]; // DequantizeLinear node
        const last_node = node_list.items[1]; // QLinearConv node

        // Step 1: Find all predecessor nodes that point to the first node
        var predecessors = std.ArrayList(*NodeZant).init(allocator);
        defer predecessors.deinit();

        for (graph.nodes.items) |node| {
            // Skip nodes that are in our pattern
            var is_pattern_node = false;
            for (node_list.items) |pattern_node| {
                if (node == pattern_node) {
                    is_pattern_node = true;
                    break;
                }
            }
            if (is_pattern_node) continue;

            // Check if this node points to our first_node
            for (node.next.items) |next_node| {
                if (next_node == first_node) {
                    try predecessors.append(node);
                    break;
                }
            }
        }

        // Step 2: Update predecessor nodes to point to the output of the last node
        for (predecessors.items) |predecessor| {
            for (predecessor.next.items, 0..) |next_node, i| {
                if (next_node == first_node) {
                    predecessor.next.items[i] = last_node.next.items[i];
                }
            }
        }

        // Step 4: Remove old nodes from graph
        var removal_count: usize = 0;
        var i: usize = node_list.items.len;
        while (i > 0) {
            i -= 1;
            const node_to_remove = node_list.items[i];

            var j: usize = 0;
            while (j < graph.nodes.items.len) {
                if (graph.nodes.items[j] == node_to_remove) {
                    _ = graph.nodes.orderedRemove(j);
                    removal_count += 1;
                    break;
                }
                j += 1;
            }
        }

        if (removal_count != node_list.items.len) {
            return error.IncompleteNodeRemoval;
        }
    }

    // Helper functions matching the Fused_Conv_Relu interface

    pub fn get_output_shape(self: Fused_Quant_Dequant) []usize {
        _ = self;
    }

    pub fn get_input_tensors(self: Fused_Quant_Dequant) anyerror![]*TensorZant {
        _ = self;
    }

    pub fn get_output_tensors(self: Fused_Quant_Dequant) anyerror![]*TensorZant {
        return try self.fused_qlinearconv.get_output_tensors();
    }

    pub fn write_op(self: Fused_Quant_Dequant, writer: std.fs.File.Writer) !void {
        _ = self;
        _ = writer;
    }

    pub fn compute_output_shape(self: Fused_Quant_Dequant) []usize {
        _ = self;
    }

    pub fn print(self: Fused_Quant_Dequant) void {
        _ = self;
    }
};
