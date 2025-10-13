const std = @import("std");
const zant = @import("zant");
const allocator = zant.utils.allocator.allocator;
const IR_zant = @import("../../IR_zant.zig");

// --- zant IR---
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;
const NodeZant_lib = IR_zant.NodeZant_lib;
const NodeZant = NodeZant_lib.NodeZant;
const GraphZant = IR_zant.GraphZant;
const IR_utils = IR_zant.utils; //this is IR utils

// --- union ---
const Op_union = @import("../op_union.zig").Op_union;
const operators = IR_zant.operators;

pub const Fused_Pad_Conv = struct {
    op_name: []const u8,
    op_Pad: operators.Pad,
    op_Conv: operators.Conv,
    // The resulting fused operation
    fused_pad_conv: operators.Conv,

    pub fn init_fused_op(fusion_list: std.ArrayList(*NodeZant)) !Fused_Pad_Conv {
        if (fusion_list.items.len != 2) return error.WrongNumberOfElements;
        if (!std.mem.eql(u8, fusion_list.items[0].op_type, "Pad")) return error.WrongOpAtPos0;
        if (!std.mem.eql(u8, fusion_list.items[1].op_type, "Conv")) return error.WrongOpAtPos1;

        const pad_op = switch (fusion_list.items[0].op) {
            .pad => |p| p,
            else => return error.InvalidPadOperation,
        };

        const conv_op = switch (fusion_list.items[1].op) {
            .conv => |c| c,
            else => return error.InvalidConvOperation,
        };

        var fused_padconv = conv_op;

        // TODO  fused_padcpnv.pad = ?

        return Fused_Pad_Conv{
            .op_name = try NodeZant_lib.getFusedOpsName(fusion_list),
            .op_Pad = pad_op,
            .op_Conv = conv_op,
            .fused_pad_conv = fused_padconv,
        };
    }

    // Root_node is Pad, searching for Conv successor
    pub fn fn_pattern_detection(graph: *GraphZant, root_node: *NodeZant) anyerror!?std.ArrayList(*NodeZant) {
        _ = graph; // Not used, we only look at successors nodes

        // Start detection from Pad nodes
        if (!std.mem.eql(u8, root_node.op_type, "Pad")) return null;

        // Check that the only successor is Conv
        if (root_node.next.items.len != 1) return null;
        const conv_node = root_node.next.items[0];
        if (!std.mem.eql(u8, conv_node.op_type, "Conv")) return null;

        var node_list = std.ArrayList(*NodeZant).init(allocator);
        errdefer node_list.deinit();

        // Node_list: Pad -> Conv
        try node_list.append(root_node);
        try node_list.append(conv_node);

        std.debug.print(" -> Found Pad->Conv pattern!", .{});
        return node_list;
    }

    pub fn patter_fusion(graph: *GraphZant, node_list: std.ArrayList(*NodeZant)) anyerror!NodeZant {
        _ = graph; // Not used, we only look at successors nodes

        // Check the number of nodes
        if (node_list.items.len != 2) return error.InvalidNumberOfOps;

        // Check Pattern matching: Pad->Conv
        if (!std.mem.eql(u8, node_list.items[0], "Pad")) return error.UnexpectedOpAtPos0;
        if (!std.mem.eql(u8, node_list.items[1], "Conv")) return error.UnexpectedOpAtPos1;

        const last_node = node_list.items[1];
        // Clone the next list instead of direct reference
        var cloned_next = std.ArrayList(*NodeZant).init(allocator);
        // Take the successors of the last_node, to make them successors of the fused_node
        for (last_node.next.items) |next_node| {
            try cloned_next.append(next_node);
        }

        return NodeZant{
            .name = try NodeZant_lib.getFusedOpsName(node_list),
            .op_type = try NodeZant_lib.getFusedOpsType(node_list),
            .op = Op_union{ .fused_Pad_Conv = try init_fused_op(node_list) },
            .next = cloned_next,
            .nodeProto = null,
            .ready = false,
            .is_fused = true,
        };
    }

    pub fn fn_pattern_sobstitution(graph: *GraphZant, fused_node: *NodeZant, node_list: std.ArrayList(*NodeZant)) anyerror!void {
        if (node_list.items.len != 2) return error.InvalidPatternLength;

        const first_node = node_list.items[0]; // Pad node
        const last_node = node_list.items[1]; // Conv node

        // Find all predecessor nodes that point to the first node in the pattern
        const predecessors_first_node = try graph.get_predecessors(first_node);

        // Update predecessor nodes to point to the fused_node instead of first_node
        for (predecessors_first_node.items) |predecessor| {
            for (predecessor.next.items, 0..) |next_node, i| {
                if (next_node == first_node)
                    predecessor.next.items[i] = fused_node;
            }
        }

        // Set the successor nodes if it hasn't been done yet
        if (fused_node.next.items.len == 0) {
            for (last_node.next.items) |successor| {
                try fused_node.next.append(successor);
            }
        }

        try graph.removeNodes(node_list);
        try graph.nodes.append(fused_node);
    }

    pub fn get_output_shape(self: Fused_Pad_Conv) []usize {
        return self.fused_pad_conv.get_output_shape();
    }

    pub fn get_input_tensors(self: Fused_Pad_Conv) anyerror![]*TensorZant {
        return try self.fused_pad_conv.get_input_tensors();
    }

    pub fn get_output_tensors(self: Fused_Pad_Conv) anyerror![]*TensorZant {
        return try self.fused_pad_conv.get_output_tensors();
    }

    pub fn write_op(self: Fused_Pad_Conv, writer: std.fs.File.Writer) !void {
        self.fused_pad_conv.write_op(writer);
    }

    pub fn compute_output_shape(self: Fused_Pad_Conv) []usize {
        return self.fused_pad_conv.compute_output_shape();
    }

    pub fn print(self: Fused_Pad_Conv) void {
        std.debug.print("\n Fused_Pad_Conv:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Fused_Pad_Conv, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        // Try to substitute in the Pad operation
        self.op_Pad.sobstitute_tensors(old_tensor, new_tensor) catch {
            // If not found in Pad, try Conv operation
            self.op_Conv.sobstitute_tensors(old_tensor, new_tensor) catch {
                // Finally, try the fused result operation
                return try self.fused_pad_conv.sobstitute_tensors(old_tensor, new_tensor);
            };
        };
    }
};
