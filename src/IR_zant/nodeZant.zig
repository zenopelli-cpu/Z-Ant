const std = @import("std");
const zant = @import("zant");

const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;
const Tensor = zant.core.tensor.Tensor;
const Op_union = @import("op_union/op_union.zig").Op_union;

//--- proto structure
const NodeProto = zant.onnx.NodeProto;
const TensorProto = zant.onnx.TensorProto;
const DataType = zant.onnx.DataType;

const utils = zant.utils;

//--- zant structures
const TensorZant = @import("tensorZant.zig").TensorZant;

// --- uops ---
const UOpBuilder = zant.uops.UOpBuilder;

pub const NodeZant = struct {
    name: ?[]const u8, //name of the node
    op_type: []const u8, //onnx name of the operation, see here: https://onnx.ai/onnx/operators/
    op: Op_union, // contains all the information of the operation
    next: std.ArrayList(*NodeZant), // points to the following nodes

    fusion_list: ?std.ArrayList(*NodeZant), // This is a list of all the operations fused in this node, by deafult is null. It is only different from Null when the node represent the fusion of two or more nodes. A fused node is created in pattern_matcher.fuse_nodes().
    nodeProto: ?*NodeProto,
    ready: bool,

    /// Initializes a NodeZant instance starting from a NodeProto instance.
    pub fn init(nodeProto: *NodeProto) !NodeZant {
        return NodeZant{
            .name = if (nodeProto.name) |n| n else "unnamed",
            .op_type = nodeProto.op_type,
            .op = try Op_union.init(nodeProto),
            .next = std.ArrayList(*NodeZant).init(allocator),
            .nodeProto = nodeProto,
            .ready = false,
            .fusion_list = null,
        };
    }

    //this method will generate a NodeZant that represent the fusion of two or more nodes.
    //OSS: it assumes that the given nodes are consecutive and in the correct order
    //returns a nodeZant where:
    //   - "name" is the concatenation of all the names
    //   - "op_type" is the concatenation of "fused_" with all the other op_type divided by "_"
    //   - "op" is given by the Op_union class
    //   - "next" if the same next of the last node of the list
    pub fn init_fused(fusion_list: std.ArrayList(*NodeZant), name: ?[]const u8, op_type: ?[]const u8) !NodeZant {

        // Check if the fusion list is empty
        if (fusion_list.items.len == 0) {
            return error.EmptyFusionList;
        }

        // Create concatenated name
        var name_buffer = std.ArrayList(u8).init(allocator);
        defer name_buffer.deinit();

        // Create concatenated op_type with "fused_" prefix
        var op_type_buffer = std.ArrayList(u8).init(allocator);
        defer op_type_buffer.deinit();
        try op_type_buffer.appendSlice("fused");

        for (fusion_list.items, 0..) |node, i| {
            if (node.name) |node_name| {
                try name_buffer.appendSlice(node_name);
            } else {
                try name_buffer.appendSlice("unnamed");
            }
            // Add separator between names (except for the last one)
            if (i < fusion_list.items.len - 1) {
                try name_buffer.append('_');
            }

            try op_type_buffer.append('_');
            try op_type_buffer.appendSlice(node.op_type);
        }

        const last_node: *NodeZant = fusion_list.items[fusion_list.items.len - 1];

        return NodeZant{
            .name = name orelse name_buffer,
            .op_type = op_type orelse op_type_buffer,
            .op = try Op_union.init_fused(fusion_list, op_type orelse op_type_buffer),
            .next = last_node.next,
            .nodeProto = null,
            .ready = false,
            .fusion_list = fusion_list,
        };
    }

    /// Deinitializes a NodeZant instance, freeing allocated resources.
    pub fn deinit(self: *NodeZant) void {
        self.next.deinit();
        if (self.fusion_list) |fl| fl.deinit();
    }

    /// Adds a new node to the next list.
    pub fn add_next(self: *NodeZant, next_node: *NodeZant) !void {
        try self.next.append(next_node);
    }

    pub fn write_op(self: *NodeZant, writer: std.fs.File.Writer) !void {
        try self.op.write_op(writer);
    }

    pub fn get_output_tensors(self: *NodeZant) ![]*TensorZant {
        return try self.op.get_output_tensors();
    }

    pub fn get_input_tensors(self: *NodeZant) ![]*TensorZant {
        return try self.op.get_input_tensors();
    }

    pub fn render_lower_math_op(self: *NodeZant, builder: *UOpBuilder) !void {
        return try self.op.render_lower_math_op(builder);
    }

    pub fn isFused(self: *NodeZant) bool {
        return !self.fusion_list == null;
    }
};
