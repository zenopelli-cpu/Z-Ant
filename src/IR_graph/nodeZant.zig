const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;
const Tensor = zant.core.Tensor;

//--- proto structure
const NodeProto = zant.onnx.NodeProto;

pub const NodeZant = struct {
    name: ?[]const u8, //name of the node
    op_type: []const u8, //onnx name of the operation, see here: https://onnx.ai/onnx/operators/
    next: std.ArrayList(*NodeZant), // points to the following nodes

    nodeProto: *NodeProto,
    ready: bool,

    /// Initializes a NodeZant instance starting from a NodeProto instance.
    pub fn init(nodeProto: *NodeProto) !NodeZant {
        return NodeZant{
            .name = nodeProto.name.?,
            .op_type = nodeProto.op_type,
            .next = try std.ArrayList(*NodeZant).init(allocator),
            .nodeProto = nodeProto,
            .ready = false,
        };
    }

    /// Deinitializes a NodeZant instance, freeing allocated resources.
    pub fn deinit(self: *NodeZant) void {
        self.next.deinit();
    }
};
