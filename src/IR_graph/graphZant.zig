const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;
const Tensor = zant.core.Tensor;
const NodeZant = @import("nodeZant.zig").NodeZant;

//--- proto structure
const GraphProto = zant.onnx.GraphProto;

pub const GraphZant = struct {
    name: ?[]const u8,
    nodes: std.ArrayList(*NodeZant),
    graphProto: *GraphProto,

    /// Initializes a GraphZant instance.
    pub fn init(graphProto: *GraphProto) !GraphZant {
        return GraphZant{
            .name = graphProto.name.?,
            .nodes = try std.ArrayList(*NodeZant).init(allocator),
            .graphProto = graphProto,
        };
    }

    /// Deinitializes a GraphZant instance, freeing allocated resources.
    pub fn deinit(self: *GraphZant) void {
        // Deinitialize each NodeZant in the nodes list
        for (self.nodes.items) |node| {
            node.deinit();
        }
        self.nodes.deinit();
    }
};
