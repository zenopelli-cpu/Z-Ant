const std = @import("std");
const zant = @import("../zant.zig");

const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;
const Tensor = zant.core.tensor.Tensor;
const Op_union = @import("op_union/op_union.zig").Op_union;

//--- proto structure
const NodeProto = zant.onnx.NodeProto;
const TensorProto = zant.onnx.TensorProto;
const DataType = zant.onnx.DataType;

const utils = zant.utils;
pub const NodeZant = struct {
    name: ?[]const u8, //name of the node
    op_type: []const u8, //onnx name of the operation, see here: https://onnx.ai/onnx/operators/
    op: Op_union, // contains all the information of the operation
    next: std.ArrayList(*NodeZant), // points to the following nodes

    nodeProto: *NodeProto,
    ready: bool,

    /// Initializes a NodeZant instance starting from a NodeProto instance.
    pub fn init(nodeProto: *NodeProto) !NodeZant {
        return NodeZant{
            .name = nodeProto.name.?,
            .op_type = nodeProto.op_type,
            .op = try Op_union.init(nodeProto),
            .next = std.ArrayList(*NodeZant).init(allocator),
            .nodeProto = nodeProto,
            .ready = false,
        };
    }

    /// Deinitializes a NodeZant instance, freeing allocated resources.
    pub fn deinit(self: *NodeZant) void {
        self.next.deinit();
    }

    /// Adds a new node to the next list.
    pub fn add_next(self: *NodeZant, next_node: *NodeZant) !void {
        try self.next.append(next_node);
    }

    pub fn protoTensor2Tensor(comptime T: anytype, proto: TensorProto) !Tensor(T) {
        // Allocate shape array
        var shape = try allocator.alloc(usize, proto.dims.len);
        for (proto.dims, 0..) |dim, i| {
            if (dim < 0) {
                return error.NegativeDimension;
            }
            shape[i] = @intCast(dim);
        }
        defer allocator.free(shape);

        var tensor = try Tensor(T).init(&allocator);
        if (proto.float_data) |float_data| {
            tensor = try Tensor(T).fromArray(&allocator, float_data, shape);
        } else if (proto.int32_data) |int32_data| {
            tensor = try Tensor(T).fromArray(&allocator, int32_data, shape);
        } else if (proto.int64_data) |int64_data| {
            tensor = try Tensor(T).fromArray(&allocator, int64_data, shape);
        } else if (proto.double_data) |double_data| {
            tensor = try Tensor(T).fromArray(&allocator, double_data, shape);
        } else if (proto.uint64_data) |uint64_data| {
            tensor = try Tensor(T).fromArray(&allocator, uint64_data, shape);
        } else if (proto.uint16_data) |uint16_data| {
            tensor = try Tensor(T).fromArray(&allocator, uint16_data, shape);
        } else if (proto.raw_data) |raw_data| {
            tensor = try parseRawData(T, shape, raw_data);
        } else {
            return error.UnsupportedDataType;
        }

        return tensor;
    }

    fn parseRawData(comptime T: type, shape: []usize, raw_data: []const u8) !Tensor(T) {
        const elem_size = @sizeOf(T);
        const num_elements = raw_data.len / elem_size;

        var result = try allocator.alloc(T, num_elements);
        defer allocator.free(result);

        for (0..num_elements) |i| {
            const offset = i * elem_size;
            result[i] = std.mem.bytesToValue(T, raw_data[offset .. offset + elem_size]);
        }

        return Tensor(T).fromArray(&allocator, result, shape);
    }
};
