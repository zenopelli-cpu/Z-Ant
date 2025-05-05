const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("../../../zant.zig");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant = @import("../../tensorZant.zig");
const TensorZant = tensorZant.TensorZant;
const globals = @import("../../globals.zig");

// https://onnx.ai/onnx/operators/onnx__Shape.html
// INPUTS:
//      - data (heterogeneous) - T: An input tensor.
// OUTPUTS:
//      - shape (heterogeneous) - T1: Shape of the input tensor
// ATTRIBUTES:
//      - start - INT: First dimension to take
//      - end - INT: Last dimension to take
pub const Shape = struct {
    data: *TensorZant,
    shape: []usize,
    //attributes:
    start: ?i64 = null,
    end: ?i64 = null,

    pub fn init(nodeProto: *NodeProto) !Shape {
        const data = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const shape = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var start: ?i64 = null;
        var end: ?i64 = null;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "start")) {
                if (attr.type == onnx.AttributeType.INT) start = attr.i;
            } else if (std.mem.eql(u8, attr.name, "end")) {
                if (attr.type == onnx.AttributeType.INT) end = attr.i;
            }
        }

        return Shape{
            .data = data,
            .shape = shape,
            .start = start,
            .end = end,
        };
    }

    pub fn get_output_shape(self: Shape) []usize {
        return self.shape;
    }

    pub fn print(self: Shape) void {
        std.debug.print("\n Shape:\n {any}", .{self});
    }
};
