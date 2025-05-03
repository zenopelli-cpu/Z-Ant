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

// https://onnx.ai/onnx/operators/onnx__Flatten.html
// INPUTS:
//      - data (heterogeneous) - T: Input tensor of any shape.
// OUTPUTS:
//      - output (heterogeneous) - T: Output tensor with shape [outer_dim, inner_dim].
// ATTRIBUTES:
//      - axis - INT (default is '1'): Indicate up to which input dimension should be flattened.
pub const Flatten = struct {
    data: *TensorZant,
    output: *TensorZant,
    //attributes:
    axis: i64, // default = 1,

    pub fn init(nodeProto: *NodeProto) !Flatten {
        const data = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var axis: i64 = 1.0;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                if (attr.type != onnx.AttributeType.INT) {
                    return error.InvalidAttributeType;
                }
                axis = attr.i;
            }
        }

        return Flatten{
            .data = data,
            .output = output,
            .axis = axis,
        };
    }

    pub fn get_output_shape(self: Flatten) []usize { // TODO
        const res: []usize = [_]usize{ 0, 0, 1, 1 };
        res[0] += self.input;
        return res;
    }

    pub fn print(self: Flatten) void { // TODO
        std.debug.print("\n Flatten:\n {any}", .{self});
    }
};
